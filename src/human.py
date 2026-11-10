"""
Closed loop model: a person moving the pendulum to a new spot while damping its swing.

    hand command  x_cmd(t) = r(t) + K [ p e(t - tau) + T_L e'(t - tau) ]   e = x_bob - x_hand, p = pos_gain
    arm           x_hand'' = wa^2 (x_cmd - x_hand) - 2 (0.7) wa x_hand'     (neuromuscular lag)
    pendulum      nonlinear, moving pivot, quadratic air drag (identified open loop)

r(t) is the intended move: a minimum jerk path from x0 to xf starting at t_r and lasting T_m.
This is the McRuer "precision model" of a human operator: gain, lead, reaction delay and a
neuromuscular lag. The same simulator also runs automatic controllers (small tau) and
open loop input shaping (K = 0).
"""
import numpy as np
from scipy.optimize import least_squares

from pendulum import G

ARM_ZETA = 0.7


def min_jerk(t, x0, xf, t_r, T_m):
    s = np.clip((t - t_r) / T_m, 0, 1)
    return x0 + (xf - x0) * (10 * s**3 - 15 * s**4 + 6 * s**5)


def simulate(t_end, ref, L, c2, K=0.0, T_L=0.0, tau=0.0, wa=12.0, x0=0.0, dt=1 / 400, c1=0.0, th0=0.0, w0=0.0, pos_gain=1.0):
    """Fixed step RK4 with a history buffer for the delay. Returns t, hand, bob, theta."""
    n = int(round(t_end / dt)) + 1
    t = np.arange(n) * dt
    nd = int(np.floor(tau / dt))
    frac = tau / dt - nd                        # fractional delay, linear interpolation in the history
    e_hist = np.zeros(n)
    ed_hist = np.zeros(n)
    s = np.array([x0, 0.0, th0, w0])           # hand x, hand v, theta, omega
    out = np.zeros((n, 3))

    def f(tt, s, cmd):
        xh, vh, th, w = s
        ah = wa**2 * (cmd - xh) - 2 * ARM_ZETA * wa * vh
        aw = -G / L * np.sin(th) - c1 * w - c2 * w * abs(w) - ah / L * np.cos(th)
        return np.array([vh, ah, w, aw])

    for i in range(n):
        xh, vh, th, w = s
        e_hist[i] = L * np.sin(th)                  # bob relative to hand
        ed_hist[i] = L * np.cos(th) * w
        j = i - nd
        if j - 1 >= 0:
            e = (1 - frac) * e_hist[j] + frac * e_hist[j - 1]
            ed = (1 - frac) * ed_hist[j] + frac * ed_hist[j - 1]
            fb = K * (pos_gain * e + T_L * ed)
        else:
            fb = 0.0
        cmd = ref(t[i]) + fb
        out[i] = (xh, xh + L * np.sin(th), th)
        if i == n - 1:
            break
        k1 = f(t[i], s, cmd)
        k2 = f(t[i] + dt / 2, s + dt / 2 * k1, cmd)
        k3 = f(t[i] + dt / 2, s + dt / 2 * k2, cmd)
        k4 = f(t[i] + dt, s + dt * k3, cmd)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, out[:, 0], out[:, 1], out[:, 2]


def fit_human(t, hand, bob, L, c2, th0=0.0, w0=0.0, wa=10.0):
    """Fit the person's feedback to the measured hand and bob paths (metres).

    With one dominant swing frequency, only the gain and phase of the person's response at that
    frequency can be identified: gain, lead and delay trade off against each other. So the model is
    kept to the most natural strategy, "move the hand toward where the bob is":
        x_cmd = r(t) + K_p * e(t - tau)
    with the arm lag fixed at a typical 0.1 s (wa = 10 rad/s). tau is then the reaction delay.
    th0, w0: the pendulum's swing at the start of the video (it isn't perfectly still).
    """
    x0, xf = float(np.median(hand[:20])), float(np.median(hand[-20:]))

    def model(p):
        K_p, tau, t_r, T_m = p
        ref = lambda tt: min_jerk(tt, x0, xf, t_r, T_m)
        ts, h, b, _ = simulate(t[-1] + 1e-9, ref, L, c2, K=K_p, T_L=0.0, tau=tau, wa=wa, x0=x0, th0=th0, w0=w0)
        return np.interp(t, ts, h), np.interp(t, ts, b)

    def res(p):
        h, b = model(p)
        return np.concatenate([h - hand, b - bob])

    lo, hi = [0.0, 0.0, 0.3, 0.2], [2.0, 0.6, 2.0, 2.0]
    best = None
    for tau0 in (0.1, 0.25, 0.45):
        s = least_squares(res, [0.2, tau0, 0.8, 0.8], bounds=(lo, hi), diff_step=1e-3)
        if best is None or s.cost < best.cost:
            best = s
    K_p, tau, t_r, T_m = best.x
    h, b = model(best.x)
    return dict(K_p=K_p, tau=tau, wa=wa, t_r=t_r, T_m=T_m, x0=x0, xf=xf, hand=h, bob=b)
