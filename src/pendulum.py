"""
Data loading, self calibration and pendulum (plant) models.

Coordinates: x to the right, theta = swing angle of the bob from vertical (positive = bob to
the right of the hand). Image pixels are converted to metres using the pendulum itself:
small swing period -> effective length (g T0^2 / 4 pi^2) -> metres per pixel.
"""
import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.signal import find_peaks, lsim, savgol_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = 9.81


def load(name):
    d = np.genfromtxt(os.path.join(ROOT, "data", f"{name}_track.csv"), delimiter=",", names=True)
    n = len(d["t"])
    t = np.linspace(d["t"][0], d["t"][-1], n)     # frames are evenly spaced; the CSV rounds t
    dx, dy = d["bob_x"] - d["hand_x"], d["bob_y"] - d["hand_y"]
    return dict(t=t, dt=float(t[1] - t[0]), hand_px=d["hand_x"], bob_px=d["bob_x"],
                theta=np.arctan2(dx, dy), L_px=float(np.median(np.hypot(dx, dy))),
                r_hand_px=float(np.median(d["hand_r"])), r_bob_px=float(np.median(d["bob_r"])))


def swing_periods(t, theta, t_from=3.0):
    """Period of each swing (upward zero crossings) and the swing amplitude around it."""
    m = t > t_from
    th = theta - np.median(theta[m])
    k = np.where((th[:-1] < 0) & (th[1:] >= 0) & m[:-1])[0]
    tz = t[k] - th[k] * (t[k + 1] - t[k]) / (th[k + 1] - th[k])
    amps = []
    for a, b in zip(k[:-1], k[1:]):
        amps.append(np.abs(th[a:b]).max())
    return np.diff(tz), np.array(amps), tz


def calibrate(ol):
    """Small angle period and metres per pixel from the open loop swing.

    Large swing period: T = T0 (1 + th^2/16 + 11 th^4/3072). Fitting T0 removes the amplitude effect.
    """
    P, A, _ = swing_periods(ol["t"], ol["theta"])
    corr = 1 + A**2 / 16 + 11 * A**4 / 3072
    T0 = float(np.median(P / corr))
    L = G * T0**2 / (4 * np.pi**2)
    return dict(T0=T0, L=L, m_per_px=L / ol["L_px"], periods=P, amps=A)


def smooth_derivs(x, dt, win=21):
    """Smoothed signal and its first two derivatives (Savitzky Golay)."""
    return (savgol_filter(x, win, 3), savgol_filter(x, win, 3, deriv=1, delta=dt),
            savgol_filter(x, win, 3, deriv=2, delta=dt))


# ---------------- plant model 1: linear transfer function (what the course fitted) ----------------
def linear_tf(wn, zeta):
    """bob_x / hand_x for a small swing pendulum with viscous damping at the pivot."""
    return [2 * zeta * wn, wn**2], [1, 2 * zeta * wn, wn**2]


def fit_linear(t, u, y, wn0):
    def res(p):
        num, den = linear_tf(*p)
        _, yh, _ = lsim((num, den), u - u[0], t)
        return yh + y[0] - y
    s = least_squares(res, [wn0, 0.01], bounds=([0.5 * wn0, 0], [2 * wn0, 1]))
    return dict(wn=s.x[0], zeta=s.x[1], y=y - s.fun)


# ---------------- plant model 2: nonlinear pendulum with a moving pivot ----------------
def pendulum_rhs(L, c1, c2, hand_acc):
    """theta'' = -(g/L) sin th - c1 th' - c2 th'|th'| - (x_hand''/L) cos th"""
    def f(t, s):
        th, w = s
        return [w, -G / L * np.sin(th) - c1 * w - c2 * w * abs(w) - hand_acc(t) / L * np.cos(th)]
    return f


def simulate_pendulum(t, hand_acc, L, c1, c2, th0=0.0, w0=0.0):
    sol = solve_ivp(pendulum_rhs(L, c1, c2, hand_acc), (t[0], t[-1]), [th0, w0], t_eval=t,
                    rtol=1e-8, atol=1e-10, max_step=0.01)
    return sol.y[0]


def fit_nonlinear(t, theta, hand_m, dt, L0):
    """Fit length, viscous damping c1, air drag c2 and the camera tilt offset (hanging angle in the image)."""
    _, _, acc = smooth_derivs(hand_m, dt)
    a = lambda tt: np.interp(tt, t, acc)

    def res(p):
        L, c1, c2, off = p
        return simulate_pendulum(t, a, L, c1, c2, theta[0] - off) + off - theta
    s = least_squares(res, [L0, 0.02, 0.05, 0.0], bounds=([0.8 * L0, 0, 0, -0.1], [1.2 * L0, 1, 5, 0.1]),
                      diff_step=1e-3)
    return dict(L=s.x[0], c1=s.x[1], c2=s.x[2], offset=s.x[3], theta=theta - s.fun)


def fit_percent(y, yhat):
    """NRMSE fit in %, same measure as MATLAB compare()."""
    return 100 * (1 - np.linalg.norm(y - yhat) / np.linalg.norm(y - y.mean()))
