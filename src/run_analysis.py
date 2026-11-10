"""
Full analysis: calibrate, identify the pendulum, identify the person, compare control strategies.
Writes results.json and the figures in media/. Takes a few minutes.

Run from the repo root:  python src/run_analysis.py
"""
import json
import os

import numpy as np

from human import fit_human, min_jerk, simulate
from pendulum import (G, ROOT, calibrate, fit_linear, fit_nonlinear, fit_percent, load, smooth_derivs,
                      swing_periods)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
TOL = 0.01                    # settled = swing (bob relative to hand) stays within 1 cm
RHO_AIR, CD_SPHERE = 1.2, 0.47


def settle_time(t, hand, bob, t0, tol=TOL):
    idx = np.where(np.abs(bob - hand) > tol)[0]
    return float(t[idx[-1]] - t0) if len(idx) else 0.0


def shaped_ref(x0, xf, t_r, T_m, amps, times):
    return lambda tt: x0 + sum(a * min_jerk(tt, 0, xf - x0, t_r + ti, T_m) for a, ti in zip(amps, times))


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.labelcolor": INK, "axes.titlecolor": INK})
    R = {}

    # ---------- 1. calibration from the pendulum itself ----------
    ol = load("ol")
    cal = calibrate(ol)
    s = cal["m_per_px"]
    R["calibration"] = dict(small_swing_period_s=cal["T0"], length_m=cal["L"], mm_per_px=s * 1000,
                            bob_diameter_mm=2 * ol["r_bob_px"] * s * 1000,
                            hand_ball_diameter_mm=2 * ol["r_hand_px"] * s * 1000)
    print("calibration", R["calibration"])

    # ---------- 2. plant: linear transfer function vs nonlinear physics ----------
    t, hand, bob_img = ol["t"], ol["hand_px"] * s, ol["bob_px"] * s
    lin = fit_linear(t, hand, bob_img, 2 * np.pi / cal["T0"])
    nl = fit_nonlinear(t, ol["theta"], hand, ol["dt"], cal["L"])
    L, c2, off = nl["L"], nl["c2"], nl["offset"]
    bob_nl = hand + L * np.sin(nl["theta"] - off)
    A_bob = np.pi * (R["calibration"]["bob_diameter_mm"] / 2000) ** 2
    R["plant"] = dict(
        linear=dict(wn=lin["wn"], zeta=lin["zeta"], fit_pct=fit_percent(bob_img, lin["y"])),
        nonlinear=dict(L_m=L, viscous_c1=nl["c1"], drag_c2=c2, fit_pct=fit_percent(bob_img, bob_nl)),
        implied_bob_mass_g=1000 * 0.5 * RHO_AIR * CD_SPHERE * A_bob * L / c2 if c2 > 0 else None,
    )
    print("plant", R["plant"])

    fig, axs = plt.subplots(3, 1, figsize=(9, 8.4), facecolor=SURFACE,
                            gridspec_kw=dict(height_ratios=[1, 1, 1.1]))
    for ax, (a, b) in zip(axs[:2], [(1, 7), (25, 31)]):
        m = (t >= a) & (t <= b)
        style(ax)
        ax.plot(t[m], (bob_img - hand)[m] * 100, "o", ms=2.5, color=INK, alpha=0.55, label="Measured")
        ax.plot(t[m], (lin["y"] - hand)[m] * 100, color=ORANGE, lw=2,
                label=f"Linear transfer function ({R['plant']['linear']['fit_pct']:.0f}% fit)")
        ax.plot(t[m], (bob_nl - hand)[m] * 100, color=BLUE, lw=2,
                label=f"Nonlinear physics + air drag ({R['plant']['nonlinear']['fit_pct']:.0f}% fit)")
        ax.set_ylabel("Bob relative\nto hand (cm)")
        ax.set_xlim(a, b)
    axs[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, ncol=3, fontsize=9)
    axs[0].set_title("Start of the run", loc="right", color=MUTED, fontsize=9)
    axs[1].set_title("25 s later: the linear model is off in both size and timing", loc="right", color=MUTED, fontsize=9)
    axs[1].set_xlabel("Time (s)")
    ax = axs[2]
    style(ax)
    P, A = cal["periods"], cal["amps"]
    a_deg = np.linspace(0, 35, 100)
    ar = np.radians(a_deg)
    ax.plot(a_deg, cal["T0"] * (1 + ar**2 / 16 + 11 * ar**4 / 3072), color=BLUE, lw=2, label="Pendulum theory")
    ax.plot(np.degrees(A), P, "o", ms=6, color=ORANGE, markeredgecolor=SURFACE, label="Each measured swing")
    ax.axhline(cal["T0"], color=MUTED, ls="--", lw=1)
    ax.text(1, cal["T0"] + 0.002, f"Small swing period {cal['T0']:.3f} s  ->  length {cal['L'] * 100:.1f} cm",
            color=INK, fontsize=9, va="bottom")
    ax.set_xlabel("Swing amplitude (deg)")
    ax.set_ylabel("Period (s)")
    ax.set_xlim(0, 35)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.85))
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "media", "plant_model.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # ---------- 3. the person as a controller ----------
    cl = load("cl")
    tc, hand_c = cl["t"], cl["hand_px"] * s
    th = cl["theta"] - off
    bob_c = hand_c + L * np.sin(th)
    th_s, th_d, _ = smooth_derivs(th, cl["dt"])
    hm = fit_human(tc, hand_c, bob_c, L, c2, th_s[0], th_d[0])
    w = np.sqrt(G / L)
    R["human"] = dict(gain=hm["K_p"], reaction_delay_s=hm["tau"], phase_lag_deg_at_swing=np.degrees(w * hm["tau"]),
                      arm_lag_s=1 / hm["wa"], move_start_s=hm["t_r"], move_duration_s=hm["T_m"],
                      move_distance_m=hm["x0"] - hm["xf"],
                      fit_hand_pct=fit_percent(hand_c, hm["hand"]),
                      fit_swing_pct=fit_percent(bob_c - hand_c, hm["bob"] - hm["hand"]),
                      measured_settle_s=settle_time(tc, hand_c, bob_c, hm["t_r"]))
    print("human", R["human"])

    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True, facecolor=SURFACE)
    for ax, meas, mod, lab in [(axs[0], hand_c * 100, hm["hand"] * 100, "Hand position (cm)"),
                               (axs[1], (bob_c - hand_c) * 100, (hm["bob"] - hm["hand"]) * 100, "Swing: bob\nrelative to hand (cm)")]:
        style(ax)
        ax.plot(tc, meas, "o", ms=2.5, color=INK, alpha=0.55, label="Measured (video)")
        ax.plot(tc, mod, color=BLUE, lw=2, label="Human operator model")
        ax.set_ylabel(lab)
    axs[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, ncol=2)
    axs[1].set_xlabel("Time (s)")
    axs[0].set_xlim(0, tc[-1])
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "media", "human_model.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # ---------- 4. same move, different strategies ----------
    x0, xf, t_r, T_m = hm["x0"], hm["xf"], hm["t_r"], hm["T_m"]
    T = 2 * np.pi * np.sqrt(L / G)
    plain = lambda tt: min_jerk(tt, x0, xf, t_r, T_m)
    runs = {}
    ts, h, b, _ = simulate(12, plain, L, c2, x0=x0)
    runs["No control: just move the hand"] = (ts, h, b)
    runs["Human operator (measured)"] = (tc, hand_c, bob_c)

    best = None                                   # automatic feedback: same arm, 20 ms sensing delay
    for Kp in np.linspace(-1.0, 1.0, 21):
        for Kv in np.linspace(-0.4, 0.4, 21):
            ts, h, b, _ = simulate(10, plain, L, c2, K=1.0, pos_gain=Kp, T_L=Kv, tau=0.02, x0=x0)
            if not np.all(np.isfinite(b)) or np.abs(b - h)[-400:].max() > TOL:
                continue
            st = settle_time(ts, h, b, t_r)
            if best is None or st < best[0]:
                best = (st, Kp, Kv)
    _, Kp, Kv = best
    ts, h, b, _ = simulate(10, plain, L, c2, K=1.0, pos_gain=Kp, T_L=Kv, tau=0.02, x0=x0)
    runs["Feedback controller (camera, 20 ms delay)"] = (ts, h, b)
    R["feedback_controller"] = dict(K_position=Kp, K_velocity=Kv, delay_s=0.02)

    shapers = {"ZV": ([0.5, 0.5], [0, T / 2]), "ZVD": ([0.25, 0.5, 0.25], [0, T / 2, T])}
    for name, (amps, times) in shapers.items():
        ts, h, b, _ = simulate(10, shaped_ref(x0, xf, t_r, T_m, amps, times), L, c2, x0=x0)
        runs[f"Input shaping {name} (no sensor)"] = (ts, h, b)
    R["strategies"] = {}
    for k, v in runs.items():
        st = settle_time(*v, t_r)
        R["strategies"][k] = dict(settle_s=st if st < v[0][-1] - t_r - 1 else None,     # None = never settled in the run
                                  peak_swing_cm=100 * float(np.abs(v[2] - v[1]).max()))
    print("strategies", R["strategies"])

    ol_end_swing = 100 * float(np.abs(bob_img - hand - np.median(bob_img - hand))[t > t[-1] - 3].max())
    R["open_loop_swing_after_30s_cm"] = ol_end_swing
    fig, axs = plt.subplots(len(runs), 1, figsize=(9, 9.5), sharex=True, facecolor=SURFACE)
    for ax, (name, (ts, h, b)) in zip(axs, runs.items()):
        style(ax)
        sw = (b - h) * 100
        ax.fill_between([0, 12], -TOL * 100, TOL * 100, color=AQUA, alpha=0.15, lw=0)
        ax.plot(ts, sw, color=BLUE if "Human" not in name else ORANGE, lw=1.8)
        st = R["strategies"][name]["settle_s"]
        lab = f"settles in {st:.1f} s" if st is not None else f"still swinging ±{ol_end_swing:.0f} cm after 30 s (open loop video)"
        ax.set_title(f"{name}:  {lab}", loc="left", fontsize=10, color=INK)
        ax.set_ylim(-36, 36)
        ax.set_yticks([-25, 0, 25])
    axs[len(runs) // 2].set_ylabel("Swing (cm)")
    axs[-1].set_xlabel("Time (s)  —  same 39 cm move in every case; shaded band = settled (±1 cm)")
    axs[-1].set_xlim(0, 8)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "media", "strategies.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # ---------- 4b. how much reaction delay suits a "chase the bob" strategy? ----------
    sweep = {}
    for tau in (0.1, 0.2, 0.25, 0.3, 0.35, 0.4):
        best = None
        for K in np.linspace(0.05, 1.5, 30):
            ts, h, b, _ = simulate(15, plain, L, c2, K=K, tau=tau, x0=x0)
            if not np.all(np.isfinite(b)) or np.abs(b - h)[-400:].max() > TOL:
                continue
            st = settle_time(ts, h, b, t_r)
            if best is None or st < best[0]:
                best = (st, K)
        sweep[f"{tau:.2f}"] = dict(best_settle_s=best[0], best_gain=best[1]) if best else None
    R["chase_strategy_delay_sweep"] = sweep
    print("delay sweep", sweep)

    # ---------- 5. robustness of input shaping to a wrong length ----------
    errs = np.linspace(-0.3, 0.3, 13)
    resid = {}
    for name, (amps, times) in shapers.items():
        r = []
        for e in errs:
            k = np.sqrt(1 + e)                    # period scales with sqrt(length)
            ts, h, b, _ = simulate(8, shaped_ref(x0, xf, t_r, T_m, amps, [x * k for x in times]), L, c2, x0=x0)
            m = ts > t_r + times[-1] * k + T_m + 0.3
            r.append(100 * float(np.abs(b - h)[m].max()))
        resid[name] = r
    R["shaper_robustness"] = dict(length_error_pct=(errs * 100).tolist(), residual_swing_cm=resid)
    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=SURFACE)
    style(ax)
    ax.axhline(TOL * 100, color=MUTED, ls="--", lw=1)
    ax.text(-29, TOL * 100 + 0.15, "1 cm", color=MUTED, ha="left", fontsize=9)
    for name, c in (("ZV", ORANGE), ("ZVD", BLUE)):
        ax.plot(errs * 100, resid[name], "-o", color=c, lw=2, ms=5, markeredgecolor=SURFACE, label=name)
        ax.text(errs[-1] * 100 + 1, resid[name][-1], name, color=INK, va="center")
    ax.set_xlabel("Error in the pendulum length used to time the move (%)")
    ax.set_ylabel("Swing left after the move (cm)")
    ax.set_xlim(-31, 34)
    ax.set_ylim(0, None)
    fig.suptitle("Input shaping: how wrong can the length be?", color=INK, fontsize=12, x=0.08, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "media", "shaper_robustness.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)

    with open(os.path.join(ROOT, "results.json"), "w") as f:
        json.dump(R, f, indent=2, default=float)
    print("saved results.json and figures")


if __name__ == "__main__":
    main()
