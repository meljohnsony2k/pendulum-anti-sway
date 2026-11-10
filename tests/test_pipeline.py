"""
Checks each step of the pipeline on synthetic data with known answers.
Run from the repo root:  python -m unittest discover -v tests
"""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from human import fit_human, min_jerk, simulate  # noqa: E402
from pendulum import G, calibrate, fit_nonlinear, simulate_pendulum  # noqa: E402
from track import COLOURS, find_ball  # noqa: E402


def synthetic_open_loop(L=0.45, c2=0.06, amp_deg=30, t_end=30, fps=60):
    t = np.arange(0, t_end, 1 / fps)
    th = simulate_pendulum(t, lambda tt: 0.0, L, 0.0, c2, np.radians(amp_deg), 0.0)
    return t, th


class TestTracker(unittest.TestCase):
    def test_finds_ball_centres(self):
        img = np.full((600, 400, 3), 200, np.uint8)
        cv2.circle(img, (150, 120), 30, (40, 230, 240), -1)      # yellow (BGR)
        cv2.circle(img, (260, 430), 32, (130, 90, 245), -1)      # pink
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hx, hy, hr = find_ball(hsv, COLOURS["hand"])
        bx, by, br = find_ball(hsv, COLOURS["bob"])
        self.assertLess(np.hypot(hx - 150, hy - 120), 1.0)
        self.assertLess(np.hypot(bx - 260, by - 430), 1.0)
        self.assertAlmostEqual(br, 32, delta=2)


class TestPlant(unittest.TestCase):
    def test_calibration_recovers_length_from_large_swings(self):
        """Period based length must be right even though the swings are up to 30 deg."""
        L = 0.45
        t, th = synthetic_open_loop(L)
        cal = calibrate(dict(t=t, theta=th, L_px=L * 1000))
        self.assertAlmostEqual(cal["L"], L, delta=0.002 * L)
        self.assertAlmostEqual(cal["m_per_px"], 0.001, delta=0.002 * 0.001)

    def test_nonlinear_fit_recovers_drag(self):
        L, c2 = 0.45, 0.06
        t, th = synthetic_open_loop(L, c2, t_end=12)
        nl = fit_nonlinear(t, th, np.zeros_like(t), t[1] - t[0], 0.44)
        self.assertAlmostEqual(nl["L"], L, delta=0.002)
        self.assertAlmostEqual(nl["c2"], c2, delta=0.003)

    def test_no_drag_no_energy_loss(self):
        t, th = synthetic_open_loop(c2=0.0, t_end=20)
        E = 0.5 * np.gradient(th, t) ** 2 * 0.45 + G * (1 - np.cos(th))
        self.assertLess(np.ptp(E[10:-10]) / E.mean(), 5e-3)     # limit set by the numerical derivative


class TestControl(unittest.TestCase):
    L, c2 = 0.45, 0.0

    def test_zv_shaper_cancels_swing(self):
        T = 2 * np.pi * np.sqrt(self.L / G)
        plain = lambda tt: min_jerk(tt, 0, 0.4, 0.5, 0.5)
        zv = lambda tt: min_jerk(tt, 0, 0.2, 0.5, 0.5) + min_jerk(tt, 0, 0.2, 0.5 + T / 2, 0.5)
        res = {}
        for name, ref in (("plain", plain), ("zv", zv)):
            t, h, b, _ = simulate(6, ref, self.L, self.c2, wa=40)
            res[name] = np.abs(b - h)[t > 3].max()
        self.assertGreater(res["plain"], 0.05)
        self.assertLess(res["zv"], 0.1 * res["plain"])

    def test_human_fit_recovers_gain_and_delay(self):
        L, c2, K, tau = 0.45, 0.06, 0.25, 0.28
        x0, xf = 1.0, 0.6
        ref = lambda tt: min_jerk(tt, x0, xf, 0.8, 0.8)
        ts, h, b, _ = simulate(6, ref, L, c2, K=K, tau=tau, wa=10, x0=x0)
        t = np.arange(0, 6, 1 / 60)
        hm = fit_human(t, np.interp(t, ts, h), np.interp(t, ts, b), L, c2)
        self.assertAlmostEqual(hm["K_p"], K, delta=0.02)
        self.assertAlmostEqual(hm["tau"], tau, delta=0.02)


if __name__ == "__main__":
    unittest.main()
