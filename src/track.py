"""
Track the yellow ball (hand / pivot) and the pink ball (pendulum bob) in a video.

Each frame: HSV colour threshold -> clean the mask -> largest blob -> centroid and radius.
Output CSV columns: t, hand_x, hand_y, bob_x, bob_y, hand_r, bob_r (pixels, image axes).

Usage:  python src/track.py video.mp4 data/out.csv
"""
import csv
import sys

import cv2
import numpy as np

# OpenCV hue runs 0..179
COLOURS = {
    "hand": [((20, 120, 120), (36, 255, 255))],                          # yellow
    "bob": [((0, 110, 110), (6, 255, 255)), ((165, 110, 110), (179, 255, 255))],  # pink, wraps past 0
}
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def find_ball(hsv, ranges, last=None):
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return last
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = stats[k, cv2.CC_STAT_AREA]
    if area < 200:
        return last
    return (float(cents[k][0]), float(cents[k][1]), float(np.sqrt(area / np.pi)))


def track(path, scale=0.5):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    rows, last = [], {"hand": None, "bob": None}
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2HSV)
        found = {}
        for name, ranges in COLOURS.items():
            found[name] = find_ball(hsv, ranges, last[name])
            last[name] = found[name]
        if found["hand"] and found["bob"]:
            h, b = found["hand"], found["bob"]
            rows.append([i / fps, h[0] / scale, h[1] / scale, b[0] / scale, b[1] / scale, h[2] / scale, b[2] / scale])
        i += 1
    return np.array(rows), fps


def main():
    src, out = sys.argv[1], sys.argv[2]
    data, fps = track(src)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "hand_x", "hand_y", "bob_x", "bob_y", "hand_r", "bob_r"])
        w.writerows(np.round(data, 3).tolist())
    print(f"{src}: {len(data)} frames at {fps:.2f} fps -> {out}")


if __name__ == "__main__":
    main()
