"""
Make the tracking preview GIF: video frames with the tracked balls and the bob's trail.
Run from the repo root:  python src/make_gif.py path/to/cl.mp4
"""
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(video, name="cl", every=3, width=360, crop=(0.25, 0.70)):
    d = np.genfromtxt(os.path.join(ROOT, "data", f"{name}_track.csv"), delimiter=",", names=True)
    cap = cv2.VideoCapture(video)
    frames, i, trail = [], 0, []
    while True:
        ok, f = cap.read()
        if not ok or i >= len(d):
            break
        hx, hy, bx, by = d["hand_x"][i], d["hand_y"][i], d["bob_x"][i], d["bob_y"][i]
        trail.append((int(bx), int(by)))
        if i % every == 0:
            g = f.copy()
            for p, q in zip(trail[-40:-1], trail[-39:]):
                cv2.line(g, p, q, (214, 120, 42), 6, cv2.LINE_AA)        # blue trail (BGR)
            cv2.line(g, (int(hx), int(hy)), (int(bx), int(by)), (60, 60, 60), 3, cv2.LINE_AA)
            cv2.circle(g, (int(hx), int(hy)), int(d["hand_r"][i]) + 8, (52, 104, 235), 6, cv2.LINE_AA)
            cv2.circle(g, (int(bx), int(by)), int(d["bob_r"][i]) + 8, (214, 120, 42), 6, cv2.LINE_AA)
            h = g.shape[0]
            g = g[int(crop[0] * h):int(crop[1] * h)]
            cv2.putText(g, f"t = {d['t'][i]:.1f} s", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (40, 40, 40), 4, cv2.LINE_AA)
            g = cv2.resize(g, (width, int(g.shape[0] * width / g.shape[1])), interpolation=cv2.INTER_AREA)
            frames.append(g)
        i += 1
    out = os.path.join(ROOT, "media", "tracking.gif")
    fps = 60 / every
    if shutil.which("ffmpeg"):                      # best colours: one shared palette for the whole clip
        with tempfile.TemporaryDirectory() as tmp:
            for k, g in enumerate(frames):
                cv2.imwrite(os.path.join(tmp, f"{k:04d}.png"), g)
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", os.path.join(tmp, "%04d.png"),
                            "-vf", "split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                            "-loop", "0", out], check=True)
    else:
        imgs = [Image.fromarray(cv2.cvtColor(g, cv2.COLOR_BGR2RGB)).quantize(128) for g in frames]
        imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print(f"{len(frames)} frames -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1])
