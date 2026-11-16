#!/usr/bin/env python3
"""Animated 'survival / horror' Shorts (1080x1920, 30 fps): procedural cinematic scenes,
an instrument-style HUD (distance counter, live heartbeat), thermal camera views,
searchlights, water, tunnels, flashes, natural narration, sound design and captions.

Usage: python3 make_survival.py story.json out.mp4
Needs next to it: make_video.py (narration/sfx helpers) and kokoro/ (voice model).
pip: kokoro-onnx soundfile numpy pillow opencv-python-headless

story.json
{
  "title": "HOW TO HIDE FROM A DRONE",       # top banner (short, uppercase)
  "subtitle": "can you stay hidden?",         # small box under the banner
  "voice": "am_onyx", "ambience": "wind",      # rain | wind | drone
  "threat": "rotor",                           # rotor | growl | engine | static | none  (gets louder as distance drops)
  "readout": "DISTANCE", "unit": "M",          # label + unit of the big counter
  "beats": [                                   # one short narrated line each (2nd person, present tense)
    {"text": "It hunts by heat.", "scene": "thermal", "distance": 300, "danger": 0.3,
     "sfx": "heartbeat", "emph": false, "formula": "j = σT⁴"},
    ...
  ]
}
scene: forest | searchlight | thermal | water | pipe | lights | window | corridor | road | black | whiteout
       (what-if style also: street | crack | crosssection | tsunami | blast | planet | moon | space | blackhole |
        body (+ "focus": brain|heart|lungs|stomach|bones|legs|skin) | fall | city | ocean | cells | grid | endcard)

WHAT-IF STYLE ("style": "whatif", see example_whatif.json): no banner; the first beat's caption is the question.
Each beat may set "label" (big serif readout top-left, e.g. "WAVE HEIGHT"), "sublabel", "unit", "value",
"from" (counter restarts from this when the label changes), "you": true (+ "you_at": [x, y] 0..1) to show a
YOU pin, "tag" (tilted sticker text like "≈ 10 KM DOWN"). danger 0..1 drives the scene intensity. The last beat
uses scene "endcard" with story["endcard"] = {stats[], kicker, punchline, tagline, note}.
distance: value the counter moves to during the beat (counts smoothly). danger 0..1 drives heart rate,
shake, red tint and chromatic aberration. emph: caption in red. sfx: knock, phone, thunder,
heartbeat, footsteps, door, whisper, stinger (see make_video.py). formula: optional small text.
"""
import json, math, os, re, subprocess, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_video as mv  # narration (Kokoro), sfx, ambience, ASS helpers

W, H, FPS, SR = 1080, 1920, 30, mv.SR
RW, RH = 540, 960                     # scene render size (upscaled)
FONT_DIR = "/usr/share/fonts/truetype/google-fonts"
F_BOLD = os.path.join(FONT_DIR, "Poppins-Bold.ttf")
F_MED = os.path.join(FONT_DIR, "Poppins-Medium.ttf")
F_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
F_ITAL = os.path.join(FONT_DIR, "Lora-Italic-Variable.ttf")
ACCENT = (255, 138, 61)               # counter colour
RED = (228, 36, 43)


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


# ====================================================================== scene helpers
YY, XX = np.mgrid[0:RH, 0:RW].astype(np.float32)
NX, NY = XX / RW, YY / RH


def noise_tex(rng, w, h, scale, octaves=3):
    acc = np.zeros((h, w), np.float32); amp = 1.0; tot = 0.0
    for o in range(octaves):
        s = max(2, scale // (2 ** o))
        small = rng.random((h // s + 2, w // s + 2)).astype(np.float32)
        acc += amp * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC); tot += amp; amp *= 0.5
    return acc / tot


def silhouette_trees(rng, w, h, ground, n, hmin, hmax, wmin, wmax):
    im = Image.new("L", (w, h), 0); d = ImageDraw.Draw(im)
    d.polygon([(0, h)] + [(x, ground + int(10 * math.sin(x / 50 + rng.random()))) for x in range(0, w + 20, 20)] + [(w, h)], fill=255)
    for _ in range(n):
        x = int(rng.integers(0, w)); th = int(rng.integers(hmin, hmax)); tw = int(rng.integers(wmin, wmax))
        d.polygon([(x - tw, ground + 5), (x + tw, ground + 5), (x + tw // 3, ground - th), (x - tw // 3, ground - th)], fill=255)
        def br(x0, y0, ang, L, wd, dep):
            if dep > 5 or L < 6:
                return
            x1 = x0 + math.cos(ang) * L; y1 = y0 - math.sin(ang) * L
            d.line([x0, y0, x1, y1], fill=255, width=max(1, int(wd)))
            for _ in range(int(rng.integers(2, 4))):
                br(x1, y1, ang + rng.uniform(-0.7, 0.7), L * rng.uniform(0.6, 0.8), wd * 0.65, dep + 1)
        for _ in range(int(rng.integers(2, 4))):
            br(x, ground - th, math.pi / 2 + rng.uniform(-0.6, 0.6), th * 0.35, tw * 0.6, 0)
    return np.asarray(im, np.float32) / 255


class Scene:
    """Precomputes static layers once; frame(t, lt, danger) returns float32 RHxRWx3 in 0..1."""
    def __init__(self, kind, rng, focus="heart"):
        self.kind, self.rng, self.focus = kind, rng, focus
        k = kind
        if k in ("forest", "road", "lights"):
            self.ground = int(RH * 0.62)
            sky = np.zeros((RH, RW, 3), np.float32)
            g = (1 - NY) ** 1.5
            sky[..., 0] = 0.020 + 0.05 * g; sky[..., 1] = 0.028 + 0.06 * g; sky[..., 2] = 0.045 + 0.09 * g
            self.sky = sky
            self.far = silhouette_trees(rng, RW * 2, RH, self.ground - 30, 26, 120, 260, 4, 9)
            self.near = silhouette_trees(rng, RW * 2, RH, int(RH * 0.80), 7, 420, 760, 12, 26) if k == "forest" else None
            self.fog = noise_tex(rng, RW * 2, RH, 180, 4)
        elif k == "thermal":
            self.tex = noise_tex(rng, RW * 2, RH * 2, 120, 4)
            self.trees = silhouette_trees(rng, RW * 2, RH * 2, int(RH * 1.3), 30, 300, 700, 8, 18)
        elif k == "water":
            self.tex = noise_tex(rng, RW, RH * 2, 90, 3)
            self.bub = [(rng.uniform(0, RW), rng.uniform(0, RH), rng.uniform(2, 6), rng.uniform(25, 70)) for _ in range(28)]
        elif k == "window":
            self.tex = noise_tex(rng, RW, RH, 60, 3)
        if k in ("space", "planet", "blackhole", "moon"):
            self.stars = np.zeros((RH * 2, RW * 2), np.float32)
            for _ in range(900):
                x, y = int(rng.integers(0, RW * 2)), int(rng.integers(0, RH * 2)); self.stars[y, x] = rng.uniform(0.3, 1.0) ** 2
            self.stars = cv2.GaussianBlur(self.stars, (3, 3), 0.6) * 3
            self.neb = noise_tex(rng, RW * 2, RH * 2, 260, 5)
            self.surf = noise_tex(rng, 1024, 512, 60, 5)
        if k == "body":
            im = Image.new("L", (RW, RH), 0); d = ImageDraw.Draw(im)
            cx = RW // 2
            d.ellipse([cx - 38, 150, cx + 38, 240], fill=255)                       # head
            d.rounded_rectangle([cx - 70, 255, cx + 70, 520], radius=40, fill=255)    # torso
            d.rounded_rectangle([cx - 105, 265, cx - 75, 500], radius=15, fill=255)   # arms
            d.rounded_rectangle([cx + 75, 265, cx + 105, 500], radius=15, fill=255)
            d.rounded_rectangle([cx - 62, 515, cx - 8, 820], radius=22, fill=255)     # legs
            d.rounded_rectangle([cx + 8, 515, cx + 62, 820], radius=22, fill=255)
            sil = np.asarray(im, np.float32) / 255
            edge = cv2.morphologyEx(sil, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
            self.sil, self.edge = sil, cv2.GaussianBlur(edge, (0, 0), 2) * 2 + edge
        if k in ("fall", "city"):
            self.city = silhouette_trees(rng, RW * 2, RH, int(RH * 0.9), 0, 1, 2, 1, 2) * 0
            im = Image.new("L", (RW * 2, RH), 0); d = ImageDraw.Draw(im); x = 0
            win = Image.new("L", (RW * 2, RH), 0); dw = ImageDraw.Draw(win)
            while x < RW * 2:
                w_ = int(rng.integers(30, 80)); h_ = int(rng.integers(120, 420))
                d.rectangle([x, RH - h_, x + w_, RH], fill=255)
                for wy in range(RH - h_ + 8, RH - 8, 14):
                    for wx in range(x + 5, x + w_ - 5, 10):
                        if rng.random() < 0.35:
                            dw.rectangle([wx, wy, wx + 4, wy + 6], fill=255)
                x += w_ + int(rng.integers(2, 10))
            self.city = np.asarray(im, np.float32) / 255; self.win = np.asarray(win, np.float32) / 255
            self.streaks = [(rng.uniform(0, RW), rng.uniform(0, RH), rng.uniform(40, 160), rng.uniform(600, 1400)) for _ in range(60)]
        if k in ("ocean", "cells", "blast", "grid"):
            self.tex = noise_tex(rng, RW, RH * 2, 70, 4)
            self.parts = [(rng.uniform(0, RW), rng.uniform(0, RH), rng.uniform(1, 3), rng.uniform(5, 25), rng.uniform(0, 6.28)) for _ in range(120)]
        if k in ("crack", "crosssection", "street", "tsunami"):
            self.tex = noise_tex(rng, RW, RH, 40, 5)
            self.tex2 = noise_tex(rng, RW * 2, RH, 120, 4)
            # a jagged crack path (top-down) and a fault path (cross-section)
            pts = []; x = RW * 0.5
            for y in range(-20, RH + 40, 18):
                x += rng.uniform(-14, 14); pts.append((x, y))
            self.crack_pts = pts
            if k == "street":
                im = Image.new("RGB", (RW, RH), (150, 156, 166)); d = ImageDraw.Draw(im)
                vx, vy = RW * 0.5, RH * 0.40
                for yy in range(int(vy), RH):                                    # sky→haze gradient + road
                    pass
                d.polygon([(0, RH), (RW, RH), (vx + 6, vy), (vx - 6, vy)], fill=(52, 54, 58))          # road
                d.polygon([(0, RH), (RW * 0.12, RH), (vx - 6, vy), (vx - 10, vy)], fill=(96, 96, 100))  # sidewalks
                d.polygon([(RW, RH), (RW * 0.88, RH), (vx + 6, vy), (vx + 10, vy)], fill=(96, 96, 100))
                for side in (-1, 1):                                             # facades
                    xb = 0 if side < 0 else RW
                    xi = vx + side * 14
                    d.polygon([(xb, -40), (xi, vy - 60), (xi, vy + 4), (xb, RH * 0.86)], fill=(78, 76, 80) if side < 0 else (92, 88, 90))
                    for fl in range(1, 14):                                      # floor lines converge to the vanishing point
                        f_ = fl / 14
                        ya = -40 + (RH * 0.86 + 40) * f_
                        d.line([(xb, ya), (xi, vy - 60 + 64 * f_)], fill=(60, 58, 62), width=2)
                    for col in range(1, 9):                                      # window columns
                        u = 1 - (col / 9) ** 1.6
                        x_ = xb + (xi - xb) * (1 - u)
                        y_top = -40 + (vy - 60 + 40) * (1 - u); y_bot = RH * 0.86 + (vy + 4 - RH * 0.86) * (1 - u)
                        d.line([(x_, y_top), (x_, y_bot)], fill=(55, 55, 60), width=max(1, int(6 * u)))
                for k2 in range(14):                                             # lamp posts
                    u = (k2 / 14) ** 2
                    for side in (-1, 1):
                        x_ = vx + side * (RW * 0.4) * (1 - u) + side * 8
                        y_ = RH - (RH - vy) * u
                        h_ = 260 * (1 - u) + 8
                        d.line([(x_, y_), (x_, y_ - h_)], fill=(40, 40, 44), width=max(1, int(5 * (1 - u))))
                self.street = np.asarray(im, np.float32) / 255
        self.grain = [rng.normal(0, 1, (RH, RW)).astype(np.float32) for _ in range(6)]

    # --------------------------------------------------------------------------------
    def frame(self, t, lt, danger, dist_n):
        k = self.kind; r = self.rng
        if k in ("forest", "road", "lights"):
            off = int((t * 8) % RW)
            img = self.sky.copy()
            far = np.roll(self.far, -off // 2, axis=1)[:, :RW]
            img *= (1 - far[..., None] * 0.92)
            img += far[..., None] * np.array([0.035, 0.045, 0.06], np.float32)
            # moving fog band
            fog = np.roll(self.fog, -int(t * 14) % (RW * 2), axis=1)[:, :RW]
            band = np.exp(-((NY - 0.6) ** 2) / 0.01)
            img += (np.clip(fog - 0.4, 0, 1) * band * 0.35)[..., None] * np.array([0.6, 0.65, 0.75], np.float32)
            if k == "forest":
                # searchlight cone from above, sweeping; intensity grows as the threat gets closer
                cx = 0.5 + 0.35 * math.sin(t * 0.45) + 0.1 * math.sin(t * 1.3)
                width = 0.06 + 0.25 * NY
                cone = np.exp(-((NX - cx) / width) ** 2) * (0.35 + 0.65 * NY)
                lit = (0.25 + 1.4 * (1 - dist_n)) * cone
                img += lit[..., None] * np.array([0.75, 0.82, 0.95], np.float32) * (0.5 + 0.5 * band[..., None] + 0.4)
                near = np.roll(self.near, -off, axis=1)[:, :RW]
                img *= (1 - near[..., None])
            elif k in ("road", "lights"):
                gy = self.ground
                rd = np.zeros_like(NX)
                vy = (YY - gy) / (RH - gy)
                rd[gy:] = (np.abs(NX[gy:] - 0.5) < (0.02 + 0.5 * vy[gy:])).astype(np.float32)
                img[gy:] = img[gy:] * 0.4 + rd[gy:, :, None] * 0.03
                # two lights approaching: separation & glare grow as distance drops
                sep = 0.02 + 0.22 * (1 - dist_n) ** 2
                yy = gy / RH - 0.01 + 0.08 * (1 - dist_n) ** 2
                for s in (-1, 1):
                    dx = NX - (0.5 + s * sep); dy = NY - yy
                    d2 = dx * dx + (dy * 1.8) ** 2
                    glow = 0.004 / (d2 + 0.0004) * (0.3 + 1.5 * (1 - dist_n))
                    img += np.clip(glow, 0, 3)[..., None] * np.array([1.0, 0.92, 0.75], np.float32) * 0.15
                streak = np.exp(-((NY - yy) / 0.004) ** 2) * (0.2 + 0.8 * (1 - dist_n)) * np.exp(-np.abs(NX - 0.5) * 3)
                img += streak[..., None] * 0.6
        elif k == "searchlight":
            cx = 0.5 + 0.08 * math.sin(t * 0.7); cy = 0.32 + 0.05 * math.sin(t * 0.5)
            dx = NX - cx; dy = (NY - cy) * RH / RW; rr = np.sqrt(dx * dx + dy * dy) + 1e-4
            ang = np.arctan2(dy, dx)
            core = 0.06 + 0.25 * (1 - dist_n)
            rays = (0.5 + 0.5 * np.sin(ang * 22 + t * 0.6)) ** 6 * np.exp(-rr * 2.2) * (0.3 + 0.9 * (1 - dist_n))
            glow = np.exp(-(rr / core) ** 2) * 2.0 + np.exp(-rr * 4) * 0.5
            v = glow + rays
            img = np.stack([v * 0.92, v * 0.96, v * 1.0], -1) + 0.01
            # rain catching the light
            rain = (self.grain[int(t * 30) % 6] > 2.6).astype(np.float32)
            rain = cv2.blur(rain, (1, 9))
            img += (rain * np.clip(v, 0, 1) * 0.8)[..., None]
        elif k == "thermal":
            ox = int(RW / 2 + 60 * math.sin(t * 0.2)); oy = int(RH / 2 + 30 * math.cos(t * 0.17))
            tex = self.tex[oy:oy + RH, ox:ox + RW]
            trees = self.trees[oy:oy + RH, ox:ox + RW]
            v = 0.32 + 0.18 * tex - 0.12 * trees
            # the hot target (you)
            tx = 0.5 + 0.03 * math.sin(t * 0.9); ty = 0.43
            body = np.exp(-(((NX - tx) / 0.035) ** 2 + ((NY - ty) / 0.07) ** 2))
            head = np.exp(-(((NX - tx) / 0.022) ** 2 + ((NY - (ty - 0.085)) / 0.016) ** 2))
            heat = np.clip(body + head, 0, 1) * (0.85 + 0.15 * math.sin(t * 9))
            v = v + heat * (0.75 + 0.3 * danger)
            v *= 0.92 + 0.08 * np.sin(YY * 1.4 + t * 40)[..., :]       # scanlines
            img = np.repeat(v[..., None], 3, -1)
        elif k == "water":
            v = 0.02 + 0.10 * np.exp(-NY * 2.5)
            surf_y = 0.10 + 0.012 * np.sin(NX * 9 + t * 1.6) + 0.006 * np.sin(NX * 23 - t * 2.3)
            surface = np.exp(-((NY - surf_y) / 0.006) ** 2) * 1.2 + (NY < surf_y) * (0.25 + 0.2 * (1 - dist_n))
            rays = (0.5 + 0.5 * np.sin(NX * 40 + np.sin(t * 0.8 + NX * 6) * 3)) ** 5 * np.exp(-(NY - surf_y).clip(0) * 4) * (NY > surf_y)
            caust = np.roll(self.tex, -int(t * 20) % (RH * 2), axis=0)[:RH] * 0.06
            v = v + surface + rays * 0.18 + caust
            img = np.stack([v * 0.35, v * 0.85, v * 0.9], -1)
        elif k == "pipe":
            dx = NX - 0.5; dy = (NY - 0.45) * RH / RW; rr = np.sqrt(dx * dx + dy * dy) + 1e-3
            rings = (0.5 + 0.5 * np.cos(np.log(rr) * 14 - t * 2.4)) ** 6 * np.exp(-rr * 1.5) * 0.25
            hole = np.exp(-(rr / (0.05 + 0.03 * math.sin(t * 0.3))) ** 2) * (0.5 + 0.6 * (1 - dist_n))
            wall = np.clip(rr * 1.1, 0, 1) * 0.04
            v = rings + hole + wall
            img = np.stack([v * 0.8, v * 0.85, v * 1.0], -1)
        elif k == "window":
            img = np.full((RH, RW, 3), 0.015, np.float32)
            x0, x1, y0, y1 = 0.28, 0.72, 0.2, 0.55
            inwin = ((NX > x0) & (NX < x1) & (NY > y0) & (NY < y1)).astype(np.float32)
            moon = 0.22 + 0.25 * (1 - (NY - y0) / (y1 - y0)) + 0.08 * self.tex
            frame = ((np.abs(NX - 0.5) < 0.006) | (np.abs(NY - 0.375) < 0.004)).astype(np.float32)
            win = inwin * moon * (1 - frame)
            # a figure appears behind the glass as danger rises
            fig = np.exp(-(((NX - 0.58) / 0.05) ** 2 + ((NY - 0.42) / 0.12) ** 2)) + np.exp(-(((NX - 0.58) / 0.032) ** 2 + ((NY - 0.29) / 0.03) ** 2))
            win = win * (1 - np.clip(fig, 0, 1) * np.clip((danger - 0.35) * 2.2, 0, 1))
            img += win[..., None] * np.array([0.55, 0.65, 0.9], np.float32)
            spill = np.exp(-(((NX - 0.5) / 0.35) ** 2 + ((NY - 0.8) / 0.18) ** 2)) * 0.06
            img += spill[..., None] * np.array([0.55, 0.65, 0.9], np.float32)
        elif k == "corridor":
            dx = np.abs(NX - 0.5); dy = np.abs(NY - 0.45) * RH / RW
            depth = np.maximum(dx, dy * 0.8)
            v = 0.02 + 0.10 * np.exp(-depth * 6)
            lines = (np.abs((np.log(depth + 1e-3) * 6 - t * 0.6) % 1 - 0.5) < 0.04) * 0.04
            flick = 1.0 if (int(t * 7) % 9) else 0.15
            endl = np.exp(-(depth / 0.05) ** 2) * 0.9 * flick
            figure = np.exp(-(((NX - 0.5) / (0.012 + 0.03 * danger)) ** 2 + ((NY - 0.45) / (0.03 + 0.08 * danger)) ** 2))
            v = (v + lines + endl) * (1 - figure * np.clip(danger * 1.5, 0, 1))
            img = np.stack([v * 1.0, v * 0.85, v * 0.65], -1)
        elif k == "whiteout":
            dx = NX - 0.5; dy = (NY - 0.4) * RH / RW; rr = np.sqrt(dx * dx + dy * dy)
            v = 1.4 - rr * 0.9 + 0.15 * math.sin(t * 20)
            img = np.stack([v, v * 0.82, v * 0.82], -1)
        elif k in ("space", "planet", "blackhole", "moon"):
            ox = int(t * 6) % RW; oy = int(t * 3) % RH
            st = self.stars[oy:oy + RH, ox:ox + RW]
            neb = self.neb[oy:oy + RH, ox:ox + RW]
            img = np.stack([0.010 + neb * 0.05, 0.008 + neb * 0.03, 0.03 + neb * 0.08], -1) + st[..., None]
            if k in ("planet", "moon"):
                cx, cy, R = 0.5, 0.52, 0.36 + 0.04 * (1 - dist_n)
                dx = (NX - cx) * RW / RH; dy = (NY - cy) * 1.0
                rr = np.sqrt(dx * dx + dy * dy) / R
                inside = rr < 1
                z = np.sqrt(np.clip(1 - rr ** 2, 0, 1))
                lon = (np.arctan2(dx, z + 1e-6) / math.pi * 0.5 + 0.5 + t * 0.01) % 1
                lat = np.clip(dy / R * 0.5 + 0.5, 0, 0.999)
                tex = self.surf[(lat * 511).astype(int), (lon * 1023).astype(int)]
                light = np.clip(0.75 * (-dx / R) + 0.45 * z + 0.15, 0, 1)
                if k == "planet":
                    land = (tex > 0.52).astype(np.float32)
                    col = np.stack([0.05 + land * 0.18, 0.18 + land * 0.30, 0.45 - land * 0.25], -1)
                    clouds = np.clip((self.surf[(lat * 511).astype(int), ((lon + 0.3 + t * 0.004) % 1 * 1023).astype(int)] - 0.6) * 4, 0, 1)
                    col = col * (1 - clouds[..., None]) + clouds[..., None] * 0.9
                else:
                    col = np.repeat((0.35 + 0.35 * tex)[..., None], 3, -1)
                planet = col * light[..., None]
                rim = np.exp(-((rr - 1) / 0.03) ** 2) * (k == "planet") * 0.6
                img = img * (1 - inside[..., None]) + planet * inside[..., None] + rim[..., None] * np.array([0.3, 0.6, 1.0], np.float32)
            if k == "blackhole":
                dx = (NX - 0.5) * RW / RH; dy = (NY - 0.48); rr = np.sqrt(dx * dx + dy * dy)
                ang = np.arctan2(dy, dx)
                ring_r = 0.18 + 0.02 * (1 - dist_n)
                disk = np.exp(-((dy * 4.5) ** 2 + 0) ) * np.exp(-((np.abs(dx) - 0) * 0)) * np.exp(-((np.sqrt(dx * dx + (dy * 4.5) ** 2) - ring_r * 1.6) / 0.06) ** 2)
                swirl = 0.6 + 0.4 * np.sin(ang * 6 - t * 3 + rr * 30)
                halo = np.exp(-((rr - ring_r) / 0.012) ** 2) * 1.2
                hot = (disk * 2.2 * swirl + halo)
                img = img * np.clip((rr - ring_r * 0.95) * 30, 0, 1)[..., None] * (rr > ring_r * 0.95)[..., None]
                img = img + hot[..., None] * np.array([1.0, 0.62, 0.25], np.float32)
        elif k == "crack":
            base = 0.42 + 0.18 * self.tex
            img = np.stack([base, base * 0.97, base * 0.93], -1)
            prog = min(1.0, 0.15 + lt * 0.25 + (1 - dist_n) * 0.3)
            m = Image.new("L", (RW, RH), 0); d = ImageDraw.Draw(m)
            n_ = max(2, int(len(self.crack_pts) * prog))
            d.line(self.crack_pts[:n_], fill=255, width=7, joint="curve")
            cr = np.asarray(m, np.float32) / 255
            sh = cv2.GaussianBlur(cr, (0, 0), 4)
            img = img * (1 - sh[..., None] * 0.5) * (1 - cr[..., None] * 0.95)
            # pebble
            pc = np.exp(-(((NX - 0.5) / 0.06) ** 2 + ((NY - 0.45) / 0.035) ** 2))
            img = img * (1 - np.clip(pc * 3, 0, 1)[..., None]) + np.clip(pc * 3, 0, 1)[..., None] * (0.55 + 0.3 * (NY < 0.44))[..., None]
        elif k == "crosssection":
            img = np.zeros((RH, RW, 3), np.float32)
            g = 0.22
            img[:int(RH * g)] = np.array([0.18, 0.33, 0.14], np.float32)        # grass strip
            layers = [(0.22, (0.42, 0.37, 0.30)), (0.40, (0.36, 0.33, 0.29)), (0.62, (0.30, 0.28, 0.27)), (0.82, (0.24, 0.22, 0.22))]
            for y0, c in layers:
                img[int(RH * y0):] = np.array(c, np.float32)
            img *= (0.85 + 0.25 * self.tex)[..., None]
            # fault line from surface (pin) down-left
            prog = min(1.0, 0.2 + lt * 0.18 + (1 - dist_n) * 0.5)
            fx0, fy0 = 0.62, g; fx1, fy1 = 0.2, 1.05
            px = fx0 + (fx1 - fx0) * prog; py = fy0 + (fy1 - fy0) * prog
            m = Image.new("L", (RW, RH), 0); d = ImageDraw.Draw(m)
            d.line([(fx0 * RW, fy0 * RH), (px * RW, py * RH)], fill=255, width=4)
            fl = np.asarray(m, np.float32) / 255
            glow = cv2.GaussianBlur(fl, (0, 0), 10) * (0.5 + 1.5 * danger)
            img = img * (1 - fl[..., None] * 0.9) + glow[..., None] * np.array([1.0, 0.45, 0.15], np.float32)
            ring = np.exp(-((np.sqrt(((NX - px) * RW / RH) ** 2 + (NY - py) ** 2) - (lt * 0.3 % 0.5)) / 0.01) ** 2) * 0.6
            img += ring[..., None] * np.array([0.8, 0.9, 1.0], np.float32)
        elif k == "street":
            img = self.street.copy()
            hz = 0.40
            vy = np.clip((NY - hz) / (1 - hz), 0, 1)
            road = ((NY > hz) & (np.abs(NX - 0.5) < 0.012 + 0.38 * vy)).astype(np.float32)
            for s_ in (-1, 1):                                            # double yellow line
                ln = np.exp(-((NX - (0.5 + s_ * (0.002 + 0.018 * vy))) / (0.0015 + 0.006 * vy)) ** 2) * road
                img += ln[..., None] * np.array([0.85, 0.7, 0.12], np.float32)
            wet = road * (0.10 + 0.1 * self.tex) * np.flipud(img[..., 1])          # wet reflections
            img += wet[..., None] * 0.6
            haze = np.exp(-((NY - hz) / 0.12) ** 2) * 0.18
            img = img * (1 - haze[..., None]) + haze[..., None] * np.array([0.7, 0.72, 0.76], np.float32)
            sh = danger * 9
            img = np.roll(img, (int(sh * math.cos(t * 23)), int(sh * math.sin(t * 31))), (0, 1))
        elif k == "tsunami":
            h_ = 0.02 + 0.28 * (1 - dist_n)              # wave height grows as value rises (dist_n falls)
            hz = 0.45
            sky = np.stack([0.50 - 0.1 * NY, 0.54 - 0.1 * NY, 0.58 - 0.1 * NY], -1)
            wave_top = hz - h_ * (0.8 + 0.2 * np.sin(NX * 7 + t * 0.6))
            sea = (NY > wave_top).astype(np.float32)
            face = np.clip((NY - wave_top) / max(0.02, h_ + 0.01), 0, 1)
            seacol = np.stack([0.16 + 0.1 * face, 0.30 + 0.08 * face, 0.32 + 0.06 * face], -1)
            foam = np.exp(-((NY - wave_top) / 0.006) ** 2) * (h_ > 0.04)
            beach = (NY > 0.78).astype(np.float32)
            img = sky * (1 - sea[..., None]) + seacol * sea[..., None] + foam[..., None] * 0.8
            img = img * (1 - beach[..., None]) + beach[..., None] * (0.42 + 0.1 * self.tex[..., None])
        elif k == "endcard":
            img = np.zeros((RH, RW, 3), np.float32)
        elif k == "body":
            img = np.full((RH, RW, 3), 0.01, np.float32)
            gridl = ((XX % 24 < 1) | (YY % 24 < 1)).astype(np.float32) * 0.03
            img += gridl[..., None] * np.array([0.3, 0.8, 1.0], np.float32)
            scan = np.exp(-((NY - (t * 0.25 % 1.2 - 0.1)) / 0.01) ** 2) * self.sil * 0.6
            img += (self.sil * 0.05 + self.edge * 0.5 + scan)[..., None] * np.array([0.3, 0.85, 1.0], np.float32)
            focus = {"brain": (0.5, 0.2), "heart": (0.53, 0.33), "lungs": (0.5, 0.34), "stomach": (0.5, 0.45),
                     "bones": (0.5, 0.6), "legs": (0.5, 0.72), "skin": (0.5, 0.4)}.get(self.focus, (0.5, 0.35))
            pulse = 0.6 + 0.4 * math.sin(t * (4 + 6 * danger))
            glow = np.exp(-(((NX - focus[0]) / 0.07) ** 2 + ((NY - focus[1]) / 0.045) ** 2)) * pulse * (0.6 + 0.8 * danger)
            img += glow[..., None] * np.array([1.0, 0.35 + 0.4 * (1 - danger), 0.2], np.float32)
        elif k in ("fall", "city"):
            img = np.zeros((RH, RW, 3), np.float32)
            g = (1 - NY); img[..., 0] = 0.02 + 0.05 * g; img[..., 1] = 0.03 + 0.06 * g; img[..., 2] = 0.06 + 0.12 * g
            rise = 0 if k == "city" else (t * 40) % RH
            shift = int(rise * (0.4 + 1.2 * (1 - dist_n))) % RH if k == "fall" else 0
            c = np.roll(self.city, int(t * 6) % (RW * 2), 1)[:, :RW]; w_ = np.roll(self.win, int(t * 6) % (RW * 2), 1)[:, :RW]
            if k == "fall":
                c = np.roll(c, -shift, 0); w_ = np.roll(w_, -shift, 0)
            img = img * (1 - c[..., None]) + c[..., None] * 0.012 + w_[..., None] * np.array([1.0, 0.8, 0.45], np.float32) * 0.9
            if k == "fall":
                lines = np.zeros((RH, RW), np.float32)
                for (x, y, L, sp) in self.streaks:
                    yy0 = int((y - t * sp * (0.4 + 1.6 * (1 - dist_n))) % (RH + 200)) - 100
                    x0 = int(x); lines[max(0, yy0):max(0, min(RH, yy0 + int(L))), x0:x0 + 1] = 0.5
                img += cv2.GaussianBlur(lines, (3, 3), 0)[..., None] * 0.6
                # falling figure, slowly tumbling
                fig = Image.new("L", (RW, RH), 0); d = ImageDraw.Draw(fig); a = t * 0.8
                cx, cy = RW / 2, RH * 0.45
                def P(x, y):
                    return (cx + x * math.cos(a) - y * math.sin(a), cy + x * math.sin(a) + y * math.cos(a))
                d.ellipse([P(0, -70)[0] - 16, P(0, -70)[1] - 16, P(0, -70)[0] + 16, P(0, -70)[1] + 16], fill=255)
                for (x0_, y0_, x1_, y1_, w0) in ((0, -52, 0, 30, 26), (0, -40, -55, -10, 10), (0, -40, 55, -70, 10), (0, 30, -30, 95, 12), (0, 30, 35, 90, 12)):
                    d.line([P(x0_, y0_), P(x1_, y1_)], fill=255, width=w0)
                f_ = np.asarray(fig, np.float32) / 255
                rim = cv2.GaussianBlur(f_, (0, 0), 3)
                img = img * (1 - f_[..., None]) + f_[..., None] * 0.02 + (rim - f_).clip(0)[..., None] * np.array([0.5, 0.7, 1.0], np.float32) * 0.8
        elif k == "ocean":
            depth = 1 - dist_n
            v = np.exp(-NY * (1.5 + 3 * depth)) * (0.25 * (1 - depth) + 0.02)
            rays = (0.5 + 0.5 * np.sin(NX * 30 + np.sin(t * 0.5 + NX * 4) * 2)) ** 6 * np.exp(-NY * 3) * (1 - depth) * 0.3
            img = np.stack([v * 0.15 + rays * 0.3, v * 0.6 + rays * 0.8, v * 0.8 + rays * 0.9], -1)
            snow = np.zeros((RH, RW), np.float32)
            for (x, y, r_, sp, ph) in self.parts:
                yy0 = int((y + t * sp) % RH); xx0 = int((x + 8 * math.sin(t * 0.5 + ph)) % RW)
                snow[yy0, xx0] = 0.6
            img += cv2.GaussianBlur(snow, (5, 5), 1.2)[..., None] * 0.8
        elif k == "blast":
            dx = (NX - 0.5) * RW / RH; dy = NY - 0.5; rr = np.sqrt(dx * dx + dy * dy)
            ph = (lt * 0.35) % 1.0
            rings = sum(np.exp(-((rr - (ph + j * 0.33) % 1 * 0.9) / 0.012) ** 2) * (1 - ((ph + j * 0.33) % 1)) for j in range(3))
            core = np.exp(-(rr / (0.05 + 0.1 * danger)) ** 2) * (1.5 + math.sin(t * 13) * 0.2)
            heat = np.clip(1 - rr * 2, 0, 1) ** 2 * 0.3
            img = np.stack([rings * 1.0 + core + heat, rings * 0.7 + core * 0.8 + heat * 0.4, rings * 0.4 + core * 0.6], -1)
        elif k == "cells":
            m = np.roll(self.tex, -int(t * 10) % (RH * 2), 0)[:RH]
            blobs = np.clip((m - 0.55) * 8, 0, 1)
            edges = np.clip(1 - np.abs(m - 0.55) * 40, 0, 1)
            img = np.stack([0.02 + blobs * 0.15 + edges * 0.5, 0.03 + blobs * 0.08 + edges * 0.2, 0.05 + blobs * 0.3 + edges * 0.7], -1)
        elif k == "grid":
            img = np.full((RH, RW, 3), 0.015, np.float32)
            gl = ((XX % 30 < 1) | (YY % 30 < 1)).astype(np.float32) * 0.06 + ((XX % 150 < 1.5) | (YY % 150 < 1.5)).astype(np.float32) * 0.08
            img += gl[..., None] * np.array([0.3, 0.75, 1.0], np.float32)
            xs = np.linspace(0, 1, RW); prog = min(1, lt / 4)
            curve = 0.75 - 0.5 * (xs ** (1.5 + 2 * danger)) * (1 - dist_n * 0.5)
            for i_ in range(int(RW * prog) - 1):
                y_ = int(curve[i_] * RH)
                img[max(0, y_ - 2):y_ + 3, i_] += np.array([1.0, 0.75, 0.3], np.float32)
            sweep = np.exp(-((NX - prog) / 0.004) ** 2) * 0.5
            img += sweep[..., None] * np.array([0.3, 0.8, 1.0], np.float32)
        else:  # black
            pulse = 0.5 + 0.5 * math.sin(t * (2 + 6 * danger))
            dx = NX - 0.5; dy = (NY - 0.45); rr = np.sqrt(dx * dx + dy * dy)
            v = 0.02 + 0.03 * np.exp(-rr * 4) * pulse
            img = np.stack([v * 1.1, v * 0.6, v * 0.6], -1)
        # grain
        img = img + self.grain[int(t * 30) % 6][..., None] * (0.012 if k != "thermal" else 0.03)
        return img


# ====================================================================== post FX
VIG = None


def post(img_small, t, danger, flash):
    """Upscale scene, add shake, chromatic aberration, vignette, red danger tint, flashes."""
    global VIG
    img = cv2.resize(np.clip(img_small, 0, 4), (W, H), interpolation=cv2.INTER_CUBIC)
    # danger shake
    if danger > 0.45:
        a = (danger - 0.45) * 14
        sx = int(a * math.sin(t * 37)); sy = int(a * math.cos(t * 29))
        img = np.roll(img, (sy, sx), (0, 1))
    # chromatic aberration
    ca = int(1 + danger * 6 + flash * 18)
    if ca > 1:
        img[..., 0] = np.roll(img[..., 0], ca, 1); img[..., 2] = np.roll(img[..., 2], -ca, 1)
    if VIG is None:
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((x - W / 2) / (W / 2)) ** 2 * 0.9 + ((y - H * 0.45) / (H / 2)) ** 2)
        VIG = np.clip(1.25 - 0.75 * r, 0.12, 1)[..., None].astype(np.float32)
    img = img * VIG
    if danger > 0.6:
        img = img * np.array([1.0 + 0.5 * (danger - 0.6), 1 - 0.4 * (danger - 0.6), 1 - 0.4 * (danger - 0.6)], np.float32)
    if flash > 0:
        img = img + flash * 1.2
    # filmic roll-off
    img = img / (1 + 0.25 * img)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


# ====================================================================== HUD
class HUD:
    def __init__(self, story):
        self.style = story.get("style", "survival")
        self.title = story.get("title", "").upper(); self.sub = story.get("subtitle", "")
        self.box = RED if self.style != "whatif" else (30, 107, 255)
        self.acc = ACCENT if self.style != "whatif" else (90, 222, 255)
        self.label2 = story.get("readout2", "TIME").upper()
        self.hist = []
        self.label = story.get("readout", "DISTANCE").upper(); self.unit = story.get("unit", "M")
        self.f_title = font(F_BOLD, 52); self.f_sub = font(F_MED, 34)
        self.f_lab = font(F_MONO, 22); self.f_big = font(F_BOLD, 120); self.f_unit = font(F_BOLD, 36)
        self.f_small = font(F_MONO, 20); self.f_form = font(F_ITAL, 30)
        self.top = self._top()
        self.ecg = []  # history of heartbeat samples

    def _top(self):
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        if self.style == "whatif":
            return im
        fs = 52
        while d.textlength(self.title, font=font(F_BOLD, fs)) > W - 140 and fs > 30:
            fs -= 2
        self.f_title = font(F_BOLD, fs)
        tw = d.textlength(self.title, font=self.f_title)
        x0 = (W - tw) / 2 - 22; y0 = 168
        d.rounded_rectangle([x0, y0, x0 + tw + 44, y0 + 78], radius=10, fill=self.box + (255,))
        d.text((W / 2, y0 + 39), self.title, font=self.f_title, fill=(255, 255, 255), anchor="mm")
        if self.sub:
            sw = d.textlength(self.sub, font=self.f_sub)
            sx = (W - sw) / 2 - 16; sy = y0 + 92
            d.rounded_rectangle([sx, sy, sx + sw + 32, sy + 52], radius=8, fill=(255, 255, 255, 255))
            d.text((W / 2, sy + 26), self.sub, font=self.f_sub, fill=(10, 10, 10), anchor="mm")
        # bottom panel frame
        d.line([(70, 1500), (W - 70, 1500)], fill=(255, 255, 255, 70), width=2)
        d.text((72, 1520), self.label, font=self.f_lab, fill=(200, 200, 200, 200))
        d.text((W - 72, 1520), "HEART RATE" if self.style != "whatif" else self.label2, font=self.f_lab, fill=(200, 200, 200, 200), anchor="ra")
        return im

    def render_whatif(self, t, value, beat, lt, endcard):
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        serif = os.path.join(FONT_DIR, "Lora-Variable.ttf")
        if beat.get("scene") == "endcard":
            ec = endcard or {}
            y = 640; fl = font(F_MED, 36)
            for i, line in enumerate(ec.get("stats", [])):
                if lt > 0.4 + i * 0.45:
                    d.text((W / 2, y), line.upper(), font=fl, fill=(235, 235, 235, 255), anchor="mm")
                y += 54
            if lt > 0.6 + 0.45 * len(ec.get("stats", [])):
                y += 60
                d.text((W / 2, y), ec.get("kicker", "").upper(), font=font(F_MED, 26), fill=(200, 200, 200, 255), anchor="mm"); y += 70
                fp = font(os.path.join(FONT_DIR, "Poppins-Bold.ttf"), 64)
                for ln in ec.get("punchline", "").upper().split("\n"):
                    d.text((W / 2, y), ln, font=fp, fill=(255, 255, 255, 255), anchor="mm"); y += 76
                y += 40
                d.text((W / 2, y), ec.get("tagline", ""), font=font(F_ITAL, 44), fill=(235, 235, 235, 255), anchor="mm"); y += 110
                d.text((W / 2, y), ec.get("note", "").upper(), font=font(F_MONO, 20), fill=(170, 170, 170, 255), anchor="mm")
            return np.asarray(im)
        label = beat.get("label") or self.label
        sub = beat.get("sublabel", "")
        unit = beat.get("unit", self.unit)
        if label:
            txt = f"{value:,.1f}" if (abs(value) < 100 and abs(value - round(value)) > 1e-6) else f"{int(round(value)):,}"
            big = f"{label.upper()} {txt} {unit}".strip()
            fb = font(serif, 64); fs = font(F_MONO, 20)
            bw = d.textlength(big, font=fb) + 40
            d.rounded_rectangle([56, 300, 56 + bw, 300 + (120 if sub else 100)], radius=6, fill=(20, 20, 20, 150))
            if sub:
                d.text((76, 316), sub.upper(), font=fs, fill=(210, 210, 210, 230))
            d.text((76, 340 if sub else 318), big, font=fb, fill=(245, 245, 245, 255))
        if beat.get("you"):
            ux, uy = beat.get("you_at", [0.62, 0.22]); ux *= W; uy *= H
            d.text((ux, uy - 120), "YOU", font=font(F_BOLD, 54), fill=(255, 255, 255, 255), anchor="mm")
            d.ellipse([ux - 22, uy - 92, ux + 22, uy - 48], fill=(230, 40, 50, 255))
            d.polygon([(ux - 18, uy - 62), (ux + 18, uy - 62), (ux, uy - 20)], fill=(230, 40, 50, 255))
            d.ellipse([ux - 8, uy - 78, ux + 8, uy - 62], fill=(255, 255, 255, 255))
        if beat.get("tag"):
            tg = beat["tag"]; ft = font(F_BOLD, 46)
            tw2 = d.textlength(tg, font=ft)
            box = Image.new("RGBA", (int(tw2 + 60), 90), (0, 0, 0, 0)); bd = ImageDraw.Draw(box)
            bd.rounded_rectangle([0, 0, tw2 + 60, 90], radius=10, fill=(245, 240, 230, 240))
            bd.text((30, 45), tg, font=ft, fill=(20, 20, 20, 255), anchor="lm")
            box = box.rotate(-12, expand=True, resample=Image.BICUBIC)
            im.alpha_composite(box, (int(W * 0.3), int(H * 0.5)))
        return np.asarray(im)

    def render(self, t, value, bpm, danger, formula, scene, blink):
        im = self.top.copy(); d = ImageDraw.Draw(im)
        # big counter
        txt = f"{value:.1f}" if (self.style == "whatif" and abs(value) < 10 and value != int(value)) else f"{int(round(value))}"
        col = (self.acc if danger < 0.7 else RED)
        d.text((72, 1560), txt, font=self.f_big, fill=col + (255,))
        tw = d.textlength(txt, font=self.f_big)
        d.text((72 + tw + 10, 1640), self.unit, font=self.f_unit, fill=col + (255,))
        if self.style == "whatif":
            # live graph of the counter + mission timer
            self.hist.append(value)
            h = self.hist[-150:]; lo, hi = min(h), max(h)
            span = (hi - lo) or 1.0
            x0, x1, yb, yt = 560, W - 72, 1690, 1580
            pts = [(x0 + (x1 - x0) * i / max(1, len(h) - 1), yb - (v - lo) / span * (yb - yt)) for i, v in enumerate(h)]
            if len(pts) > 1:
                d.line(pts, fill=col + (255,), width=4, joint="curve")
                d.ellipse([pts[-1][0] - 6, pts[-1][1] - 6, pts[-1][0] + 6, pts[-1][1] + 6], fill=col + (255,))
            d.line([(x0, yb + 6), (x1, yb + 6)], fill=(255, 255, 255, 60), width=1)
            d.text((W - 72, 1700), f"T+ {int(t // 60):02d}:{t % 60:04.1f}", font=self.f_small, fill=(220, 220, 220, 220), anchor="ra")
        else:
            # ECG trace (scrolling)
            def beat(p):
                return (math.exp(-((p - 0.30) / 0.012) ** 2) * 1.0 - math.exp(-((p - 0.33) / 0.012) ** 2) * 0.45
                        + math.exp(-((p - 0.55) / 0.05) ** 2) * 0.18)
            pts = []
            x0, x1, yc, amp = 560, W - 72, 1640, 70 * (0.6 + 0.6 * danger)
            n = 140
            for i in range(n):
                tt = t - (n - i) / n * 2.2
                p = (tt * bpm / 60.0) % 1.0
                pts.append((x0 + (x1 - x0) * i / (n - 1), yc - beat(p) * amp))
            d.line(pts, fill=col + (255,), width=4, joint="curve")
            d.ellipse([pts[-1][0] - 6, pts[-1][1] - 6, pts[-1][0] + 6, pts[-1][1] + 6], fill=col + (255,))
            d.text((W - 72, 1700), f"{int(bpm)} BPM", font=self.f_small, fill=(220, 220, 220, 220), anchor="ra")
        if formula:
            d.text((72, 1730), formula, font=self.f_form, fill=(210, 210, 210, 200))
        # thermal camera overlay
        if scene == "thermal":
            g = (235, 235, 235, 200)
            d.text((80, 470), "WHT", font=self.f_small, fill=g); d.text((80, 496), "NFOV  x 4.0", font=self.f_small, fill=g)
            d.text((W - 80, 470), "IR 1", font=self.f_small, fill=g, anchor="ra")
            if blink:
                d.ellipse([W - 170, 498, W - 156, 512], fill=(230, 40, 40, 255))
            d.text((W - 80, 496), "REC", font=self.f_small, fill=g, anchor="ra")
            hd = int(220 + 3 * math.sin(t * 0.4))
            for k in range(-3, 4):
                x = W / 2 + k * 120 - (t * 12 % 120)
                d.line([(x, 560), (x, 580)], fill=g, width=2)
            d.text((W / 2, 530), f"{hd}", font=self.f_small, fill=g, anchor="mm")
            d.polygon([(W / 2 - 8, 548), (W / 2 + 8, 548), (W / 2, 558)], fill=g)
            cx, cy = W / 2 + 32 * math.sin(t * 0.9), int(H * 0.43)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    bx, by = cx + sx * 95, cy + sy * 150
                    d.line([(bx, by), (bx - sx * 30, by)], fill=g, width=3); d.line([(bx, by), (bx, by - sy * 30)], fill=g, width=3)
            d.line([(70, H * 0.5), (W - 70, H * 0.5)], fill=(235, 235, 235, 60), width=1)
            d.text((80, 1440), "ROW PROFILE  x", font=self.f_small, fill=g)
        return np.asarray(im)


def alpha_over(base, rgba):
    a = rgba[..., 3:4].astype(np.float32) / 255
    return (base.astype(np.float32) * (1 - a) + rgba[..., :3].astype(np.float32) * a).astype(np.uint8)


# ====================================================================== audio
def threat_sound(kind, n):
    r = np.random.default_rng(3); t = np.arange(n) / SR
    if kind == "rotor":
        chop = (0.5 + 0.5 * np.sin(2 * np.pi * 11 * t)) ** 3          # smooth blade thump, no hard gating
        base = mv.lp_fast(r.standard_normal(n), 220) * (0.25 + chop)
        thump = mv.lp_fast(np.sin(2 * np.pi * 22 * t) * chop, 120)
        return mv.norm(base + thump * 0.8 + mv.lp_fast(r.standard_normal(n), 90) * 0.5, 1.0)
    if kind == "growl":
        f = 48 + 6 * np.sin(2 * np.pi * 0.3 * t)
        g = np.sin(2 * np.pi * np.cumsum(f) / SR) * (0.6 + 0.4 * np.sin(2 * np.pi * 7 * t))
        return mv.norm(g + mv.lp_fast(r.standard_normal(n), 160) * 0.5, 1.0)
    if kind == "engine":
        f = 60 + 10 * np.sin(2 * np.pi * 0.1 * t)
        ph = 2 * np.pi * np.cumsum(f) / SR
        e = np.tanh(3 * np.sin(ph)) + 0.4 * np.sin(2 * ph)
        return mv.norm(mv.lp_fast(e, 300), 1.0)
    if kind == "static":
        return mv.norm(mv.lp_fast(mv.hp_fast(r.standard_normal(n), 800), 3000) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t)), 1.0)
    return np.zeros(n)


def heartbeat_track(bpm_curve, n):
    out = np.zeros(n); t = 0.0
    while t < n / SR - 0.5:
        i = int(t * SR); bpm = bpm_curve(t)
        for off, amp in ((0, 1.0), (0.2, 0.65)):
            m = int(0.16 * SR); tt = np.arange(m) / SR
            s = int((t + off) * SR)
            if s + m < n:
                out[s:s + m] += np.sin(2 * np.pi * 50 * tt) * np.exp(-tt * 24) * amp
        t += 60.0 / bpm
    return mv.norm(out, 1.0)


# ====================================================================== captions
def make_ass(timeline, path, whatif=False):
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Poppins,74,&H00FFFFFF,&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,2,0,1,4,3,2,90,90,560,1
Style: Emph,Poppins,92,&H002B24E4,&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,1,0,1,5,4,2,80,80,560,1
Style: Serif,Lora,62,&H00FFFFFF,&H000000FF,&H50101010,&H64000000,0,-1,0,0,100,100,0,0,1,2.2,3,2,110,110,600,1
Style: SerifE,Lora,66,&H00FFFFFF,&H000000FF,&H50101010,&H64000000,-1,-1,0,0,100,100,0,0,1,2.4,3,2,110,110,600,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    if whatif:
        for (start, end, text, beat) in timeline:
            if beat.get("scene") == "endcard":
                continue
            st = "SerifE" if beat.get("emph") else "Serif"
            txt = text.replace("{", "(").replace("}", ")")
            words = txt.split()
            if len(txt) > 26:
                mid = len(words) // 2; txt = " ".join(words[:mid]) + "\\N" + " ".join(words[mid:])
            ev.append(f"Dialogue: 0,{mv.ass_time(start)},{mv.ass_time(end + 0.3)},{st},,0,0,0,,{{\\fad(150,150)}}{txt}")
        open(path, "w", encoding="utf-8").write(head + "\n".join(ev) + "\n")
        return
    for (start, end, text, beat) in timeline:
        words = text.upper().split(); chunks, cur = [], []
        for w in words:
            cur.append(w)
            if len(cur) >= 3 or (re.search(r"[.,!?…:;]$", w) and len(cur) >= 1):
                chunks.append(cur); cur = []
        if cur:
            chunks.append(cur)
        nch = sum(len(" ".join(c)) for c in chunks) or 1; t = start
        style = "Emph" if beat.get("emph") else "Cap"
        for c in chunks:
            txt = " ".join(c).replace("{", "(").replace("}", ")")
            if len(c) >= 2 and len(txt) > 12:
                txt = " ".join(c[:len(c) // 2 + len(c) % 2]) + "\\N" + " ".join(c[len(c) // 2 + len(c) % 2:])
            d = (end - start) * len(" ".join(c)) / nch
            fx = "{\\fad(40,60)\\fscx85\\fscy85\\t(0,110,\\fscx100\\fscy100)}"
            ev.append(f"Dialogue: 0,{mv.ass_time(t)},{mv.ass_time(t + d)},{style},,0,0,0,,{fx}{txt}")
            t += d
    open(path, "w", encoding="utf-8").write(head + "\n".join(ev) + "\n")


# ====================================================================== main
def main():
    story_path, out = sys.argv[1], sys.argv[2]
    story = json.load(open(story_path, encoding="utf-8"))
    beats = story["beats"]
    work = os.path.join(os.path.dirname(os.path.abspath(out)), "work_s"); os.makedirs(work, exist_ok=True)
    rng = np.random.default_rng(sum(map(ord, story.get("title", "x"))) * 31)

    # ---- narration + sfx (reuse make_video's narrate: items need "text"/"sfx")
    items = [{"text": b["text"], "sfx": b.get("sfx")} for b in beats]
    voice_tr, fx_tr, tl, total, flashes = mv.narrate(items, story.get("voice", "am_onyx"), work)
    timeline = [(s, e, txt, beats[i]) for i, (s, e, txt, _) in enumerate(tl)]
    n = len(voice_tr)

    # ---- per-time curves
    starts = [s for s, _, _, _ in timeline] + [total]
    def beat_at(t):
        for i in range(len(beats)):
            if t < starts[i + 1]:
                return i
        return len(beats) - 1
    dists = [float(b.get("value", b.get("distance", 100))) for b in beats]
    whatif = story.get("style") == "whatif"
    dmax = max(dists + [1.0])
    def dist_at(t):
        i = beat_at(t); a = dists[i - 1] if i > 0 else dists[0]; b = dists[i]
        if whatif and (i == 0 or beats[i].get("label") != beats[i - 1].get("label")):
            a = float(beats[i].get("from", 0))
        u = min(1, (t - starts[i]) / max(0.6, min(2.0, starts[i + 1] - starts[i])))
        u = u * u * (3 - 2 * u)
        return a + (b - a) * u
    dangers = [float(b.get("danger", 0.3)) for b in beats]
    def danger_at(t):
        i = beat_at(t); a = dangers[i - 1] if i > 0 else dangers[0]; b = dangers[i]
        u = min(1, (t - starts[i]) / 1.0)
        return a + (b - a) * u
    bpm_at = lambda t: 72 + 95 * danger_at(t)

    # ---- audio mix
    thr = threat_sound(story.get("threat", "rotor"), n)
    tt = np.arange(n) / SR
    prox = np.array([1 - dist_at(x) / dmax for x in np.linspace(0, total, 400)])
    prox = np.interp(tt, np.linspace(0, total, 400), prox)
    hb = heartbeat_track(bpm_at, n)
    bed = mv.ambience(story.get("ambience", "wind"), n)
    voice_tr = mv.norm(mv.lp_fast(voice_tr, 9000), 0.95)       # soften synthetic fizz/sibilance
    venv = mv.lp_fast(np.abs(voice_tr), 3); duck = 1 - np.clip(venv * 6, 0, 0.6); fxd = 1 - np.clip(venv * 8, 0, 0.75)
    grid = np.linspace(0, total, 400)
    dz_curve = np.interp(tt, grid, [danger_at(x) for x in grid])
    mix = (voice_tr + fx_tr * 0.28 * fxd + bed * 0.6 * duck + thr * (0.04 + 0.22 * prox ** 1.5) * duck
           + hb * (0.05 + 0.16 * dz_curve) * duck * (0.35 if whatif else 1.0))
    mix = np.tanh(mix * 1.15) * 0.92
    import soundfile as sf
    wav = os.path.join(work, "mix.wav"); sf.write(wav, mix.astype(np.float32), SR)
    make_ass(timeline, os.path.join(work, "caps.ass"), whatif)

    # ---- video frames → ffmpeg
    hud = HUD(story)
    scenes = {}
    nframes = int(total * FPS)
    raw = os.path.join(work, "frames.mp4")
    enc = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p", raw],
                           stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    last_i = -1; cut_t = 0.0
    for f in range(nframes):
        t = f / FPS; i = beat_at(t)
        kind = beats[i].get("scene", "black")
        if i != last_i:
            if kind != (beats[last_i].get("scene") if last_i >= 0 else None):
                cut_t = t
            last_i = i
        key = (kind, beats[i].get("focus", "heart"))
        if key not in scenes:
            scenes[key] = Scene(kind, rng, beats[i].get("focus", "heart"))
        dz = danger_at(t); dn = (1 - dz) if whatif else dist_at(t) / dmax
        flash = max(0.0, 1 - (t - cut_t) * 8) * 0.6          # short flash on every cut
        for fl in flashes:
            if 0 <= t - fl < 0.25:
                flash = max(flash, 1 - (t - fl) * 4)
        img = scenes[key].frame(t, t - starts[i], dz, dn)
        frame = post(img, t, dz, flash)
        if whatif:
            overlay = hud.render_whatif(t, dist_at(t), beats[i], t - starts[i], story.get("endcard"))
        else:
            overlay = hud.render(t, dist_at(t), bpm_at(t), dz, beats[i].get("formula", ""), kind, int(t * 2) % 2 == 0)
        frame = alpha_over(frame, overlay)
        enc.stdin.write(frame.tobytes())
    enc.stdin.close(); enc.wait()

    ass = os.path.join(work, "caps.ass").replace(":", "\\:")
    mv.run(["ffmpeg", "-y", "-i", raw, "-i", wav, "-vf", f"ass={ass}:fontsdir={FONT_DIR}", "-map", "0:v", "-map", "1:a",
            "-t", f"{total:.2f}", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-maxrate", "4500k", "-bufsize", "9000k",
            "-pix_fmt", "yuv420p", "-af", "aresample=48000:resampler=soxr,lowpass=f=11000", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", out])
    print(f"OK {out} {total:.1f}s {os.path.getsize(out)} bytes")


if __name__ == "__main__":
    main()
