#!/usr/bin/env python3
"""Vertical (1080x1920) scary-story video: natural narration (Kokoro), parallax scenery,
fog, rain, lightning, sound effects and burned-in captions.

Usage: python3 make_video.py story.json out.mp4
Files expected next to this script: kokoro/kokoro-v1.0.onnx, kokoro/voices-v1.0.bin
Needs: ffmpeg, pip: kokoro-onnx soundfile numpy pillow.

story.json:
{
  "title": "Don't Let Him In",
  "voice": "am_onyx",                       # optional
  "ambience": "rain" | "wind" | "drone",    # optional, default rain
  "lines": [                                 # narration, one sentence/short beat per item
     "text",                                 # plain line
     {"text": "...", "sfx": "knock", "scene": "house"}   # sfx before the line; scene change at this line
  ]
}
sfx: knock, phone, thunder, heartbeat, footsteps, door, whisper, stinger
scene: road, house, forest, graveyard, hall, field
"""
import json, math, os, re, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H, FPS, SR = 1080, 1920, 30, 24000
HERE = os.path.dirname(os.path.abspath(__file__))
PAD = 700  # extra width for parallax travel


def run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if r.returncode:
        sys.stderr.write(r.stderr.decode()[-3000:]); raise SystemExit(1)


# =============================================================== AUDIO
def env(n, a, d):
    e = np.ones(n); a = max(1, int(a * SR)); d = max(1, int(d * SR))
    e[:a] = np.linspace(0, 1, a); e[-d:] *= np.linspace(1, 0, d)
    return e


def lowpass(x, cutoff):
    # one-pole low pass
    a = math.exp(-2 * math.pi * cutoff / SR); y = np.empty_like(x); acc = 0.0
    b = 1 - a
    for i in range(len(x)):
        acc = b * x[i] + a * acc; y[i] = acc
    return y


def lp_fast(x, cutoff):
    # FFT brick-ish low pass (fast for long signals)
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / (1 + (f / cutoff) ** 4)
    return np.fft.irfft(X, len(x))


def hp_fast(x, cutoff):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / (1 + (cutoff / np.maximum(f, 1)) ** 4)
    return np.fft.irfft(X, len(x))


def norm(x, peak):
    m = np.max(np.abs(x)) or 1
    return x / m * peak


rng_a = np.random.default_rng(7)


def sfx(kind):
    r = rng_a
    if kind == "knock":
        out = np.zeros(int(1.6 * SR))
        for k in range(3):
            n = int(0.12 * SR); t = np.arange(n) / SR
            hit = (np.sin(2 * np.pi * 95 * t) * 0.8 + lp_fast(r.standard_normal(n), 900) * 0.6) * np.exp(-t * 38)
            s = int(k * 0.42 * SR); out[s:s + n] += hit
        return norm(out, 0.9)
    if kind == "phone":
        out = []
        for _ in range(2):
            t = np.arange(int(1.0 * SR)) / SR
            tone = (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 480 * t)) * (np.sin(2 * np.pi * 20 * t) > 0)
            out += [tone * env(len(t), .01, .05), np.zeros(int(0.6 * SR))]
        return norm(np.concatenate(out), 0.35)
    if kind == "thunder":
        n = int(4.5 * SR); t = np.arange(n) / SR
        crack = hp_fast(r.standard_normal(n), 1500) * np.exp(-t * 9) * 0.5
        rumble = lp_fast(r.standard_normal(n), 120) * (1 - np.exp(-t * 8)) * np.exp(-t * 0.8)
        rumble *= 1 + 0.6 * np.sin(2 * np.pi * 1.7 * t)
        return norm(norm(rumble, 1) + crack, 0.95)
    if kind == "heartbeat":
        out = np.zeros(int(3.2 * SR))
        for k in range(4):
            for off, amp in ((0, 1), (0.22, 0.7)):
                n = int(0.18 * SR); t = np.arange(n) / SR
                thump = np.sin(2 * np.pi * 52 * t) * np.exp(-t * 22) * amp
                s = int((k * 0.8 + off) * SR); out[s:s + n] += thump
        return norm(out, 0.95)
    if kind == "footsteps":
        out = np.zeros(int(3.0 * SR))
        for k in range(6):
            n = int(0.09 * SR); t = np.arange(n) / SR
            step = lp_fast(r.standard_normal(n), 700) * np.exp(-t * 45)
            s = int((0.15 + k * 0.45) * SR); out[s:s + n] += step * (0.6 + 0.08 * k)
        return norm(out, 0.7)
    if kind == "door":
        n = int(2.4 * SR); t = np.arange(n) / SR
        f = 180 + 120 * np.sin(2 * np.pi * 0.35 * t) + 40 * r.standard_normal(n).cumsum() / np.sqrt(SR)
        ph = 2 * np.pi * np.cumsum(f) / SR
        creak = np.sign(np.sin(ph)) * (0.5 + 0.5 * np.sin(2 * np.pi * 13 * t)) * env(n, .2, .5)
        return norm(lp_fast(creak, 2500), 0.4)
    if kind == "whisper":
        n = int(2.2 * SR); t = np.arange(n) / SR
        w = lp_fast(hp_fast(r.standard_normal(n), 1800), 5500) * (0.5 + 0.5 * np.sin(2 * np.pi * 3.1 * t) ** 2) * env(n, .4, .8)
        return norm(w, 0.25)
    if kind == "stinger":
        n = int(3.0 * SR); t = np.arange(n) / SR
        rise = sum(np.sin(2 * np.pi * (f0 * (1 + t * 0.6)) * t) for f0 in (220, 233, 311)) * (t / 3) ** 2
        hit_n = int(2.0 * SR); th = np.arange(hit_n) / SR
        hit = (lp_fast(r.standard_normal(hit_n), 300) * 2 + np.sin(2 * np.pi * 41 * th)) * np.exp(-th * 2.2)
        out = np.zeros(int(5.0 * SR)); out[:n] += norm(rise, 0.35); out[n - int(0.05 * SR):n - int(0.05 * SR) + hit_n] += norm(hit, 1.0)
        return out
    return np.zeros(1)


def ambience(kind, n):
    r = rng_a; t = np.arange(n) / SR
    drone = (np.sin(2 * np.pi * 55 * t) * 0.5 + np.sin(2 * np.pi * 58.2 * t) * 0.4 + np.sin(2 * np.pi * 82.4 * t) * 0.15)
    drone *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.07 * t)
    bed = norm(drone, 0.07)
    if kind == "rain":
        rain = hp_fast(r.standard_normal(n), 900); rain = lp_fast(rain, 7000)
        bed += norm(rain, 0.10)
    if kind in ("wind", "rain"):
        wind = lp_fast(r.standard_normal(n), 400) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.11 * t + 1) ** 2)
        bed += norm(wind, 0.08 if kind == "wind" else 0.04)
    fade = np.minimum(1, np.minimum(t / 2, (t[-1] - t) / 2))
    return bed * fade


def _split_points(audio, sr, texts):
    """Find where each sentence starts inside one synthesized paragraph (pauses between sentences)."""
    n = len(audio); hop = int(0.01 * sr)
    env_ = np.sqrt(np.convolve(audio ** 2, np.ones(hop) / hop, mode="same"))[::hop]
    thr = max(1e-4, 0.06 * np.percentile(env_, 95))
    quiet = env_ < thr
    runs, i = [], 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            if (j - i) >= 6 and i > 0 and j < len(quiet):          # >= 60 ms pause inside the paragraph
                runs.append(((i + j) / 2 * hop, (j - i)))
            i = j
        else:
            i += 1
    lens = np.array([len(x) + 1 for x in texts], float); cum = np.cumsum(lens)[:-1] / lens.sum()
    est = cum * n; pts = []; last = 0
    for e in est:
        cands = [c for c, ln in runs if c > last + 0.25 * sr]
        if cands:
            c = min(cands, key=lambda c: abs(c - e) - 0.002 * sr * 0)
            if abs(c - e) > 0.35 * n / max(1, len(texts)):
                c = e
        else:
            c = e
        pts.append(int(c)); last = c
    return pts


def narrate(items, voice, work, speed=1.0, group=4):
    """Synthesize the narration. Consecutive lines are read together (up to `group` lines, split at
    sound effects) so the intonation is natural; sentence start times are recovered from the pauses."""
    from kokoro_onnx import Kokoro
    k = Kokoro(os.path.join(HERE, "kokoro", "kokoro-v1.0.onnx"), os.path.join(HERE, "kokoro", "voices-v1.0.bin"))
    lang = "en-gb" if voice.startswith("b") else "en-us"
    pieces, timeline, t = [], [], 0.8
    # build groups
    groups, cur = [], []
    for idx, it in enumerate(items):
        if cur and (it.get("sfx") or len(cur) >= group):
            groups.append(cur); cur = []
        cur.append(idx)
    if cur:
        groups.append(cur)
    for g in groups:
        first = items[g[0]]
        if first.get("sfx"):
            s = sfx(first["sfx"]); pieces.append(("sfx", t, s, first["sfx"]))
            t += {"knock": 1.3, "phone": 1.6, "thunder": 1.2, "heartbeat": 0.6, "footsteps": 1.6,
                  "door": 1.2, "whisper": 0.4, "stinger": 0}.get(first["sfx"], 0.5)
        texts = [items[i]["text"].strip() for i in g]
        audio, sr = k.create(" ".join(texts), voice=voice, speed=speed, lang=lang)
        audio = np.asarray(audio, dtype=np.float64)
        fl = min(len(audio) // 4, int(0.012 * sr))
        if fl > 1:
            audio[:fl] *= np.linspace(0, 1, fl); audio[-fl:] *= np.linspace(1, 0, fl)
        pts = _split_points(audio, sr, texts) if len(texts) > 1 else []
        if sr != SR:
            ratio = SR / sr
            audio = np.interp(np.arange(0, len(audio) * ratio) / ratio, np.arange(len(audio)), audio)
            pts = [int(p * ratio) for p in pts]
        bounds = [0] + pts + [len(audio)]
        for j, i in enumerate(g):
            a, b = bounds[j] / SR, bounds[j + 1] / SR
            timeline.append((t + a, t + b, items[i]["text"], items[i].get("scene")))
        pieces.append(("voice", t, audio, None))
        t += len(audio) / SR + 0.45
    total = t + 2.5
    n = int(total * SR)
    voice_tr = np.zeros(n); fx_tr = np.zeros(n)
    flashes = []
    for kind, start, a, name in pieces:
        s = int(start * SR); e = min(n, s + len(a))
        (voice_tr if kind == "voice" else fx_tr)[s:e] += a[:e - s]
        if name == "thunder":
            flashes.append(start)
    return voice_tr, fx_tr, timeline, total, flashes


# =============================================================== CAPTIONS
def ass_time(s):
    return f"{int(s // 3600)}:{int(s % 3600 // 60):02d}:{s % 60:05.2f}"


def make_ass(title, timeline, path):
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Poppins,88,&H00FFFFFF,&H000000FF,&H00000000,&HA0000000,-1,0,0,0,100,100,1,0,1,7,4,5,90,90,0,1
Style: Title,Poppins,76,&H00FFFFFF,&H000000FF,&H00000000,&HA0000000,-1,0,0,0,100,100,1,0,1,6,4,8,80,80,280,1
Style: Red,Poppins,50,&H002424E6,&H000000FF,&H00000000,&HA0000000,-1,0,0,0,100,100,4,0,1,4,2,8,80,80,210,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = [f"Dialogue: 1,{ass_time(0)},{ass_time(3.4)},Red,,0,0,0,,{{\\fad(200,300)}}SCARY STORY",
          f"Dialogue: 1,{ass_time(0)},{ass_time(3.4)},Title,,0,0,0,,{{\\fad(200,300)}}{title.upper()}"]
    for start, end, line, _ in timeline:
        words = line.split(); chunks, cur = [], []
        for w in words:
            cur.append(w)
            if len(cur) >= 4 or (re.search(r"[.,!?…:;\"”]$", w) and len(cur) >= 2):
                chunks.append(cur); cur = []
        if cur:
            chunks.append(cur)
        nch = sum(len(" ".join(c)) for c in chunks) or 1; t = start
        for c in chunks:
            txt = " ".join(c).upper().replace("{", "(").replace("}", ")"); d = (end - start) * len(txt) / nch
            ev.append(f"Dialogue: 0,{ass_time(t)},{ass_time(t + d)},Cap,,0,0,0,,{{\\fad(60,60)\\fscx92\\fscy92\\t(0,120,\\fscx100\\fscy100)}}{txt}")
            t += d
    open(path, "w", encoding="utf-8").write(head + "\n".join(ev) + "\n")


# =============================================================== SCENERY
def noise(rng, w, h, scale, octaves=3):
    acc = np.zeros((h, w), np.float32); amp = 1; tot = 0
    for o in range(octaves):
        s = max(2, scale // (2 ** o))
        small = rng.random((h // s + 2, w // s + 2)).astype(np.float32)
        acc += amp * np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), np.float32) / 255
        tot += amp; amp *= 0.5
    return acc / tot


def tree(dr, rng, x, base, h, width, col, depth=0):
    """Recursive gnarled dead tree."""
    def branch(x0, y0, ang, length, wdt, d):
        if d > 6 or length < 12:
            return
        x1 = x0 + math.cos(ang) * length; y1 = y0 - math.sin(ang) * length
        dr.line([x0, y0, x1, y1], fill=col, width=max(1, int(wdt)))
        dr.ellipse([x1 - wdt / 2, y1 - wdt / 2, x1 + wdt / 2, y1 + wdt / 2], fill=col)
        for _ in range(rng.integers(2, 4)):
            branch(x1, y1, ang + rng.uniform(-0.75, 0.75), length * rng.uniform(0.6, 0.8), wdt * 0.66, d + 1)
    trunk_h = h * 0.45
    pts = [(x - width, base), (x + width, base), (x + width * 0.45, base - trunk_h), (x - width * 0.45, base - trunk_h)]
    dr.polygon(pts, fill=col)
    for _ in range(rng.integers(2, 4)):
        branch(x, base - trunk_h, math.pi / 2 + rng.uniform(-0.5, 0.5), h * 0.28, width * 0.9, 0)


def house(dr, rng, x, base, s, col, lit):
    w, h = int(420 * s), int(330 * s)
    dr.rectangle([x, base - h, x + w, base], fill=col)
    dr.polygon([(x - 30 * s, base - h), (x + w * 0.5, base - h - 230 * s), (x + w + 30 * s, base - h)], fill=col)
    tw = int(120 * s)  # tower
    dr.rectangle([x + w - tw, base - h - 200 * s, x + w + 10 * s, base - h], fill=col)
    dr.polygon([(x + w - tw - 15 * s, base - h - 200 * s), (x + w - tw / 2 + 5 * s, base - h - 380 * s), (x + w + 25 * s, base - h - 200 * s)], fill=col)
    dr.rectangle([x + 60 * s, base - h - 170 * s, x + 95 * s, base - h - 60 * s], fill=col)  # chimney
    wins = [(x + 70 * s, base - h + 70 * s), (x + 250 * s, base - h + 70 * s), (x + 70 * s, base - h + 190 * s), (x + w - tw + 30 * s, base - h - 150 * s)]
    for i, (wx, wy) in enumerate(wins):
        on = i in lit
        c = (255, 196, 92) if on else (14, 12, 16)
        dr.rectangle([wx, wy, wx + 60 * s, wy + 80 * s], fill=c)
        dr.line([wx + 30 * s, wy, wx + 30 * s, wy + 80 * s], fill=col, width=max(2, int(6 * s)))
        dr.line([wx, wy + 40 * s, wx + 60 * s, wy + 40 * s], fill=col, width=max(2, int(6 * s)))
        if on and i == lit[0]:
            dr.ellipse([wx + 18 * s, wy + 14 * s, wx + 42 * s, wy + 40 * s], fill=col)
            dr.rectangle([wx + 14 * s, wy + 38 * s, wx + 46 * s, wy + 80 * s], fill=col)
    dr.rectangle([x + 180 * s, base - 120 * s, x + 240 * s, base], fill=(20, 16, 14))


def layer(size):
    im = Image.new("RGBA", size, (0, 0, 0, 0)); return im, ImageDraw.Draw(im)


def build_scene(rng, kind, folder, palette):
    """Writes sky.png (opaque) + far/mid/near/fog.png (RGBA) into folder."""
    os.makedirs(folder, exist_ok=True)
    lw = W + PAD
    sky_c, glow_c, far_c = palette
    yy = np.linspace(0, 1, H)[:, None, None]
    sky = (np.array(sky_c) * (1 - yy) * 0.9 + np.array(glow_c) * np.exp(-((yy - 0.62) ** 2) / 0.02) * 0.55 + 6) * np.ones((1, lw, 1))
    # clouds
    cl = noise(rng, lw, H, 320, 4)
    cl = np.clip((cl - 0.45) * 2.2, 0, 1)[..., None] * (1 - yy) ** 0.6
    sky = sky * (1 - cl * 0.35) + cl * np.array(glow_c) * 0.25
    interior = kind == "hall"
    if not interior:
        mx, my, mr = rng.integers(300, lw - 300), rng.integers(330, 640), rng.integers(85, 125)
        Y, X = np.mgrid[0:H, 0:lw]; d = np.sqrt((X - mx) ** 2 + (Y - my) ** 2)
        halo = np.clip(1 - d / (mr * 6), 0, 1) ** 2.2
        crater = noise(rng, lw, H, 18, 2)
        disk = np.clip((mr - d) / 2, 0, 1) * (0.82 + 0.18 * crater)
        moon = np.array([232, 226, 205]); sky = sky * (1 - disk[..., None]) + disk[..., None] * moon + halo[..., None] * np.array(glow_c) * 0.9
        sky = sky * (1 - cl * 0.55 * (d < mr * 3)[..., None]) + 0  # clouds crossing the moon
    Image.fromarray(np.clip(sky, 0, 255).astype(np.uint8)).save(os.path.join(folder, "sky.png"))

    ground = int(H * rng.uniform(0.68, 0.73))
    farcol = tuple(int(c) for c in far_c) + (255,)
    midcol = (10, 9, 13, 255); nearcol = (2, 2, 3, 255)

    # FAR: hills / distant trees (atmospheric colour)
    far, fd = layer((lw, H))
    if not interior:
        pts = [(0, H)] + [(x, ground - 120 - int(70 * math.sin(x / 260 + rng.random()))) for x in range(0, lw + 60, 60)] + [(lw, H)]
        fd.polygon(pts, fill=farcol)
        for _ in range(int(rng.integers(10, 18))):
            x = int(rng.integers(0, lw)); tree(fd, rng, x, ground - 100, int(rng.integers(220, 380)), 7, farcol)
    far = far.filter(ImageFilter.GaussianBlur(2.5)); far.save(os.path.join(folder, "far.png"))

    # MID: the subject of the scene
    mid, md = layer((lw, H))
    md.rectangle([0, ground, lw, H], fill=midcol)
    cx = lw // 2
    if kind == "house":
        house(md, rng, cx - 260, ground + 10, 1.25, midcol, [0, 3] if rng.random() < .5 else [1])
        for x in (cx - 520, cx + 420):
            tree(md, rng, x, ground + 10, int(rng.integers(650, 900)), 22, midcol)
        for k in range(9):  # fence
            fx = cx - 600 + k * 70; md.rectangle([fx, ground - 60, fx + 12, ground + 20], fill=midcol)
        md.rectangle([cx - 610, ground - 40, cx + 30, ground - 30], fill=midcol)
    elif kind in ("forest", "field"):
        for _ in range(int(rng.integers(6, 10)) if kind == "forest" else 2):
            x = int(rng.integers(0, lw)); tree(md, rng, x, ground + 10, int(rng.integers(700, 1200)), int(rng.integers(18, 34)), midcol)
        if kind == "field":
            md.rectangle([cx - 8, ground - 330, cx + 8, ground + 10], fill=midcol)  # scarecrow
            md.rectangle([cx - 120, ground - 270, cx + 120, ground - 256], fill=midcol)
            md.ellipse([cx - 42, ground - 400, cx + 42, ground - 320], fill=midcol)
            md.polygon([(cx - 70, ground - 395), (cx, ground - 470), (cx + 70, ground - 395)], fill=midcol)
            md.ellipse([cx - 22, ground - 375, cx - 8, ground - 361], fill=(200, 30, 30, 255)); md.ellipse([cx + 8, ground - 375, cx + 22, ground - 361], fill=(200, 30, 30, 255))
    elif kind == "graveyard":
        for k in range(10):
            gx = int(80 + k * (lw - 160) / 10 + rng.integers(-30, 30)); gh = int(rng.integers(110, 200)); gw = int(rng.integers(60, 100))
            tilt = rng.uniform(-6, 6)
            stone, sd = layer((gw + 40, gh + 40));
            if rng.random() < .35:
                sd.rectangle([gw // 2 + 10, 10, gw // 2 + 30, gh + 30], fill=midcol); sd.rectangle([10, 50, gw + 30, 70], fill=midcol)
            else:
                sd.rounded_rectangle([20, 20, gw + 20, gh + 40], radius=gw // 2, fill=midcol)
            stone = stone.rotate(tilt, expand=True); mid.alpha_composite(stone, (gx, ground - gh + int(rng.integers(0, 40))))
        tree(md, rng, cx + 300, ground + 10, 1100, 30, midcol)
        md.rectangle([cx - 200, ground - 420, cx - 180, ground + 10], fill=midcol); md.rectangle([cx - 290, ground - 420, cx - 90, ground - 405], fill=midcol)
    elif kind == "road":
        md.polygon([(cx - 6, ground + 2), (cx + 6, ground + 2), (cx + 700, H), (cx - 700, H)], fill=(24, 22, 26, 255))
        for k in range(8):
            y0 = ground + 10 + k ** 2 * 14; y1 = y0 + 6 + k * 8; s = (y0 - ground) / (H - ground)
            md.polygon([(cx - 3 - 6 * s * 3, y0), (cx + 3 + 6 * s * 3, y0), (cx + 4 + 6 * s * 4, y1), (cx - 4 - 6 * s * 4, y1)], fill=(150, 140, 110, 255))
        px = cx + 360
        md.rectangle([px, ground - 820, px + 16, ground + 260], fill=midcol); md.rectangle([px - 140, ground - 820, px + 16, ground - 808], fill=midcol)
        md.polygon([(px - 160, ground - 812), (px - 110, ground - 812), (px - 120, ground - 790), (px - 150, ground - 790)], fill=(255, 220, 150, 255))
        lamp = Image.new("L", (lw, H), 0); ImageDraw.Draw(lamp).polygon([(px - 150, ground - 790), (px - 420, ground + 300), (px + 140, ground + 300)], fill=90)
        lamp = lamp.filter(ImageFilter.GaussianBlur(45))
        beam = Image.new("RGBA", (lw, H), (255, 214, 150, 0)); beam.putalpha(lamp); mid.alpha_composite(beam)
        fx = px - 150; md.ellipse([fx, ground + 40, fx + 46, ground + 88], fill=midcol); md.polygon([(fx - 8, ground + 300), (fx + 54, ground + 300), (fx + 40, ground + 86), (fx + 6, ground + 86)], fill=midcol)
        for x in (cx - 560, cx - 380):
            tree(md, rng, x, ground + 10, int(rng.integers(600, 900)), 20, midcol)
    elif kind == "hall":
        md.rectangle([0, 0, lw, H], fill=(7, 6, 8, 255))
        cy = int(H * 0.44)
        for i in range(14, 0, -1):
            s = i / 14; v = int(12 + 34 * (1 - s) ** 1.5)
            md.rectangle([cx - int(W * 0.62 * s), cy - int(H * 0.40 * s), cx + int(W * 0.62 * s), cy + int(H * 0.56 * s)], outline=(v, v - 2, v + 3, 255), width=5)
            if i % 3 == 0:  # side doors
                dx = int(W * 0.62 * s); dh = int(H * 0.28 * s); dw = int(W * 0.10 * s)
                md.rectangle([cx - dx + 8, cy + int(H * 0.56 * s) - dh, cx - dx + 8 + dw, cy + int(H * 0.56 * s)], fill=(16, 13, 15, 255))
                md.rectangle([cx + dx - 8 - dw, cy + int(H * 0.56 * s) - dh, cx + dx - 8, cy + int(H * 0.56 * s)], fill=(16, 13, 15, 255))
        light = Image.new("L", (lw, H), 0); ImageDraw.Draw(light).rectangle([cx - 80, cy - 170, cx + 80, cy + 180], fill=255)
        glow = light.filter(ImageFilter.GaussianBlur(70)).point(lambda v: int(v * 0.8))
        g = Image.new("RGBA", (lw, H), (230, 180, 110, 0)); g.putalpha(glow); mid.alpha_composite(g)
        md.rectangle([cx - 80, cy - 170, cx + 80, cy + 180], fill=(236, 196, 128, 255))
        md.ellipse([cx - 24, cy - 60, cx + 24, cy - 8], fill=nearcol); md.polygon([(cx - 36, cy + 180), (cx + 36, cy + 180), (cx + 28, cy - 10), (cx - 28, cy - 10)], fill=nearcol)
    mid.save(os.path.join(folder, "mid.png"))

    # NEAR: big dark foreground framing
    near, nd = layer((lw + PAD // 2, H))
    if not interior:
        pts = [(0, H)] + [(x, H - 160 - int(60 * math.sin(x / 140 + 2) + rng.integers(0, 25))) for x in range(0, lw + PAD // 2 + 30, 30)] + [(lw + PAD // 2, H)]
        nd.polygon(pts, fill=nearcol)
        for _ in range(140):  # grass blades
            x = int(rng.integers(0, lw + PAD // 2)); y = H - 170 + int(rng.integers(-20, 30)); hgt = int(rng.integers(40, 140))
            nd.line([x, y, x + int(rng.integers(-25, 25)), y - hgt], fill=nearcol, width=int(rng.integers(3, 7)))
        side = int(rng.integers(0, 2))
        tx = 60 if side == 0 else lw + PAD // 2 - 60
        tree(nd, rng, tx, H + 40, 2100, 70, nearcol)
    else:
        nd.rectangle([0, 0, 110, H], fill=nearcol); nd.rectangle([lw + PAD // 2 - 110, 0, lw + PAD // 2, H], fill=nearcol)
    near = near.filter(ImageFilter.GaussianBlur(3)); near.save(os.path.join(folder, "near.png"))

    # FOG band (RGBA), twice as wide to drift
    fw = lw * 2
    f = noise(rng, fw, H, 260, 4)
    band = np.exp(-((np.linspace(0, 1, H) - (ground / H + 0.03)) ** 2) / 0.012)[:, None]
    a = np.clip((f - 0.35) * 1.6, 0, 1) * band * (150 if not interior else 90)
    fog = np.zeros((H, fw, 4), np.uint8); fog[..., :3] = (175, 180, 195); fog[..., 3] = a.astype(np.uint8)
    Image.fromarray(fog).save(os.path.join(folder, "fog.png"))
    return ground


def rain_layer(path, rng):
    im, d = layer((W, H * 2))
    for _ in range(1400):
        x = int(rng.integers(0, W)); y = int(rng.integers(0, H * 2)); L = int(rng.integers(30, 70))
        d.line([x, y, x - 6, y + L], fill=(190, 200, 220, int(rng.integers(40, 110))), width=2)
    im.save(path)


def vignette(path):
    Y, X = np.mgrid[0:H, 0:W]; v = np.sqrt(((X - W / 2) / (W / 2)) ** 2 * 0.8 + ((Y - H / 2) / (H / 2)) ** 2)
    a = np.clip((v - 0.55) * 300, 0, 235).astype(np.uint8); im = np.zeros((H, W, 4), np.uint8); im[..., 3] = a
    Image.fromarray(im).save(path)


PALETTES = [((22, 30, 58), (120, 140, 190), (40, 46, 70)),   # cold blue
            ((36, 18, 22), (170, 70, 60), (60, 30, 34)),     # blood red
            ((14, 32, 30), (90, 150, 130), (30, 52, 50)),    # sickly green
            ((30, 20, 44), (140, 100, 180), (50, 38, 70))]   # violet


def render_scene(folder, dur, idx, raining, out):
    zoom_in = idx % 2 == 0
    sp = [6, 14, 30, 55, 70]  # px/s: sky, far, mid, near, fog
    dirn = -1 if idx % 2 == 0 else 1
    def xexpr(speed, w_extra):
        base = -(w_extra // 2)
        return f"'{base}+{dirn}*t*{speed}'"
    inputs = []
    for name in ("sky", "far", "mid", "near", "fog"):
        inputs += ["-loop", "1", "-framerate", str(FPS), "-t", f"{dur:.2f}", "-i", os.path.join(folder, name + ".png")]
    fc = (f"[0:v]crop={W}:{H}:x='{PAD // 2}-{dirn}*t*{sp[0]}':y=0[sky];"
          f"[sky][1:v]overlay=x={xexpr(sp[1], PAD)}:y=0[a];"
          f"[a][2:v]overlay=x={xexpr(sp[2], PAD)}:y=0[b];"
          f"[b][4:v]overlay=x='-{W}+mod(t*{sp[4]}\\,{W})':y=0[c];"
          f"[c][3:v]overlay=x={xexpr(sp[3], PAD + PAD // 2)}:y=0[d]")
    last = "d"
    if raining:
        inputs += ["-loop", "1", "-framerate", str(FPS), "-t", f"{dur:.2f}", "-i", os.path.join(os.path.dirname(folder), "rain.png")]
        fc += f";[{last}][5:v]overlay=x=0:y='-{H}+mod(t*1500\\,{H})'[r]"; last = "r"
    z = f"1.0+0.06*t/{dur:.2f}" if zoom_in else f"1.06-0.06*t/{dur:.2f}"
    fc += (f";[{last}]scale=w='iw*({z})':h='ih*({z})':eval=frame,crop={W}:{H},setsar=1,format=yuv420p[v]")
    run(["ffmpeg", "-y", *inputs, "-filter_complex", fc, "-map", "[v]", "-t", f"{dur:.2f}", "-r", str(FPS),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", out])


def still_clip(png, dur, idx, raining, out, work):
    frames = int(dur * FPS) + 1
    zin = idx % 2 == 0
    z = f"1.0+0.06*on/{frames}" if zin else f"1.06-0.06*on/{frames}"
    sx = "trunc(iw/2-(iw/zoom/2))"; sy = "trunc(ih/2-(ih/zoom/2))"
    raining = False  # rain and shake waste bitrate in the small upload
    inputs = ["-loop", "1", "-i", png]
    fc = f"[0:v]scale={W*2}:{H*2},zoompan=z='{z}':x='{sx}':y='{sy}':d={frames}:s={W}x{H}:fps={FPS}[a]"
    last = "a"
    if raining:
        inputs += ["-loop", "1", "-framerate", str(FPS), "-t", f"{dur:.2f}", "-i", os.path.join(work, "rain.png")]
        fc += f";[a][1:v]overlay=x=0:y='-{H}+mod(t*1500\\,{H})'[r]"; last = "r"
    fc += f";[{last}]format=yuv420p[v]"
    run(["ffmpeg", "-y", *inputs, "-filter_complex", fc, "-map", "[v]", "-t", f"{dur:.2f}", "-r", str(FPS),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", out])


def make_small(src, dst, total, limit_mb=2.85):
    """720x1280 H.264 (2-pass, 24 fps) under ~2.85 MB so it fits through the Drive connector."""
    log = os.path.join(os.path.dirname(os.path.abspath(dst)), "pass")
    for attempt in range(4):
        vbr = int((limit_mb * 8 * 1000 / total) * (0.97 - 0.05 * attempt) - 42)
        common = ["-vf", "scale=720:1280:flags=lanczos", "-r", "24", "-c:v", "libx264", "-preset", "slow",
                  "-b:v", f"{vbr}k", "-passlogfile", log, "-pix_fmt", "yuv420p"]
        run(["ffmpeg", "-y", "-i", src, *common, "-pass", "1", "-an", "-f", "mp4", os.devnull])
        run(["ffmpeg", "-y", "-i", src, *common, "-pass", "2", "-c:a", "aac", "-b:a", "40k", "-ac", "1",
             "-movflags", "+faststart", dst])
        if os.path.getsize(dst) < limit_mb * 1e6:
            return
    raise SystemExit("could not get the small file under the limit")


# =============================================================== MAIN
def main():
    story_path, out = sys.argv[1], sys.argv[2]
    story = json.load(open(story_path, encoding="utf-8"))
    items = [{"text": x} if isinstance(x, str) else x for x in story["lines"]]
    work = os.path.join(os.path.dirname(os.path.abspath(out)), "work"); os.makedirs(work, exist_ok=True)
    seed = sum(map(ord, story["title"])) * 7919
    rng = np.random.default_rng(seed)
    amb = story.get("ambience", "rain")

    voice_tr, fx_tr, timeline, total, flashes = narrate(items, story.get("voice", "am_onyx"), work)
    if not any(it.get("sfx") == "stinger" for it in items):  # end on a hit
        s = sfx("stinger"); st = int(max(0, timeline[-1][1] - 2.4) * SR)
        fx_tr[st:st + len(s)] += s[:len(fx_tr) - st]
    bed = ambience(amb, len(voice_tr))
    # duck the bed under the voice
    voice_tr = norm(voice_tr, 0.95)
    venv = lp_fast(np.abs(voice_tr), 3); duck = 1 - np.clip(venv * 6, 0, 0.6)
    fx_duck = 1 - np.clip(venv * 8, 0, 0.75)          # effects drop well under the voice
    mix = voice_tr * 1.0 + fx_tr * 0.28 * fx_duck + bed * 0.7 * duck
    mix = np.tanh(mix * 1.2) * 0.92
    import soundfile as sf
    wav = os.path.join(work, "mix.wav"); sf.write(wav, mix.astype(np.float32), SR)
    make_ass(story["title"], timeline, os.path.join(work, "caps.ass"))

    # scene plan: change scene at lines that request it, else every ~13s
    cuts = [0.0]; kinds = [items[0].get("scene") or rng.choice(["road", "house", "forest", "graveyard", "field"])]
    for (st, en, txt, sc) in timeline[1:]:
        if sc:
            cuts.append(st); kinds.append(sc or rng.choice([k for k in ["road", "house", "forest", "graveyard", "hall", "field"] if k != kinds[-1]]))
    cuts.append(total)
    pal = PALETTES[int(rng.integers(0, len(PALETTES)))]
    rain_layer(os.path.join(work, "rain.png"), rng); vignette(os.path.join(work, "vig.png"))
    clips = []
    for i, k in enumerate(kinds):
        dur = cuts[i + 1] - cuts[i] + (0.8 if i < len(kinds) - 1 else 0)
        clip = os.path.join(work, f"c{i}.mp4")
        if isinstance(k, dict):   # 3D shot rendered with Blender
            sys.path.insert(0, HERE); import scene3d
            png = os.path.join(work, f"shot{i}.png")
            if not os.path.exists(png):
                scene3d.render_shot(k, png, 720, 1280, 32)
            outdoor = k.get("set") in ("forest", "road")
            still_clip(png, dur, i, amb == "rain" and outdoor, clip, work)
        else:                     # 2D parallax fallback
            folder = os.path.join(work, f"s{i}"); build_scene(rng, k, folder, pal)
            render_scene(folder, dur, i, amb == "rain" and k != "hall", clip)
        clips.append((clip, dur))
    inputs, fc, prev, off = [], [], "0:v", 0.0
    for c, _ in clips:
        inputs += ["-i", c]
    for i in range(1, len(clips)):
        off += clips[i - 1][1] - 0.8
        fc.append(f"[{prev}][{i}:v]xfade=transition=fade:duration=0.8:offset={off:.2f}[x{i}]"); prev = f"x{i}"
    bg = os.path.join(work, "bg.mp4")
    if len(clips) > 1:
        run(["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(fc), "-map", f"[{prev}]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", bg])
    else:
        os.replace(clips[0][0], bg)

    flash = "+".join(f"0.55*between(t\\,{f:.2f}\\,{f + 0.07:.2f})+0.3*between(t\\,{f + 0.16:.2f}\\,{f + 0.22:.2f})" for f in flashes) or "0"
    ass = os.path.join(work, "caps.ass").replace(":", "\\:")
    vf = (f"[0:v][1:v]overlay=0:0,eq=brightness='-0.01+{flash}':eval=frame,"
          f"ass={ass}:fontsdir=/usr/share/fonts/truetype/google-fonts[v]")
    run(["ffmpeg", "-y", "-i", bg, "-loop", "1", "-i", os.path.join(work, "vig.png"), "-i", wav, "-filter_complex", vf,
         "-map", "[v]", "-map", "2:a", "-t", f"{total:.2f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-maxrate", "6M", "-bufsize", "12M", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out])
    small = os.path.splitext(out)[0] + "_small.mp4"
    make_small(out, small, total)
    print(f"OK {out} {small} {total:.1f}s {os.path.getsize(small)} bytes")


if __name__ == "__main__":
    main()
