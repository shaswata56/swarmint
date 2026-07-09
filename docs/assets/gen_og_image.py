"""Generate docs/assets/og-image.png (1200x630 social card).

Run: ..\\.venv\\Scripts\\python.exe docs\\assets\\gen_og_image.py
Regenerate whenever the tagline or brand colors in docs/index.html change.
"""
import math
import random

from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 630
BG = (11, 16, 32)       # --bg dark
FG = (232, 236, 244)    # --fg dark
MUTED = (153, 163, 181)  # --muted dark
ACCENT = (106, 147, 255)  # --accent dark
OK = (70, 211, 138)      # --ok dark
LINE = (34, 42, 61)      # --line dark

FONT_DIR = "C:/WINDOWS/Fonts/"
title_font = ImageFont.truetype(FONT_DIR + "segoeuib.ttf", 64)
tagline_font = ImageFont.truetype(FONT_DIR + "segoeui.ttf", 30)
mono_font = ImageFont.truetype(FONT_DIR + "consola.ttf", 24)
badge_font = ImageFont.truetype(FONT_DIR + "consolab.ttf", 22)

random.seed(7)  # deterministic layout across regenerations

img = Image.new("RGB", (W, H), BG)
draw = ImageDraw.Draw(img, "RGBA")

# --- swarm-node network motif, right half, behind text ---
nodes = []
for i in range(26):
    x = random.uniform(620, 1150)
    y = random.uniform(60, 570)
    nodes.append((x, y))

# faint connecting gossip edges between nearby nodes
for i, (x1, y1) in enumerate(nodes):
    for j, (x2, y2) in enumerate(nodes):
        if j <= i:
            continue
        d = math.hypot(x2 - x1, y2 - y1)
        if d < 160:
            alpha = max(0, int(70 * (1 - d / 160)))
            draw.line([(x1, y1), (x2, y2)], fill=(106, 147, 255, alpha), width=2)

# nodes themselves; a few marked as byzantine/quarantined (dim/red-ish)
for i, (x, y) in enumerate(nodes):
    r = random.uniform(4, 9)
    if i % 9 == 0:
        color = (240, 110, 110, 200)  # flagged malicious node
    else:
        color = ACCENT + (220,)
    draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 255, 255, 60), width=1)

# --- left-side text block ---
pad = 72
y = 150

draw.text((pad, y), "swarmint", font=title_font, fill=FG)
y += 92

tagline_lines = [
    "Decentralized, Byzantine-robust",
    "swarm learning.",
]
for line in tagline_lines:
    draw.text((pad, y), line, font=tagline_font, fill=MUTED)
    y += 42

y += 26
sub_lines = [
    "Tiny prototype models gossip knowledge",
    "peer-to-peer. No central server.",
    "No gradient sharing. Poison-resistant.",
]
for line in sub_lines:
    draw.text((pad, y), line, font=mono_font, fill=(205, 216, 255))
    y += 34

# bottom badge row
badge_y = H - 90
badges = [
    ("AGPL-3.0-or-later", OK),
    ("Python 3.10+", ACCENT),
    ("swarmint.org", MUTED),
]
bx = pad
for text, color in badges:
    tw = draw.textlength(text, font=badge_font)
    bw = tw + 28
    bh = 40
    draw.rounded_rectangle([bx, badge_y, bx + bw, badge_y + bh], radius=8,
                            outline=color + (255,), width=2)
    draw.text((bx + 14, badge_y + 8), text, font=badge_font, fill=color)
    bx += bw + 16

# subtle top/bottom border matching site line color
draw.line([(0, 0), (W, 0)], fill=LINE, width=4)
draw.line([(0, H - 1), (W, H - 1)], fill=LINE, width=4)

out_path = "docs/assets/og-image.png"
img.save(out_path, "PNG")
print(f"wrote {out_path} ({W}x{H})")
