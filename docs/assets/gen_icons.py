"""Generate the PNG favicon set from the same mesh motif as favicon.svg.

Run: ..\\.venv\\Scripts\\python.exe docs\\assets\\gen_icons.py
Keep in sync with docs/favicon.svg (three connected nodes, no center, on the
site's dark background). Drawn at 4x and downscaled for clean anti-aliasing.
"""
from PIL import Image, ImageDraw

BG = (11, 16, 32)        # --bg dark (#0b1020)
NODE = (106, 147, 255)   # --accent dark (#6a93ff)
HILITE = (205, 216, 255)  # --code-fg (#cdd8ff)
LINE = (106, 147, 255)

# geometry in a 64x64 space (matches favicon.svg)
TOP = (32, 18); BL = (17, 45); BR = (47, 45)
EDGES = [(TOP, BL), (TOP, BR), (BL, BR)]
NODES = [(TOP, 7, HILITE), (BL, 6, NODE), (BR, 6, NODE)]


def draw(size: int, rounded: bool = True, pad: float = 0.0) -> Image.Image:
    ss = 4
    big = size * ss
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # The 64-unit motif maps across the FULL canvas, minus optional padding
    # (touch icons look better with breathing room). Work in pixels throughout.
    off = pad * big
    scale = (big - 2 * off) / 64.0

    def pt(p):
        return (off + p[0] * scale, off + p[1] * scale)

    if rounded:
        d.rounded_rectangle([0, 0, big - 1, big - 1], radius=int(14 / 64 * big), fill=BG)

    lw = max(2, int(3 * scale))
    for a, b in EDGES:
        ax, ay = pt(a); bx, by = pt(b)
        d.line([ax, ay, bx, by], fill=LINE + (205,), width=lw)
    for center, rad, color in NODES:
        cx, cy = pt(center); rr = rad * scale
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=color)

    return img.resize((size, size), Image.LANCZOS)


OUT = "docs"
targets = [
    ("favicon-16.png", 16, True, 0.0),
    ("favicon-32.png", 32, True, 0.0),
    ("favicon-192.png", 192, True, 0.06),
    ("icon-512.png", 512, True, 0.06),
    ("apple-touch-icon.png", 180, True, 0.10),  # extra padding; iOS masks corners
]
for name, size, rounded, pad in targets:
    img = draw(size, rounded, pad)
    if name == "apple-touch-icon.png":
        # iOS ignores alpha and adds its own rounding — flatten onto solid bg.
        flat = Image.new("RGB", img.size, BG)
        flat.paste(img, (0, 0), img)
        flat.save(f"{OUT}/{name}", "PNG")
    else:
        img.save(f"{OUT}/{name}", "PNG")
    print(f"wrote {OUT}/{name} ({size}x{size})")
