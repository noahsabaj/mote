"""Render the app icons: a cream tile with a rust serif lowercase m (the studio's mark).

    python web/icons/make_icons.py      # writes web/public/icons/*.png

Georgia from the Windows font directory; falls back to DejaVu Serif / Pillow's default if absent.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CREAM = (250, 249, 247)
RUST = (163, 74, 31)
OUT = Path(__file__).resolve().parents[1] / "public" / "icons"
FONTS = [
    r"C:\Windows\Fonts\georgia.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
]


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONTS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def tile(px: int, glyph_frac: float = 0.62) -> Image.Image:
    """Opaque square: iOS rounds the corners itself; maskable icons keep the glyph in the safe zone."""
    img = Image.new("RGB", (px, px), CREAM)
    draw = ImageDraw.Draw(img)
    f = font(int(px * 0.9))
    # Scale the glyph by its ink box so the lowercase m fills `glyph_frac` of the tile width.
    l, t, r, b = draw.textbbox((0, 0), "m", font=f, anchor="ls")
    scale = (px * glyph_frac) / (r - l)
    f = font(max(8, int(px * 0.9 * scale)))
    l, t, r, b = draw.textbbox((0, 0), "m", font=f, anchor="ls")
    x = (px - (r - l)) / 2 - l
    y = (px - (b - t)) / 2 - t
    draw.text((x, y), "m", font=f, fill=RUST, anchor="ls")
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, px in [("apple-touch-icon.png", 180), ("icon-192.png", 192), ("icon-512.png", 512),
                     ("favicon-32.png", 32), ("favicon-64.png", 64)]:
        tile(px).save(OUT / name, optimize=True)
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
