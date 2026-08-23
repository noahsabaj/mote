"""Render the app icons: the Mote mark (the boundary ring, turned) in rust on a cream tile, plus favicon.svg.

    python web/icons/make_icons.py      # writes web/public/icons/*.png and web/public/favicon.svg

The mark is defined once, in brand/build.py (`draw_boundary_ring`, `ring_svg`, `ring_params`); this
script only lays it on tiles at the sizes web/index.html and the manifest link. Opaque square tiles:
iOS rounds the corners itself, and maskable icons keep the mark inside the safe zone.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "web" / "public" / "icons"
FAVICON = ROOT / "web" / "public" / "favicon.svg"

_spec = importlib.util.spec_from_file_location("brand_build", ROOT / "brand" / "build.py")
brand = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(brand)

CREAM, RUST = brand.LIGHT["bg"], brand.LIGHT["fg"]
PEACH = brand.DARK["fg"]


def tile(px: int, ss: int = 4) -> Image.Image:
    """Supersample, draw at the optical weights for `px`, downsample."""
    P = px * ss
    img = Image.new("RGB", (P, P), CREAM)
    brand.draw_boundary_ring(ImageDraw.Draw(img), P, RUST, CREAM, px=px)
    return img.resize((px, px), Image.LANCZOS)


def favicon_svg() -> str:
    """Transparent, the 16-px weights; rust in light tabs, peach in dark ones where the browser honours it."""
    stroke, dot_r, gap = brand.ring_params(16)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" role="img" aria-label="Mote">'
        "<style>svg{color:%s}@media(prefers-color-scheme:dark){svg{color:%s}}</style>"
        % (brand.hex_of(RUST), brand.hex_of(PEACH))
        + brand.ring_svg(stroke, dot_r, gap)
        + "</svg>\n"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, px in [("apple-touch-icon.png", 180), ("icon-192.png", 192), ("icon-512.png", 512),
                     ("favicon-16.png", 16), ("favicon-32.png", 32), ("favicon-64.png", 64)]:
        tile(px).save(OUT / name, optimize=True)
        print("wrote", OUT / name)
    FAVICON.write_text(favicon_svg(), encoding="utf-8")
    print("wrote", FAVICON)


if __name__ == "__main__":
    main()
