"""Render the Mote logo candidates: SVG sources, baked tiles, PNG previews, contact sheet.

    python brand/build.py        # rewrites brand/{marks,tiles,png}/ and brand/contact-sheet.html

Adopted: `boundary-ring`, turned so the gap sits at half past one, with optical sizes (see
`ring_params`). `web/icons/make_icons.py` imports `draw_boundary_ring` / `ring_svg` from here, so
this file is the single definition of the mark; the header (web/src/components/Wordmark.svelte) and
the pairing page (morpheme/serve/pairing.py) carry a copy of the 32-px SVG body.

Each mark is authored once, in 0-100 design units, twice over: an SVG body (currentColor, so the
tile decides the colour) and a Pillow draw function (so we get real pixels at real icon sizes
without a rasteriser dependency). Palette is the studio's, from web/src/app.css.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent

# --accent on --bg, in the two schemes.
LIGHT = {"fg": (163, 74, 31), "bg": (250, 249, 247)}   # rust on cream
DARK = {"fg": (224, 160, 112), "bg": (19, 18, 17)}     # peach on near-black

SIZES = [512, 180, 64, 32, 16]
RADIUS_FRAC = 0.22          # previews are rounded; real iOS/Android tiles ship square
SERIF = [
    r"C:\Windows\Fonts\georgia.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
]

# Boundary ring: an r=27 circle with one gap and the mote sitting in it. The gap is centred at
# GAP_AT degrees (image coordinates: 0 = 3 o'clock, clockwise), i.e. half past one — at twelve it
# reads as the power glyph. Stroke, dot and gap are optical: a 6-unit stroke is a hairline at 16 px.
RING_R = 27.0
GAP_AT = 315.0
RING_C = 2 * math.pi * RING_R


def ring_params(px):
    """(stroke width, dot radius, gap arc length) in design units for a tile rendered `px` wide."""
    if px is None or px >= 64:
        return 7.0, 9.0, 37.0
    if px >= 32:
        return 8.0, 9.5, 38.0
    return 11.0, 11.0, 42.0


def ring_dot():
    a = math.radians(GAP_AT)
    return 50 + RING_R * math.cos(a), 50 + RING_R * math.sin(a)


def ring_svg(stroke: float, dot_r: float, gap: float) -> str:
    """SVG body of the mark in currentColor; the dash pattern opens the gap at the path start (3
    o'clock) and the rotation carries it to GAP_AT."""
    dx, dy = ring_dot()
    return (
        f'<circle cx="50" cy="50" r="{RING_R:g}" fill="none" stroke="currentColor"'
        f' stroke-width="{stroke:g}" stroke-dasharray="{RING_C - gap:.2f} {gap:.2f}"'
        f' stroke-dashoffset="{RING_C - gap / 2:.2f}" transform="rotate({GAP_AT - 360:g} 50 50)"/>'
        f'<circle cx="{dx:.2f}" cy="{dy:.2f}" r="{dot_r:g}" fill="currentColor"/>'
    )


BYTE = [0, 1, 1, 0, 1, 1, 0, 1]                        # 0x6D, lowercase 'm'
BYTE_XS = [9, 31, 53, 75]
BYTE_YS = [31, 53]

ORDER = ["serif-m", "mote", "boundary-ring", "break", "byte-0x6d"]
TITLES = {
    "serif-m": "the studio's serif m (current icon)",
    "mote": "the mote",
    "boundary-ring": "the boundary ring (adopted)",
    "break": "the break",
    "byte-0x6d": "0x6D",
}


# --- SVG bodies (design units, currentColor) ----------------------------------------------------

def _byte_body() -> str:
    cells = []
    for i, bit in enumerate(BYTE):
        dim = "" if bit else ' fill-opacity="0.16"'
        cells.append(
            f'<rect x="{BYTE_XS[i % 4]}" y="{BYTE_YS[i // 4]}" width="16" height="16" rx="4"'
            f' fill="currentColor"{dim}/>'
        )
    return "".join(cells)


SVG = {
    "serif-m": '<text x="50" y="50" font-family="Georgia,&apos;Times New Roman&apos;,serif"'
               ' font-size="74" text-anchor="middle" dominant-baseline="central"'
               ' fill="currentColor">m</text>',
    "mote": '<circle cx="50" cy="50" r="14" fill="currentColor"/>',
    "boundary-ring": ring_svg(*ring_params(None)),
    "break": (
        '<path d="M12 50 H34" fill="none" stroke="currentColor" stroke-width="6"'
        ' stroke-linecap="round"/>'
        '<circle cx="50" cy="50" r="10.5" fill="currentColor"/>'
        '<path d="M66 50 H88" fill="none" stroke="currentColor" stroke-width="6"'
        ' stroke-linecap="round"/>'
    ),
    "byte-0x6d": _byte_body(),
}


# --- Pillow drawing (same design units, scaled) --------------------------------------------------

def _blend(fg, bg, a):
    return tuple(round(f * a + b * (1 - a)) for f, b in zip(fg, bg))


def _font(size: int):
    for path in SERIF:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def draw_serif_m(d, P, fg, bg, **_):
    """Scale the glyph by its ink box so the m fills 62% of the tile, as web/icons/make_icons.py."""
    f = _font(int(P * 0.9))
    l, t, r, b = d.textbbox((0, 0), "m", font=f, anchor="ls")
    f = _font(max(8, int(P * 0.9 * (P * 0.62) / (r - l))))
    l, t, r, b = d.textbbox((0, 0), "m", font=f, anchor="ls")
    d.text(((P - (r - l)) / 2 - l, (P - (b - t)) / 2 - t), "m", font=f, fill=fg, anchor="ls")


def draw_mote(d, P, fg, bg, **_):
    u = P / 100
    d.ellipse([36 * u, 36 * u, 64 * u, 64 * u], fill=fg)


def draw_boundary_ring(d, P, fg, bg, px=None, **_):
    """`P` is the canvas in pixels (supersampled), `px` the size the tile will be shown at."""
    u = P / 100
    stroke, dot_r, gap = ring_params(px)
    gap_deg = gap / RING_C * 360
    outer = RING_R + stroke / 2                        # Pillow strokes inward from the bbox
    box = [(50 - outer) * u, (50 - outer) * u, (50 + outer) * u, (50 + outer) * u]
    start = GAP_AT + gap_deg / 2                       # 0 deg is 3 o'clock, clockwise
    d.arc(box, start, start + 360 - gap_deg, fill=fg, width=max(1, round(stroke * u)))
    dx, dy = ring_dot()
    d.ellipse([(dx - dot_r) * u, (dy - dot_r) * u, (dx + dot_r) * u, (dy + dot_r) * u], fill=fg)


def draw_break(d, P, fg, bg, **_):
    u = P / 100
    for x0, x1 in [(12, 34), (66, 88)]:
        d.rounded_rectangle([x0 * u, 47 * u, x1 * u, 53 * u], radius=3 * u, fill=fg)
    d.ellipse([39.5 * u, 39.5 * u, 60.5 * u, 60.5 * u], fill=fg)


def draw_byte(d, P, fg, bg, **_):
    u = P / 100
    off = _blend(fg, bg, 0.16)                         # pre-blended: ImageDraw replaces, not blends
    for i, bit in enumerate(BYTE):
        x, y = BYTE_XS[i % 4] * u, BYTE_YS[i // 4] * u
        d.rounded_rectangle([x, y, x + 16 * u, y + 16 * u], radius=4 * u, fill=fg if bit else off)


DRAW = {
    "serif-m": draw_serif_m,
    "mote": draw_mote,
    "boundary-ring": draw_boundary_ring,
    "break": draw_break,
    "byte-0x6d": draw_byte,
}


# --- Output --------------------------------------------------------------------------------------

def png(name: str, px: int, scheme: dict, ss: int = 4) -> Image.Image:
    """Supersample, draw in design units, downsample — these are hairlines at 16px."""
    P = px * ss
    img = Image.new("RGBA", (P, P), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, P - 1, P - 1], radius=RADIUS_FRAC * P, fill=scheme["bg"])
    DRAW[name](d, P, scheme["fg"], scheme["bg"], px=px)
    return img.resize((px, px), Image.LANCZOS)


def hex_of(rgb) -> str:
    return "#%02x%02x%02x" % rgb


def svg_mark(name: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100"'
        f' role="img" aria-label="Mote mark: {TITLES[name]}"><title>{TITLES[name]}</title>'
        f"{SVG[name]}</svg>\n"
    )


def svg_tile(name: str, scheme: dict, label: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100"'
        f' role="img" aria-label="Mote mark: {TITLES[name]} ({label})">'
        f"<title>{TITLES[name]} - {label}</title>"
        f'<rect width="100" height="100" rx="{RADIUS_FRAC * 100:.0f}" fill="{hex_of(scheme["bg"])}"/>'
        f'<g color="{hex_of(scheme["fg"])}">{SVG[name]}</g></svg>\n'
    )


def contact_sheet() -> str:
    rows = []
    for name in ORDER:
        cells = [f"<th scope=\"row\">{TITLES[name]}</th>"]
        for scheme in (LIGHT, DARK):
            for px in (96, 32, 16):
                body = ring_svg(*ring_params(px)) if name == "boundary-ring" else SVG[name]
                cells.append(
                    f'<td><svg viewBox="0 0 100 100" width="{px}" height="{px}" aria-hidden="true"'
                    f' style="background:{hex_of(scheme["bg"])};color:{hex_of(scheme["fg"])};'
                    f'border-radius:{RADIUS_FRAC * px:.0f}px">{body}</svg></td>'
                )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    body = "".join(rows)
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\n"
        "<title>Mote - logo candidates</title>\n<style>\n"
        "  body { margin: 2.5rem; background: #f1eee8; color: #1c1a17;\n"
        "         font: 15px/1.6 system-ui, -apple-system, 'Segoe UI', sans-serif; }\n"
        "  h1 { font-weight: 500; font-size: 1.25rem; margin: 0 0 1.75rem; }\n"
        "  table { border-collapse: collapse; }\n"
        "  th, td { padding: 0.6rem 0.9rem; text-align: left; vertical-align: middle; }\n"
        "  thead th { font-weight: 400; font-size: 12px; color: #6f6960; }\n"
        "  tbody th { font-weight: 400; white-space: nowrap; padding-right: 1.5rem; }\n"
        "  tbody tr + tr { border-top: 1px solid #e3ded5; }\n"
        "  svg { display: block; }\n"
        "  @media (prefers-color-scheme: dark) {\n"
        "    body { background: #1d1b19; color: #ece8e2; }\n"
        "    thead th { color: #8b847c; }\n"
        "    tbody tr + tr { border-top-color: #2b2825; }\n"
        "  }\n</style></head>\n<body>\n<h1>Mote - logo candidates</h1>\n"
        "<table><thead><tr><th></th>"
        "<th scope=\"col\">light</th><th scope=\"col\">32</th><th scope=\"col\">16</th>"
        "<th scope=\"col\">dark</th><th scope=\"col\">32</th><th scope=\"col\">16</th>"
        f"</tr></thead>\n<tbody>{body}</tbody></table>\n</body></html>\n"
    )


def main() -> None:
    for sub in ("marks", "tiles", "png"):
        (ROOT / sub).mkdir(exist_ok=True)
    for name in ORDER:
        (ROOT / "marks" / f"{name}.svg").write_text(svg_mark(name), encoding="utf-8")
        for scheme, label in [(LIGHT, "light"), (DARK, "dark")]:
            (ROOT / "tiles" / f"{name}-{label}.svg").write_text(
                svg_tile(name, scheme, label), encoding="utf-8")
            for px in SIZES:
                png(name, px, scheme).save(ROOT / "png" / f"{name}-{px}-{label}.png", optimize=True)
    (ROOT / "contact-sheet.html").write_text(contact_sheet(), encoding="utf-8")
    print(f"wrote {len(ORDER)} marks, {len(ORDER) * 2} tiles, "
          f"{len(ORDER) * 2 * len(SIZES)} pngs, contact-sheet.html")


if __name__ == "__main__":
    main()
