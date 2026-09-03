"""Generate deterministic Pawprint ID PWA icons from simple vector primitives."""

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "public" / "icons"
SCALE = 4

INK = "#17201d"
PAPER = "#fcfdf9"
MINT = "#cdebdc"
GREEN = "#1e6b52"


def scaled_box(values: tuple[float, float, float, float], factor: float) -> tuple[int, int, int, int]:
    return tuple(round(value * factor) for value in values)  # type: ignore[return-value]


def draw_icon(size: int, filename: str, *, maskable: bool = False) -> None:
    canvas_size = size * SCALE
    image = Image.new("RGBA", (canvas_size, canvas_size), INK if maskable else (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    factor = canvas_size / 512

    if not maskable:
        draw.rounded_rectangle(
            (0, 0, canvas_size - 1, canvas_size - 1),
            radius=round(116 * factor),
            fill=INK,
        )

    def ellipse(box: tuple[float, float, float, float], color: str) -> None:
        draw.ellipse(scaled_box(box, factor), fill=color)

    ellipse((119, 133, 221, 235), MINT)
    ellipse((291, 133, 393, 235), MINT)
    ellipse((205, 77, 307, 179), PAPER)
    ellipse((119, 220, 393, 440), PAPER)
    ellipse((181, 274, 331, 404), GREEN)

    image = image.resize((size, size), Image.Resampling.LANCZOS)
    image.save(OUTPUT / filename, optimize=True)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    draw_icon(192, "icon-192.png")
    draw_icon(512, "icon-512.png")
    draw_icon(512, "icon-maskable-512.png", maskable=True)
    draw_icon(180, "apple-touch-icon.png")


if __name__ == "__main__":
    main()
