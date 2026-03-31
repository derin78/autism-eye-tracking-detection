"""
Generate stimulus images for the autism screening test.
Each image is a split-screen with a FACE on one side and an OBJECT on the other.
The face side alternates between left and right.

Run:  python stimuli/generate_stimuli.py
"""

import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# Stimulus definitions: (face_description, object_description, face_side)
STIMULI = [
    ("👤 Face",    "🏠 House",     "left"),
    ("👤 Face",    "🚗 Car",       "right"),
    ("👤 Face",    "🌳 Tree",      "left"),
    ("👤 Face",    "📱 Phone",     "right"),
    ("👤 Face",    "⚽ Ball",      "left"),
    ("👤 Face",    "📚 Book",      "right"),
    ("👤 Face",    "🎸 Guitar",    "left"),
    ("👤 Face",    "🔑 Key",       "right"),
    ("👤 Face",    "🧸 Toy",       "left"),
    ("👤 Face",    "🎂 Cake",      "right"),
]

# Colors for the illustrations
FACE_BG = (255, 228, 196)      # Warm peach
OBJECT_BG = (200, 220, 240)    # Cool blue-gray
FACE_COLOR = (180, 120, 80)    # Skin tone
HAIR_COLOR = (60, 40, 30)      # Dark brown
EYE_COLOR = (40, 80, 40)       # Dark green
OBJECT_COLORS = [
    (180, 80,  80),   # House – red
    (80,  80,  180),  # Car – blue
    (60,  140, 60),   # Tree – green
    (100, 100, 100),  # Phone – gray
    (200, 200, 60),   # Ball – yellow
    (140, 100, 60),   # Book – brown
    (180, 100, 40),   # Guitar – orange
    (200, 180, 50),   # Key – gold
    (180, 130, 180),  # Toy – purple
    (220, 160, 120),  # Cake – tan
]


def draw_face(draw, cx, cy, size):
    """Draw a simple illustrated face."""
    r = size // 2

    # Head
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=FACE_COLOR, outline=(120, 80, 50), width=3)

    # Hair
    draw.arc([cx - r - 5, cy - r - 15, cx + r + 5, cy], 180, 0, fill=HAIR_COLOR, width=20)

    # Eyes
    er = size // 10
    ey = cy - size // 8
    draw.ellipse([cx - r // 3 - er, ey - er, cx - r // 3 + er, ey + er], fill="white", outline=(80, 80, 80))
    draw.ellipse([cx + r // 3 - er, ey - er, cx + r // 3 + er, ey + er], fill="white", outline=(80, 80, 80))
    # Pupils
    pr = er // 2
    draw.ellipse([cx - r // 3 - pr, ey - pr, cx - r // 3 + pr, ey + pr], fill=EYE_COLOR)
    draw.ellipse([cx + r // 3 - pr, ey - pr, cx + r // 3 + pr, ey + pr], fill=EYE_COLOR)

    # Nose
    draw.polygon([(cx, cy), (cx - size // 14, cy + size // 8), (cx + size // 14, cy + size // 8)],
                 fill=(160, 100, 70))

    # Mouth (smile)
    mw = size // 4
    draw.arc([cx - mw, cy + size // 10, cx + mw, cy + size // 3], 0, 180, fill=(180, 60, 60), width=3)


def draw_house(draw, cx, cy, size):
    s = size // 2
    # Walls
    draw.rectangle([cx - s, cy - s // 3, cx + s, cy + s], fill=(200, 100, 100), outline=(120, 60, 60), width=2)
    # Roof
    draw.polygon([(cx - s - 20, cy - s // 3), (cx, cy - s), (cx + s + 20, cy - s // 3)],
                 fill=(120, 60, 60), outline=(80, 40, 40), width=2)
    # Door
    draw.rectangle([cx - s // 5, cy + s // 5, cx + s // 5, cy + s], fill=(100, 60, 40))
    # Windows
    ww = s // 4
    draw.rectangle([cx - s + 15, cy - s // 6, cx - s + 15 + ww, cy + s // 8], fill=(180, 220, 255), outline=(80, 80, 80))
    draw.rectangle([cx + s - 15 - ww, cy - s // 6, cx + s - 15, cy + s // 8], fill=(180, 220, 255), outline=(80, 80, 80))


def draw_car(draw, cx, cy, size):
    s = size // 2
    # Body
    draw.rounded_rectangle([cx - s, cy, cx + s, cy + s // 2], radius=15, fill=(80, 80, 180), outline=(50, 50, 120), width=2)
    # Top
    draw.rounded_rectangle([cx - s // 2, cy - s // 3, cx + s // 2, cy + 5], radius=12, fill=(100, 100, 200), outline=(50, 50, 120), width=2)
    # Wheels
    draw.ellipse([cx - s + 10, cy + s // 3, cx - s + 50, cy + s // 2 + 15], fill=(40, 40, 40))
    draw.ellipse([cx + s - 50, cy + s // 3, cx + s - 10, cy + s // 2 + 15], fill=(40, 40, 40))
    # Windows
    draw.rectangle([cx - s // 3, cy - s // 5, cx - 5, cy - 2], fill=(180, 220, 255))
    draw.rectangle([cx + 5, cy - s // 5, cx + s // 3, cy - 2], fill=(180, 220, 255))


def draw_generic_object(draw, cx, cy, size, color, label):
    """Draw a simple geometric shape with label for objects we don't have specific drawings for."""
    s = size // 2
    # Main shape
    draw.rounded_rectangle([cx - s, cy - s, cx + s, cy + s], radius=20, fill=color, outline=(80, 80, 80), width=3)
    # Label text (use label without emoji)
    clean = label.split(" ")[-1] if " " in label else label
    try:
        font = ImageFont.truetype("arial.ttf", size // 4)
    except (OSError, IOError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), clean, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((cx - tw // 2, cy - th // 2), clean, fill="white", font=font)


# Specific draw functions per object index
OBJECT_DRAWERS = {
    0: draw_house,
    1: draw_car,
    # 2–9 use generic
}


def generate_all():
    """Generate all stimulus images."""
    os.makedirs(config.STIMULI_DIR, exist_ok=True)

    W, H = config.SCREEN_W, config.SCREEN_H
    half_w = W // 2

    for i, (face_label, obj_label, face_side) in enumerate(STIMULI):
        img = Image.new("RGB", (W, H), (30, 30, 30))
        draw = ImageDraw.Draw(img)

        # Background halves
        if face_side == "left":
            draw.rectangle([0, 0, half_w, H], fill=FACE_BG)
            draw.rectangle([half_w, 0, W, H], fill=OBJECT_BG)
            face_cx, face_cy = half_w // 2, H // 2
            obj_cx, obj_cy = half_w + half_w // 2, H // 2
        else:
            draw.rectangle([0, 0, half_w, H], fill=OBJECT_BG)
            draw.rectangle([half_w, 0, W, H], fill=FACE_BG)
            obj_cx, obj_cy = half_w // 2, H // 2
            face_cx, face_cy = half_w + half_w // 2, H // 2

        # Draw divider line
        draw.line([(half_w, 0), (half_w, H)], fill=(100, 100, 100), width=2)

        # Draw face
        face_size = min(half_w, H) // 2
        draw_face(draw, face_cx, face_cy, face_size)

        # Draw object
        obj_size = min(half_w, H) // 3
        obj_drawer = OBJECT_DRAWERS.get(i)
        if obj_drawer:
            obj_drawer(draw, obj_cx, obj_cy, obj_size)
        else:
            draw_generic_object(draw, obj_cx, obj_cy, obj_size, OBJECT_COLORS[i], obj_label)

        # Save
        path = os.path.join(config.STIMULI_DIR, f"stimulus_{i+1:02d}.png")
        img.save(path, "PNG")
        print(f"  ✅ {path} ({face_label} {face_side} | {obj_label})")

    print(f"\n✅ Generated {len(STIMULI)} stimulus images in {config.STIMULI_DIR}/")


if __name__ == "__main__":
    generate_all()
