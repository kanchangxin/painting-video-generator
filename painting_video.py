#!/usr/bin/env python3
"""Turn one still image into a forward-only, hand-painted-process MP4."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


BRUSHES = ("pencil", "watercolor", "marker")


@dataclass
class BrushStroke:
    points: list[tuple[float, float]]
    width: int
    region: int | None = None

    @property
    def length(self) -> float:
        return sum(math.dist(a, b) for a, b in zip(self.points, self.points[1:]))


@dataclass
class PaintPhase:
    name: str
    target: np.ndarray
    strokes: list[BrushStroke]
    weight: float
    region_mask: np.ndarray | None = None
    stage: str = ""


def _hand_exit_pose(
    frame_size: tuple[int, int],
    tip: tuple[float, float],
    previous_side: int | None,
) -> tuple[float, int]:
    """Choose an arm exit direction from the pen tip's half of the frame.

    Side -1 means left for portrait video and top for landscape video; side 1
    means right and bottom respectively. A small deadband prevents chatter at
    the midpoint while the brush moves around it.
    """
    width, height = frame_size
    portrait = height >= width
    axis_length = width if portrait else height
    coordinate = tip[0] if portrait else tip[1]
    midpoint = axis_length / 2
    deadband = max(8.0, axis_length * 0.04)

    if previous_side is None:
        side = -1 if coordinate < midpoint else 1
    elif previous_side < 0:
        side = 1 if coordinate > midpoint + deadband else -1
    else:
        side = -1 if coordinate < midpoint - deadband else 1

    if portrait:
        angle = math.pi if side < 0 else 0.0
    else:
        angle = -math.pi / 2 if side < 0 else math.pi / 2
    return angle, side


@dataclass
class PhotoHandRenderer:
    """Scaled photographed hand, rotated around the real marker-tip anchor."""
    sprite: Image.Image
    anchor: tuple[int, int]
    source_angle: float
    exit_distance: float
    cache: dict[int, tuple[Image.Image, tuple[int, int], Image.Image]] = field(default_factory=dict)
    exit_side: int | None = None

    def draw(self, frame: np.ndarray, tip: tuple[float, float]) -> np.ndarray:
        height, width = frame.shape[:2]
        target_angle, self.exit_side = _hand_exit_pose((width, height), tip, self.exit_side)
        pil_angle = math.degrees(self.source_angle - target_angle)
        angle_key = round(pil_angle / 4)  # small angle quantization avoids per-frame resampling noise
        if angle_key not in self.cache:
            angle = angle_key * 4
            sw, sh = self.sprite.size
            side = math.ceil(math.hypot(sw, sh)) + 4
            center = side // 2
            base = Image.new("RGBA", (side, side), (0, 0, 0, 0))
            base.alpha_composite(self.sprite, (center - self.anchor[0], center - self.anchor[1]))
            rotated = base.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
            bbox = rotated.getchannel("A").getbbox()
            if bbox:
                cropped = rotated.crop(bbox)
                rotated_anchor = (center - bbox[0], center - bbox[1])
            else:
                cropped, rotated_anchor = self.sprite, self.anchor
            alpha = cropped.getchannel("A").filter(ImageFilter.GaussianBlur(5))
            alpha = ImageEnhance.Brightness(alpha).enhance(0.28)
            shadow_alpha = Image.new("L", alpha.size, 0)
            shadow_alpha.paste(alpha, (4, 7))
            shadow = Image.new("RGBA", cropped.size, (30, 24, 21, 0))
            shadow.putalpha(shadow_alpha)
            self.cache[angle_key] = (cropped, rotated_anchor, shadow)
        hand, anchor, shadow = self.cache[angle_key]
        x = round(tip[0] - anchor[0])
        y = round(tip[1] - anchor[1])
        canvas = Image.fromarray(frame, "RGB").convert("RGBA")
        canvas.alpha_composite(shadow, (x + 2, y + 3))
        canvas.alpha_composite(hand, (x, y))
        return np.asarray(canvas.convert("RGB"), dtype=np.uint8)


def _prepare_photo_hand(
    hand_image: Image.Image | None,
    frame_size: tuple[int, int],
    hand_scale: float = 1.0,
) -> PhotoHandRenderer | None:
    if hand_image is None:
        return None
    width, height = frame_size
    source_width, source_height = hand_image.size
    source_anchor = (source_width * (38 / 1041), source_height * (41 / 1437))
    source_exit = (source_width * 0.96, source_height * 0.97)
    source_exit_distance = math.dist(source_anchor, source_exit)
    axis_length = width if height >= width else height
    required_exit_distance = axis_length / 2 + max(8.0, axis_length * 0.04)
    # The long forearm is intentionally allowed to run beyond the canvas edge,
    # as it would when a real artist sits just outside the camera frame.
    requested_height = round(min(height * 0.92, width * 0.85) * hand_scale)
    minimum_exit_height = math.ceil(required_exit_distance * source_height / source_exit_distance)
    target_height = max(100, requested_height, minimum_exit_height)
    target_width = round(target_height * hand_image.width / hand_image.height)
    sprite = hand_image.convert("RGBA").resize((target_width, target_height), Image.Resampling.LANCZOS)
    anchor = (round(target_width * (38 / 1041)), round(target_height * (41 / 1437)))
    arm_exit = (round(target_width * 0.96), round(target_height * 0.97))
    exit_distance = math.dist(anchor, arm_exit)
    source_angle = math.atan2(arm_exit[1] - anchor[1], arm_exit[0] - anchor[0])
    return PhotoHandRenderer(sprite, anchor, source_angle, exit_distance)


def _fit_image(image: Image.Image, max_side: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGBA")
    background = Image.new("RGBA", image.size, (248, 244, 234, 255))
    background.alpha_composite(image)
    image = background.convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    width, height = image.size
    return image.resize((width - width % 2, height - height % 2), Image.Resampling.LANCZOS)


def _curve(center: tuple[float, float], angle: float, length: float, bend: float,
           width: int, region: int | None = None) -> BrushStroke:
    """Create a slightly irregular, hand-drawn cubic-like path."""
    cx, cy = center
    dx, dy = math.cos(angle), math.sin(angle)
    nx, ny = -dy, dx
    points = []
    for step in range(7):
        u = step / 6 - 0.5
        curve = bend * (1.0 - (2 * u) ** 2)
        points.append((cx + dx * length * u + nx * curve,
                       cy + dy * length * u + ny * curve))
    return BrushStroke(points, width, region)


def _edge_strokes(image: Image.Image, count: int, length: float, width: int,
                  seed: int, threshold: float) -> list[BrushStroke]:
    # A small blur suppresses texture and lettering noise so the strokes track
    # larger shapes in the drawing rather than every tiny printed edge.
    gray = np.asarray(ImageOps.grayscale(image.filter(ImageFilter.GaussianBlur(1.4))), dtype=np.float32)
    gy, gx = np.gradient(gray)
    magnitude = np.hypot(gx, gy)
    cutoff = np.percentile(magnitude, threshold)
    ys, xs = np.where(magnitude >= cutoff)
    if len(xs) == 0:
        return []
    rng = np.random.default_rng(seed)
    strength = magnitude[ys, xs]
    # Prefer clear boundaries and space the seeds out so the result reads as
    # deliberate contour drawing instead of a field of unrelated dashes.
    order = np.argsort(strength + rng.random(len(strength)) * max(1.0, float(strength.max()) * 0.12))[::-1]
    seeds: list[tuple[float, float]] = []
    min_gap = max(9, round(length * 0.32))
    for index in order:
        x, y = float(xs[index]), float(ys[index])
        if all((x - sx) ** 2 + (y - sy) ** 2 > min_gap ** 2 for sx, sy in seeds):
            seeds.append((x, y))
            if len(seeds) >= count:
                break

    h, w = gray.shape
    def trace(seed_x: float, seed_y: float, direction: float, steps: int) -> list[tuple[float, float]]:
        x, y = seed_x, seed_y
        points = [(x, y)]
        step_size = max(2.0, length / max(8, steps * 2))
        for _ in range(steps):
            ix = int(np.clip(round(x), 0, w - 1))
            iy = int(np.clip(round(y), 0, h - 1))
            # Tangent direction is perpendicular to the local gradient.
            angle = math.atan2(float(gx[iy, ix]), -float(gy[iy, ix]))
            vx, vy = math.cos(angle), math.sin(angle)
            if vx * math.cos(direction) + vy * math.sin(direction) < 0:
                vx, vy = -vx, -vy
            x += vx * step_size
            y += vy * step_size
            if x < 0 or y < 0 or x >= w or y >= h:
                break
            points.append((x, y))
            direction = math.atan2(vy, vx)
        return points

    strokes = []
    for x, y in seeds:
        angle = math.atan2(float(gx[round(y), round(x)]), -float(gy[round(y), round(x)]))
        steps = 6
        backward = trace(x, y, angle + math.pi, steps)
        forward = trace(x, y, angle, steps)
        points = list(reversed(backward[1:])) + forward
        if len(points) >= 2:
            strokes.append(BrushStroke(points, width))
    return strokes


def _shadow_strokes(image: Image.Image, count: int, width: int, seed: int) -> list[BrushStroke]:
    gray = np.asarray(ImageOps.grayscale(image), dtype=np.uint8)
    dark = gray < np.percentile(gray, 43)
    ys, xs = np.where(dark)
    if len(xs) == 0:
        return []
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(xs), size=min(count, len(xs)), replace=False)
    return [_curve((float(xs[i]), float(ys[i])), rng.uniform(-0.35, 0.35),
                   rng.uniform(18, 48), rng.uniform(-2.5, 2.5), width) for i in chosen]


def _coverage_strokes(width: int, height: int, band: int, seed: int,
                      center_out: bool = False) -> list[BrushStroke]:
    rng = np.random.default_rng(seed)
    spacing = max(8, band * 0.72)
    rows = np.arange(-band / 2, height + band / 2, spacing)
    if center_out:
        mid = len(rows) // 2
        order = [mid]
        for distance in range(1, len(rows)):
            if mid + distance < len(rows):
                order.append(mid + distance)
            if mid - distance >= 0:
                order.append(mid - distance)
        rows = rows[order]
    strokes = []
    for i, y in enumerate(rows):
        left_to_right = i % 2 == 0
        start_x, end_x = (-band / 2, width + band / 2) if left_to_right else (width + band / 2, -band / 2)
        bend = rng.uniform(-band * 0.10, band * 0.10)
        points = []
        for j in range(9):
            u = j / 8
            x = start_x + (end_x - start_x) * u
            yy = y + bend * math.sin(u * math.pi) + rng.uniform(-1, 1) * band * 0.025
            points.append((x, yy))
        strokes.append(BrushStroke(points, band))
    return strokes


def _wash_strokes(width: int, height: int, band: int, seed: int) -> list[BrushStroke]:
    """Broad overlapping patches for the first wash, avoiding scanline reveals."""
    rng = np.random.default_rng(seed)
    spacing_y = max(12, band * 0.72)
    rows = list(np.arange(-band / 2, height + band / 2, spacing_y))
    middle = len(rows) // 2
    rows = [rows[middle], *[rows[i] for d in range(1, len(rows))
                            for i in (middle + d, middle - d) if 0 <= i < len(rows)]]
    pieces = max(3, round(width / max(150, band * 5)))
    strokes = []
    for row_index, y in enumerate(rows):
        edges = np.linspace(-band * 0.45, width + band * 0.45, pieces + 1)
        columns = list(range(pieces))
        rng.shuffle(columns)
        for column in columns:
            x0, x1 = edges[column], edges[column + 1]
            overlap = (x1 - x0) * 0.12
            if column > 0:
                x0 -= overlap
            if column < pieces - 1:
                x1 += overlap
            if rng.random() < 0.5:
                x0, x1 = x1, x0
            curve = rng.uniform(-band * 0.20, band * 0.20)
            points = []
            for step in range(7):
                u = step / 6
                x = x0 + (x1 - x0) * u
                yy = y + curve * math.sin(math.pi * u) + rng.uniform(-1, 1) * band * 0.035
                points.append((x, yy))
            strokes.append(BrushStroke(points, band))
    return strokes


def _path_position(points: list[tuple[float, float]], fraction: float) -> tuple[tuple[float, float], tuple[float, float]]:
    lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
    total = max(sum(lengths), 1e-6)
    remaining = min(1.0, max(0.0, fraction)) * total
    for a, b, length in zip(points, points[1:], lengths):
        if remaining <= length or length == lengths[-1]:
            local = min(1.0, remaining / max(length, 1e-6))
            tangent = ((b[0] - a[0]) / max(length, 1e-6), (b[1] - a[1]) / max(length, 1e-6))
            return ((a[0] + (b[0] - a[0]) * local, a[1] + (b[1] - a[1]) * local), tangent)
        remaining -= length
    return points[-1], (1.0, 0.0)


def _draw_stroke(
    draw: ImageDraw.ImageDraw,
    stroke: BrushStroke,
    fraction: float,
    brush: str,
    preserve_width: bool = False,
) -> None:
    fraction = min(1.0, max(0.0, fraction))
    if fraction <= 0:
        return
    points = stroke.points
    total = stroke.length
    stop = fraction * total
    visible = [points[0]]
    walked = 0.0
    for a, b in zip(points, points[1:]):
        length = math.dist(a, b)
        if walked + length >= stop:
            part = max(0.0, (stop - walked) / max(length, 1e-6))
            visible.append((a[0] + (b[0] - a[0]) * part, a[1] + (b[1] - a[1]) * part))
            break
        visible.append(b)
        walked += length
    line_width = max(1, stroke.width)
    if not preserve_width and brush == "pencil":
        line_width = max(1, round(line_width * 0.44))
    elif not preserve_width and brush == "watercolor":
        line_width = max(1, round(line_width * 1.55))
    elif not preserve_width and brush == "marker":
        line_width = max(1, round(line_width * 1.7))
    coords = [(round(x), round(y)) for x, y in visible]
    draw.line(coords, fill=255, width=line_width, joint="curve")
    radius = max(0, line_width // 2)
    if radius:
        x, y = coords[-1]
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=255)


def _stroke_mask(phase: PaintPhase, amount: float, size: tuple[int, int], brush: str) -> tuple[Image.Image, tuple[float, float], tuple[float, float]]:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    final_color_pass = phase.name == "全彩图层收尾"
    pencil_color_block = brush == "pencil" and phase.stage in (
        "背景大色块", "主体铺色", "阴影与受光",
    )
    preserve_width = final_color_pass or pencil_color_block
    travel_lengths = [math.dist(a.points[-1], b.points[0]) * 0.28
                      for a, b in zip(phase.strokes, phase.strokes[1:])]
    total = sum(max(1.0, s.length) for s in phase.strokes) + sum(travel_lengths)
    distance = min(1.0, max(0.0, amount)) * total
    cursor: tuple[float, float] = (size[0] / 2, size[1] / 2)
    tangent = (1.0, 0.0)
    if phase.strokes and amount <= 0:
        first = phase.strokes[0].points
        dx, dy = first[1][0] - first[0][0], first[1][1] - first[0][1]
        norm = math.hypot(dx, dy) or 1.0
        cursor, tangent = first[0], (dx / norm, dy / norm)
    for index, stroke in enumerate(phase.strokes):
        length = max(1.0, stroke.length)
        if distance >= length:
            _draw_stroke(draw, stroke, 1.0, brush, preserve_width=preserve_width)
            distance -= length
            cursor, tangent = _path_position(stroke.points, 1.0)
        elif distance > 0:
            part = distance / length
            _draw_stroke(draw, stroke, part, brush, preserve_width=preserve_width)
            cursor, tangent = _path_position(stroke.points, part)
            distance = 0
            break
        else:
            break
        if index < len(travel_lengths):
            travel = travel_lengths[index]
            start, end = stroke.points[-1], phase.strokes[index + 1].points[0]
            if distance >= travel:
                distance -= travel
                cursor = end
                dx, dy = end[0] - start[0], end[1] - start[1]
                norm = math.hypot(dx, dy) or 1
                tangent = (dx / norm, dy / norm)
            elif travel > 0:
                progress = distance / travel
                dx, dy = end[0] - start[0], end[1] - start[1]
                cursor = (start[0] + dx * progress, start[1] + dy * progress)
                norm = math.hypot(dx, dy) or 1
                tangent = (dx / norm, dy / norm)
                distance = 0
                break
    if brush == "watercolor":
        if not final_color_pass:
            radius = phase.strokes[0].width * 0.28 if phase.strokes else 1.5
            mask = mask.filter(ImageFilter.GaussianBlur(max(1.0, radius)))
    elif brush == "pencil" and not final_color_pass and not pencil_color_block:
        # Small deterministic gaps give graphite strokes a dry, grainy edge.
        noise = np.random.default_rng(91).random((size[1], size[0]))
        grain = Image.fromarray(np.uint8(noise > 0.11) * 255, "L")
        mask = Image.fromarray(np.minimum(np.asarray(mask), np.asarray(grain)))
    if phase.region_mask is not None:
        region = Image.fromarray(np.uint8(phase.region_mask) * 255, "L")
        mask = Image.fromarray(np.minimum(np.asarray(mask), np.asarray(region)))
    return mask, cursor, tangent


def _draw_hand_and_brush(frame: np.ndarray, tip: tuple[float, float], tangent: tuple[float, float], brush: str) -> np.ndarray:
    """Draw a softly shaded, antialiased hand and brush at the active tip."""
    height, width = frame.shape[:2]
    supersample = 2
    overlay = Image.new("RGBA", (width * supersample, height * supersample), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    tx, ty = tip
    ux, uy = -tangent[0], -tangent[1]
    norm = math.hypot(ux, uy) or 1
    ux, uy = ux / norm, uy / norm
    vx, vy = -uy, ux
    scale = max(0.72, min(1.30, min(width, height) / 536))

    def p(back: float, side: float) -> tuple[int, int]:
        return (round((tx + scale * (ux * back + vx * side)) * supersample),
                round((ty + scale * (uy * back + vy * side)) * supersample))

    def sw(value: float) -> int:
        return max(1, round(value * scale * supersample))

    def ellipse_local(back: float, side: float, rx: float, ry: float,
                     color: tuple[int, int, int, int], outline=None, outline_width=1):
        vertices = []
        for i in range(24):
            angle = 2 * math.pi * i / 24
            vertices.append(p(back + rx * math.cos(angle), side + ry * math.sin(angle)))
        draw.polygon(vertices, fill=color)
        if outline:
            draw.line(vertices + [vertices[0]], fill=outline, width=sw(outline_width), joint="curve")

    def poly(coords: list[tuple[float, float]], fill: tuple[int, int, int, int], outline=(50, 40, 36, 255), stroke=2):
        pixels = [p(a, b) for a, b in coords]
        draw.polygon(pixels, fill=fill)
        if stroke:
            draw.line(pixels + [pixels[0]], fill=outline, width=sw(stroke), joint="curve")

    result = Image.fromarray(frame, "RGB").convert("RGBA")

    # Brush body, metal ferrule and tapered bristles use warm specular edges.
    draw.line((p(0, 0), p(13, 0)), fill=(43, 34, 29, 255), width=sw(5))
    draw.line((p(12, 0), p(29, 0)), fill=(163, 143, 109, 255), width=sw(9))
    draw.line((p(14, -3), p(27, -3)), fill=(224, 206, 166, 255), width=sw(2))
    draw.line((p(25, 0), p(92, 0)), fill=(82, 48, 31, 255), width=sw(13))
    draw.line((p(34, -3), p(85, -3)), fill=(187, 133, 79, 255), width=sw(2))
    draw.line((p(39, 3), p(82, 3)), fill=(46, 34, 29, 180), width=sw(1))

    # Sleeve, wrist and palm are layered around the handle to read as a grip.
    poly([(116, -31), (137, -23), (108, 30), (86, 16)], (115, 120, 113, 255), (69, 72, 67, 170), 1)
    poly([(94, -23), (117, -17), (94, 26), (72, 12)], (192, 151, 119, 255), (137, 95, 69, 140), 1)
    draw.line((p(91, -19), p(111, -14)), fill=(232, 199, 161, 220), width=sw(2))
    poly([(54, -28), (72, -34), (93, -22), (103, -9), (96, 14), (78, 27), (57, 24), (43, 9)],
         (205, 145, 108, 255), (128, 79, 57, 100), 1)
    # Soft palm highlights and shaded creases give the grip more volume.
    ellipse_local(72, -14, 17, 9, (238, 183, 145, 150))
    draw.line((p(62, 15), p(74, 13), p(85, 7)), fill=(151, 91, 65, 95), width=sw(1.2), joint="curve")
    # Four curved fingers wrap over the shaft, with restrained nail highlights.
    for i, back in enumerate((44, 53, 62, 71)):
        # Curved finger pads fold over the shaft and disappear into the palm.
        finger = (p(back - 3, -21), p(back - 1, -14), p(back + 3, -6), p(back + 5, 2))
        draw.line(finger, fill=(170, 104, 75, 105), width=sw(13), joint="curve")
        draw.line(finger, fill=(218 - i * 2, 163 - i, 126, 255), width=sw(10), joint="curve")
        nail = (p(back - 3, -20), p(back, -20), p(back + 1, -17), p(back - 2, -16))
        draw.polygon(nail, fill=(239, 198, 171, 220))
    # The thumb arches from the palm across the handle to close the grip.
    thumb = (p(34, 17), p(31, 10), p(35, 3), p(43, -3), p(50, -4))
    draw.line(thumb, fill=(158, 92, 66, 115), width=sw(18), joint="curve")
    draw.line(thumb, fill=(219, 158, 121, 255), width=sw(15), joint="curve")
    draw.line((p(35, 12), p(39, 5), p(46, 0)), fill=(245, 199, 163, 190), width=sw(2), joint="curve")

    # A soft cast shadow follows the actual silhouette, avoiding a rectangular
    # halo and keeping the hand visually grounded on the painted surface.
    alpha = overlay.getchannel("A").filter(ImageFilter.GaussianBlur(max(2, round(8 * scale * supersample))))
    alpha = ImageEnhance.Brightness(alpha).enhance(0.34)
    shadow_alpha = Image.new("L", alpha.size, 0)
    shadow_alpha.paste(alpha, (round(4 * scale * supersample), round(7 * scale * supersample)))
    shadow = Image.new("RGBA", overlay.size, (44, 33, 26, 0))
    shadow.putalpha(shadow_alpha)
    shadow = shadow.resize((width, height), Image.Resampling.LANCZOS)
    result.alpha_composite(shadow)
    rgba = np.asarray(overlay, dtype=np.int16).copy()
    texture = np.random.default_rng(606).normal(0, 1.8, (height * supersample, width * supersample, 1))
    rgba[:, :, :3] = np.clip(rgba[:, :, :3] + texture, 0, 255)
    overlay = Image.fromarray(np.uint8(rgba), "RGBA").resize((width, height), Image.Resampling.LANCZOS)
    result.alpha_composite(overlay)
    return np.asarray(result.convert("RGB"), dtype=np.uint8)


def _make_phases(image: Image.Image, brush: str) -> list[PaintPhase]:
    width, height = image.size
    source = np.asarray(image, dtype=np.uint8)
    gray = ImageOps.grayscale(image)
    sketch_target = ImageOps.colorize(gray, (83, 77, 72), (224, 216, 201)).convert("RGB")
    line_target = ImageOps.colorize(gray, (42, 35, 33), (235, 226, 210)).convert("RGB")

    # Quantized color regions supply broad, image-aware paint areas. Filling
    # common regions first approximates background masses; smaller regions
    # then build the subject and secondary details without a full-frame wash.
    limited = image.quantize(colors=8, method=Image.Quantize.MEDIANCUT)
    labels = np.asarray(limited, dtype=np.uint8)
    palette = np.asarray(limited.getpalette(), dtype=np.uint8).reshape(-1, 3)
    regions = []
    for label in np.unique(labels):
        region = labels == label
        if region.any():
            color = tuple(int(c) for c in palette[label])
            target = np.empty_like(source)
            target[:] = color
            regions.append((int(region.sum()), target, region))
    regions.sort(key=lambda item: item[0], reverse=True)

    scale = min(width, height) / 536
    composition_image = image.filter(ImageFilter.GaussianBlur(max(3, width / 105)))
    rough = _edge_strokes(composition_image, 82, 92 * scale, max(2, round(3 * scale)), 12, 81)
    phases = [
        PaintPhase(
            "构图草图与明暗关系", np.asarray(sketch_target, dtype=np.uint8), rough,
            0.13, stage="构图与明暗",
        ),
    ]

    # Split the largest color masses from the smaller subject/detail regions.
    # The threshold is area-based so the same logic works across image sizes.
    background_regions = []
    subject_regions = []
    cumulative_area = 0
    background_limit = round(width * height * 0.56)
    for region_data in regions:
        if cumulative_area < background_limit:
            background_regions.append(region_data)
            cumulative_area += region_data[0]
        else:
            subject_regions.append(region_data)
    if not subject_regions and len(background_regions) > 1:
        subject_regions.append(background_regions.pop())

    broad_band = max(18, round(min(width, height) * 0.048))
    detail_band = max(14, round(min(width, height) * 0.034))
    for index, (_, target, region) in enumerate(background_regions):
        phases.append(PaintPhase(
            f"背景大色块{index + 1}", target,
            _coverage_strokes(width, height, broad_band, 40 + index),
            0.34 / max(1, len(background_regions)), region,
            stage="背景大色块",
        ))
    for index, (_, target, region) in enumerate(subject_regions):
        phases.append(PaintPhase(
            f"主体色块{index + 1}", target,
            _coverage_strokes(width, height, detail_band, 60 + index),
            0.30 / max(1, len(subject_regions)), region,
            stage="主体铺色",
        ))

    gray_values = np.asarray(gray, dtype=np.uint8)
    dark_cut = np.percentile(gray_values, 42)
    light_cut = np.percentile(gray_values, 76)
    dark_region = gray_values <= dark_cut
    light_region = gray_values >= light_cut
    tone_band = max(15, round(min(width, height) * 0.030))
    shadows = _coverage_strokes(width, height, tone_band, 34, True)
    highlights = _coverage_strokes(width, height, tone_band, 35, True)
    shadow_target = np.asarray(ImageEnhance.Brightness(image).enhance(0.63), dtype=np.uint8)
    highlight_target = np.asarray(ImageEnhance.Brightness(image).enhance(1.18), dtype=np.uint8)
    phases.extend([
        PaintPhase("暗部塑形", shadow_target, shadows, 0.065, dark_region, "阴影与受光"),
        PaintPhase("受光提亮", highlight_target, highlights, 0.045, light_region, "阴影与受光"),
    ])

    # Contours and small high-contrast features (including text and icons)
    # arrive late, after the large forms and color relationships are readable.
    fine = _edge_strokes(image, 168, 43 * scale, max(1, round(1.5 * scale)), 23, 72)
    phases.append(PaintPhase(
        "轮廓线与文字图标", np.asarray(line_target, dtype=np.uint8), fine,
        0.095, stage="线条与最终细节",
    ))
    phases.extend([
        PaintPhase(
            "全彩图层收尾", source,
            _coverage_strokes(width, height, max(22, round(min(width, height) * 0.045)), 101, True),
            0.22, stage="线条与最终细节",
        ),
    ])
    total_weight = sum(phase.weight for phase in phases)
    for phase in phases:
        phase.weight /= total_weight
    return phases


def build_frames(
    image: Image.Image,
    frame_count: int,
    effect: str = "sequential",
    brush: str = "watercolor",
    size: int = 960,
    hand_image: Image.Image | None = None,
    hand_scale: float = 1.0,
) -> Iterator[np.ndarray]:
    """Yield RGB frames from progressive, individually animated brush strokes."""
    if frame_count < 2:
        raise ValueError("视频至少需要 2 帧，请增大时长或帧率。")
    if effect not in ("sequential", "interleaved"):
        raise ValueError(f"未知进程类型: {effect}")
    if brush not in BRUSHES:
        raise ValueError(f"未知笔刷: {brush}")

    canvas = _fit_image(image, size)
    hand_renderer = _prepare_photo_hand(hand_image, canvas.size, hand_scale)
    paper_rng = np.random.default_rng(404)
    grain = paper_rng.normal(0.0, 1.25, (canvas.height, canvas.width, 1))
    paper = np.clip(np.full((canvas.height, canvas.width, 3), (239, 235, 223), dtype=np.float32) + grain, 0, 255).astype(np.uint8)
    phases = _make_phases(canvas, brush)
    weights = np.asarray([phase.weight for phase in phases], dtype=np.float64)
    cumulative = np.cumsum(weights)
    size_xy = canvas.size

    for frame_index in range(frame_count):
        t = frame_index / (frame_count - 1)
        # Reserve the last 6% of the clip for a visible brush lift and hand
        # exit. The painting itself is fully finished before that movement.
        drawing_t = min(1.0, t / 0.94)
        result = paper.copy()
        progresses = []
        for index, phase in enumerate(phases):
            if effect == "sequential":
                amount = np.clip((drawing_t - (cumulative[index] - weights[index])) / weights[index], 0, 1)
            else:
                # Each pass starts slightly before the previous one ends, but never reverses.
                start = max(0.0, cumulative[index] - weights[index] * 1.65)
                amount = np.clip((drawing_t - start) / max(1e-6, 1 - start), 0, 1)
            progresses.append(float(amount))
            if amount <= 0:
                continue
            mask_image, cursor, tangent = _stroke_mask(phase, float(amount), size_xy, brush)
            mask = np.asarray(mask_image, dtype=np.float32)[..., None] / 255
            if index < len(phases) - 1:
                opacity = {"pencil": 0.84, "watercolor": 0.70, "marker": 1.0}[brush]
                mask *= opacity
            target = phase.target.astype(np.float32)
            result = np.uint8(np.clip(result.astype(np.float32) * (1 - mask) + target * mask, 0, 255))

        active = next((i for i, amount in enumerate(progresses) if amount < 1), None)
        if t >= 0.94:
            active = len(phases) - 1
        if active is not None:
            _, cursor, _ = _stroke_mask(phases[active], progresses[active], size_xy, brush)
            if t >= 0.94:
                _, cursor, _ = _stroke_mask(phases[-1], 1.0, size_xy, brush)
                vx = cursor[0] - size_xy[0] / 2
                vy = cursor[1] - size_xy[1] / 2
                norm = math.hypot(vx, vy) or 1.0
                exit_amount = min(1.0, (t - 0.94) / 0.06)
                ease = exit_amount * exit_amount * (3 - 2 * exit_amount)
                cursor = (cursor[0] + vx / norm * 360 * ease,
                          cursor[1] + vy / norm * 360 * ease)
            if hand_renderer is not None:
                result = hand_renderer.draw(result, cursor)
            else:
                result = _draw_hand_and_brush(result, cursor, tangent, brush)
        yield result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="把静态人物图变成单向推进的手绘过程 MP4。")
    parser.add_argument("image", type=Path, help="输入图片路径")
    parser.add_argument("output", type=Path, help="输出 MP4 路径")
    parser.add_argument("--duration", type=float, default=8.0, help="视频时长秒数（默认 8）")
    parser.add_argument("--fps", type=int, default=24, help="帧率（默认 24）")
    parser.add_argument("--effect", choices=("sequential", "interleaved"), default="sequential", help="阶段调度：顺序或交错")
    parser.add_argument(
        "--brush", choices=BRUSHES, default="watercolor",
        help="笔触：pencil=细颗粒铅笔，watercolor=柔边半透明水彩，marker=粗实不透明马克笔",
    )
    parser.add_argument("--size", type=int, default=960, help="输出画面最长边像素（默认 960）")
    parser.add_argument(
        "--hand-image", type=Path,
        default=Path(__file__).parent / "assets" / "hand_brush.png",
        help="带透明背景的手和画笔 PNG（默认使用照片素材；传 none 可用程序绘制手）",
    )
    parser.add_argument(
        "--hand-scale", type=float, default=1.0,
        help="实拍手臂缩放比例（默认 1.0；大于 1 会让手臂更明显地延伸出画面）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.image.is_file():
        parser.error(f"找不到输入图片：{args.image}")
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration 必须是大于 0 的有限秒数")
    if args.fps < 1 or args.fps > 60:
        parser.error("--fps 必须在 1 到 60 之间")
    if args.size < 64:
        parser.error("--size 必须至少为 64")
    if not math.isfinite(args.hand_scale) or not 0.5 <= args.hand_scale <= 2.0:
        parser.error("--hand-scale 必须是 0.5 到 2.0 之间的有限数值")
    hand_image = None
    if str(args.hand_image).lower() != "none":
        if not args.hand_image.is_file():
            parser.error(f"找不到手部图片：{args.hand_image}")
        try:
            with Image.open(args.hand_image) as hand_source:
                hand_image = hand_source.convert("RGBA")
        except (OSError, ValueError) as error:
            parser.error(f"无法读取手部图片：{error}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        parser.error("未找到 ffmpeg，请先安装 FFmpeg 并确保它在 PATH 中")

    frame_count = max(2, round(args.duration * args.fps))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(args.image) as source:
            first = _fit_image(source, args.size)
            width, height = first.size
            command = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264",
                "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(args.output),
            ]
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            assert encoder.stdin is not None
            try:
                for frame_number, frame in enumerate(
                    build_frames(
                        source, frame_count, args.effect, args.brush, args.size,
                        hand_image, args.hand_scale,
                    ), start=1
                ):
                    encoder.stdin.write(frame.tobytes())
                    if frame_number == 1 or frame_number == frame_count or frame_number % max(1, args.fps) == 0:
                        print(f"绘制进度：{frame_number}/{frame_count} 帧", file=sys.stderr)
                encoder.stdin.close()
                error = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
                status = encoder.wait()
            except BaseException:
                encoder.kill()
                encoder.wait()
                raise
            if status:
                raise RuntimeError(error.strip() or f"FFmpeg 以状态码 {status} 退出")
    except (OSError, ValueError, RuntimeError) as error:
        print(f"生成失败：{error}", file=sys.stderr)
        return 1

    print(f"已生成：{args.output}（约 {frame_count / args.fps:.2f} 秒，{width}×{height}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
