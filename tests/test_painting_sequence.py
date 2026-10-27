import unittest
import math

import numpy as np
from PIL import Image, ImageDraw

from painting_video import (
    BrushStroke,
    PaintPhase,
    _coverage_strokes,
    _fit_image,
    _hand_exit_pose,
    _make_phases,
    _prepare_photo_hand,
    _stroke_mask,
    build_frames,
)


class PaintingSequenceTests(unittest.TestCase):
    def test_hand_exit_orientation_uses_frame_axis_and_pen_tip_half(self):
        with self.subTest(orientation="portrait"):
            angle, side = _hand_exit_pose((536, 960), (100, 300), None)
            self.assertEqual(side, -1)
            self.assertAlmostEqual(angle, math.pi)
            angle, side = _hand_exit_pose((536, 960), (400, 300), None)
            self.assertEqual(side, 1)
            self.assertAlmostEqual(angle, 0.0)

        with self.subTest(orientation="landscape"):
            angle, side = _hand_exit_pose((960, 536), (300, 100), None)
            self.assertEqual(side, -1)
            self.assertAlmostEqual(angle, -math.pi / 2)
            angle, side = _hand_exit_pose((960, 536), (300, 400), None)
            self.assertEqual(side, 1)
            self.assertAlmostEqual(angle, math.pi / 2)

    def test_hand_exit_orientation_has_midline_hysteresis(self):
        width, height = (536, 960)
        center = width / 2
        deadband = width * 0.04

        angle, side = _hand_exit_pose((width, height), (center + deadband * 0.5, 200), -1)
        self.assertEqual(side, -1)
        self.assertAlmostEqual(angle, math.pi)

        angle, side = _hand_exit_pose((width, height), (center + deadband * 1.1, 200), -1)
        self.assertEqual(side, 1)
        self.assertAlmostEqual(angle, 0.0)

    def test_hand_scale_keeps_the_arm_exit_past_the_selected_frame_edge(self):
        hand = Image.new("RGBA", (1043, 1504), (160, 110, 75, 255))

        for frame_size, axis_length in [((536, 960), 536), ((960, 536), 536)]:
            with self.subTest(frame_size=frame_size):
                renderer = _prepare_photo_hand(hand, frame_size, 0.5)
                required_distance = axis_length / 2 + max(8, axis_length * 0.04)
                self.assertGreaterEqual(renderer.exit_distance, required_distance)

    def test_artwork_stages_progress_from_composition_to_finish(self):
        image = Image.new("RGB", (240, 136), (70, 80, 100))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 239, 42), fill=(90, 45, 35))
        draw.ellipse((46, 18, 145, 130), fill=(180, 85, 48))
        draw.rectangle((166, 18, 228, 120), fill=(215, 190, 145))
        draw.line((177, 34, 219, 34), fill=(40, 30, 25), width=3)

        phases = _make_phases(image, "watercolor")
        stages = list(dict.fromkeys(phase.stage for phase in phases))

        self.assertEqual(stages, [
            "构图与明暗",
            "背景大色块",
            "主体铺色",
            "阴影与受光",
            "线条与最终细节",
        ])
        self.assertTrue(all(phase.weight > 0 for phase in phases))
        self.assertTrue(np.array_equal(phases[-1].target, np.asarray(image)))

    def test_final_color_pass_is_complete_for_every_brush(self):
        size = (128, 128)
        phase = PaintPhase(
            "全彩图层收尾",
            np.zeros((size[1], size[0], 3), dtype=np.uint8),
            _coverage_strokes(*size, 24, 101, True),
            1.0,
            stage="线条与最终细节",
        )

        for brush in ("pencil", "watercolor", "marker"):
            with self.subTest(brush=brush):
                mask = np.asarray(_stroke_mask(phase, 1.0, size, brush)[0])
                self.assertTrue(np.all(mask == 255))

    def test_last_frame_restores_the_complete_source_for_every_brush(self):
        image = Image.new("RGB", (64, 48), (85, 40, 25))
        ImageDraw.Draw(image).ellipse((14, 4, 50, 45), fill=(225, 156, 52))
        expected = np.asarray(_fit_image(image, 64))
        transparent_hand = Image.new("RGBA", (1, 1), (0, 0, 0, 0))

        for brush in ("pencil", "watercolor", "marker"):
            with self.subTest(brush=brush):
                frames = list(build_frames(image, 2, brush=brush, size=64, hand_image=transparent_hand))
                self.assertTrue(np.array_equal(frames[-1], expected))

    def test_hand_scale_changes_forearm_sprite_size(self):
        hand = Image.new("RGBA", (100, 200), (160, 110, 75, 255))

        regular = _prepare_photo_hand(hand, (960, 536), 1.0)
        enlarged = _prepare_photo_hand(hand, (960, 536), 1.25)

        self.assertEqual(regular.sprite.height, 493)
        self.assertEqual(enlarged.sprite.height, 616)

    def test_pencil_color_and_tone_passes_do_not_break_into_scanline_gaps(self):
        size = (128, 128)
        for stage, name in (("背景大色块", "背景大色块1"), ("阴影与受光", "暗部塑形")):
            with self.subTest(stage=stage):
                phase = PaintPhase(
                    name,
                    np.zeros((size[1], size[0], 3), dtype=np.uint8),
                    _coverage_strokes(*size, 24, 40, False),
                    1.0,
                    stage=stage,
                )
                mask = np.asarray(_stroke_mask(phase, 1.0, size, "pencil")[0])
                self.assertGreater(float(mask.mean()), 245.0)

    def test_brush_styles_have_visibly_distinct_stroke_masks(self):
        size = (128, 128)
        phase = PaintPhase(
            "样式检查",
            np.zeros((size[1], size[0], 3), dtype=np.uint8),
            [
                # A full-width straight pass makes width, grain and edge
                # softness differences measurable and deterministic.
                BrushStroke([(4, 64), (124, 64)], 16),
            ],
            1.0,
        )
        masks = {
            brush: np.asarray(_stroke_mask(phase, 1.0, size, brush)[0])
            for brush in ("pencil", "watercolor", "marker")
        }

        self.assertGreater(int(masks["marker"].sum()), int(masks["pencil"].sum()))
        self.assertGreater(np.count_nonzero((masks["watercolor"] > 0) & (masks["watercolor"] < 255)), 0)
        self.assertFalse(np.array_equal(masks["marker"], masks["watercolor"]))


if __name__ == "__main__":
    unittest.main()
