"""Person detection and batch initialization without loading models or requiring CUDA."""
import os
import sys
import unittest
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np


class PersonDetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if find_spec('torch') is None or find_spec('pytorch3d') is None:
            raise unittest.SkipTest('torch and PyTorch3D are optional pipeline dependencies')
        import torch
        from vhrm_studio.pipeline import runner
        cls.torch = torch
        cls.runner = runner

    def setUp(self):
        self.pipeline = self.runner.SkinUVPipeline.__new__(self.runner.SkinUVPipeline)
        self.pipeline.device = self.torch.device('cpu')
        self.pipeline.detector = mock.Mock()

    def detector_result(self, boxes):
        xyxy = self.torch.tensor(boxes, dtype=self.torch.float32).reshape(-1, 4)
        return [SimpleNamespace(boxes=SimpleNamespace(xyxy=xyxy))]

    def batch(self, n=8):
        return self.torch.arange(n, dtype=self.torch.float32)[:, None, None, None].expand(n, 3, 4, 5).clone()

    def test_detector_converts_rgb_to_bgr_without_mutating_input(self):
        rgb = np.full((4, 5, 3), [10, 20, 30], dtype=np.uint8)
        original = rgb.copy()
        self.pipeline.detector.predict.return_value = self.detector_result(
            [[1, 2, 4, 4], [0, 0, 5, 4]])

        box = self.pipeline.detect_person_bbox(rgb)

        np.testing.assert_array_equal(box, [0, 0, 5, 4])
        image = self.pipeline.detector.predict.call_args.args[0]
        np.testing.assert_array_equal(image, rgb[..., ::-1])
        np.testing.assert_array_equal(rgb, original)
        self.assertTrue(image.flags.c_contiguous)
        self.assertEqual(box.dtype, np.float32)
        self.assertEqual(self.pipeline.detector.predict.call_args.kwargs,
                         {'device': 'cpu', 'classes': 0, 'conf': self.runner.YOLO_CONF,
                          'save': False, 'verbose': False})

    def test_detector_returns_none_when_no_person_meets_threshold(self):
        self.pipeline.detector.predict.return_value = self.detector_result([])
        self.assertIsNone(self.pipeline.detect_person_bbox(np.zeros((4, 5, 3), np.uint8)))

    def test_latest_batch_frame_seeds_all_original_frames(self):
        frames = self.batch()
        original = frames.clone()
        expected = np.array([0, 0, 5, 4], np.float32)
        with mock.patch.object(self.pipeline, 'detect_person_bbox', return_value=expected) as detect:
            box, index, calls = self.pipeline._detect_batch_person_bbox(frames)

        np.testing.assert_array_equal(box, expected)
        self.assertEqual((index, calls), (7, 1))
        detect.assert_called_once()
        image = detect.call_args.args[0]
        np.testing.assert_array_equal(image, np.full((4, 5, 3), 7, np.uint8))
        self.assertEqual(image.dtype, np.uint8)
        self.assertTrue(self.torch.equal(frames, original))

    def test_detection_retries_earlier_frames_in_same_batch(self):
        expected = np.array([0, 0, 5, 4], np.float32)
        with mock.patch.object(self.pipeline, 'detect_person_bbox',
                               side_effect=[None, None, expected]) as detect:
            box, index, calls = self.pipeline._detect_batch_person_bbox(self.batch())

        np.testing.assert_array_equal(box, expected)
        self.assertEqual((index, calls), (5, 3))
        self.assertEqual([int(call.args[0][0, 0, 0]) for call in detect.call_args_list], [7, 6, 5])

    def test_first_frame_can_still_seed_batch_when_later_frames_fail(self):
        expected = np.array([0, 0, 5, 4], np.float32)
        with mock.patch.object(self.pipeline, 'detect_person_bbox',
                               side_effect=[None] * 7 + [expected]):
            box, index, calls = self.pipeline._detect_batch_person_bbox(self.batch())

        np.testing.assert_array_equal(box, expected)
        self.assertEqual((index, calls), (0, 8))

    def test_no_person_checks_every_frame_before_skipping_batch(self):
        with mock.patch.object(self.pipeline, 'detect_person_bbox', return_value=None) as detect:
            box, index, calls = self.pipeline._detect_batch_person_bbox(self.batch())

        self.assertIsNone(box)
        self.assertIsNone(index)
        self.assertEqual(calls, 8)
        self.assertEqual([int(call.args[0][0, 0, 0]) for call in detect.call_args_list], list(range(7, -1, -1)))

    def test_single_frame_batch(self):
        expected = np.array([0, 0, 5, 4], np.float32)
        with mock.patch.object(self.pipeline, 'detect_person_bbox', return_value=expected):
            box, index, calls = self.pipeline._detect_batch_person_bbox(self.batch(1))

        np.testing.assert_array_equal(box, expected)
        self.assertEqual((index, calls), (0, 1))

    def test_empty_batch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'empty batch'):
            self.pipeline._detect_batch_person_bbox(self.batch(0))

    def test_detector_errors_propagate(self):
        with mock.patch.object(self.pipeline, 'detect_person_bbox', side_effect=RuntimeError('detector failed')):
            with self.assertRaisesRegex(RuntimeError, 'detector failed'):
                self.pipeline._detect_batch_person_bbox(self.batch())


if __name__ == '__main__':
    unittest.main()
