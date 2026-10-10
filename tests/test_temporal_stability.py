"""Crop-invariant temporal stabilization and person-bbox robustness, without models or CUDA."""
import copy
import os
import sys
import unittest
from importlib.util import find_spec
from pathlib import Path

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np


class TemporalStabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if find_spec('torch') is None or find_spec('pytorch3d') is None:
            raise unittest.SkipTest('torch and PyTorch3D are optional pipeline dependencies')
        import torch
        from vhrm_studio.pipeline import geometry, temporal
        from vhrm_studio.pipeline.config import TEMPORAL_CONFIG
        cls.torch, cls.geometry, cls.temporal, cls.config = torch, geometry, temporal, TEMPORAL_CONFIG

    def crop(self, cx, cy, size=524.0):
        return self.geometry.gen_trans_from_patch_cv(cx, cy, size, size, 256, 256, 1.0, 0.0, inv=True)

    def frame_origin(self, translation, crop):
        """Frame-pixel projection of the model origin under PEAR's crop camera."""
        ndc = 24.0 * translation[:, :2] / translation[:, 2:]
        crop_px = (1.0 - ndc.numpy()) * 128.0
        return crop_px @ crop[:, :2].T + crop[:, 2]

    def outputs(self, translation):
        torch = self.torch
        n = translation.shape[0]
        camera = torch.eye(4).repeat(n, 1, 1)
        camera[:, :3, 3] = translation
        return {'pd_cam': camera, 'body_param': {}, 'flame_param': {}}

    def test_crop_move_is_not_filtered_as_motion(self):
        torch, T = self.torch, self.temporal.TemporalParameterStabilizer
        stabilizer = T(copy.deepcopy(self.config), 30.0, 4)
        still = torch.tensor([[0.02, -0.15, 7.4]])
        a, b = self.crop(376.0, 250.0), self.crop(376.0, 239.0)
        times = np.arange(8) / 30.0
        stabilizer.stabilize_model_outputs(self.outputs(still.repeat(4, 1)), times[:4],
                                           T.crop_camera(a, 256, 24.0, 'cpu'))
        # The same still person seen through a crop 11 px higher: PEAR's raw translation compensates the move.
        raw_b = T._change_camera(still, T.crop_camera(a, 256, 24.0, 'cpu'), T.crop_camera(b, 256, 24.0, 'cpu'))
        result, _ = stabilizer.stabilize_model_outputs(self.outputs(raw_b.repeat(4, 1)), times[4:],
                                                       T.crop_camera(b, 256, 24.0, 'cpu'))
        np.testing.assert_allclose(self.frame_origin(result['pd_cam'][:, :3, 3], b),
                                   self.frame_origin(still, a).repeat(4, 0), atol=1e-2)

    def test_bbox_deadband_keeps_a_still_box(self):
        bbox_filter = self.temporal.BBoxTemporalFilter(True, 0.0, 8 / 30, deadband=0.03)
        first = bbox_filter.update(np.array([114.0, 0.0, 524.0, 479.0]), 640, 480, 0.0)
        np.testing.assert_array_equal(bbox_filter.update(np.array([114.0, 10.0, 524.0, 470.0]), 640, 480, 0.3), first)
        moved = bbox_filter.update(np.array([60.0, 0.0, 524.0, 479.0]), 640, 480, 0.6)
        self.assertAlmostEqual(float(moved[0]), 60.0, places=3)

    def test_person_bbox_ignores_small_fragments(self):
        ids = np.zeros((512, 512), dtype=np.uint8)
        ids[100:400, 150:350] = 5
        ids[80:84, 240:244] = 5  # false-positive speck close enough to be merged by the dilation
        bbox = self.geometry.bbox_from_class_ids(ids, np.array([[1.0, 0, 0], [0, 1.0, 0]], np.float32), 512, 512, 0.0)
        np.testing.assert_allclose(bbox, [150, 100, 199, 299])


if __name__ == '__main__':
    unittest.main()
