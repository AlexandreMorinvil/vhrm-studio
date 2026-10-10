import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from vhrm_studio import app
from vhrm_studio.mesh import SmplxMesh
import h5py
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


class VideoDependencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_import_failure_is_preserved_and_reported_at_startup(self):
        for exception in ("ModuleNotFoundError('No module named av', name='av')",
                          "ImportError('DLL load failed: Application Control policy blocked pyio')"):
            code = f"""
import builtins
original_import = builtins.__import__
error = {exception}
def guarded_import(name, *args, **kwargs):
    if name == 'av':
        raise error
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from vhrm_studio import app
assert app.av is None
assert app._AV_IMPORT_ERROR is error
assert str(error) in app._pyav_error_message()
"""
            with self.subTest(exception=exception):
                result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                                        env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1]
                                                                          / 'src')})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('RuntimeWarning:', result.stderr)
                self.assertIn('Restart the viewer', result.stderr)

    def test_missing_package_identifies_the_interpreter_and_install_command(self):
        error = ModuleNotFoundError("No module named 'av'", name='av')
        with mock.patch.object(app, 'av', None), mock.patch.object(app, '_AV_IMPORT_ERROR', error):
            message = app._pyav_error_message()
            self.assertIn('PyAV is not installed', message)
            self.assertIn(sys.executable, message)
            self.assertIn('-m pip install av', message)
            with self.assertRaises(RuntimeError) as raised:
                app.VideoSource(Path('source.avi'), {})
            self.assertIs(raised.exception.__cause__, error)

    def test_native_import_failure_is_not_reported_as_a_missing_package(self):
        error = ImportError('DLL load failed: Application Control policy blocked pyio')
        with mock.patch.object(app, 'av', None), mock.patch.object(app, '_AV_IMPORT_ERROR', error):
            message = app._pyav_error_message()
        self.assertIn('PyAV could not be imported', message)
        self.assertNotIn('PyAV is not installed', message)
        self.assertIn(str(error), message)
        self.assertIn('administrator-approved package', message)
        self.assertIn(sys.executable, message)

    def test_source_video_session_reports_the_original_import_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'source_skin_uv.h5'
            video = root / 'source.avi'
            video.touch()

            def write():
                with h5py.File(path, 'w') as file:
                    file.attrs['texture_size'] = 8
                    file.create_dataset('frame_status', data=np.array([1], np.uint8))

            app.h5call(write)
            error = ImportError('DLL load failed: signing policy blocked PyAV')
            with mock.patch.object(app, 'av', None), mock.patch.object(app, '_AV_IMPORT_ERROR', error):
                session = app.Session(path, SimpleNamespace(), video_override=video)
                try:
                    self.assertIsNone(session.video)
                    self.assertIn(str(error), session.video_error)
                    self.assertIn('PyAV could not be imported', session.video_error)
                finally:
                    session.close()

    def test_mp4_export_warns_before_using_the_existing_opencv_fallback(self):
        error = ImportError('DLL load failed while importing pyio')
        with mock.patch.object(app, 'av', None), mock.patch.object(app, '_AV_IMPORT_ERROR', error), \
                mock.patch.object(app.cv2, 'VideoWriter') as video_writer, \
                self.assertWarnsRegex(RuntimeWarning, 'OpenCV mp4v fallback'):
            writer = app.Mp4Writer(Path('output.mp4'), 20.0, 32, 32)
            self.assertIs(writer.cv, video_writer.return_value)
            self.assertEqual(writer.codec, 'opencv mp4v')
            writer.close()
        video_writer.return_value.release.assert_called_once()


class PackedUVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / 'packed_skin_uv.h5'
        self.rgb = np.zeros((4, 16, 3), np.uint8)
        self.rgb[1, 2] = [220, 100, 50]
        self.body = np.zeros((4, 16), bool)
        self.body[1:3, 1:5] = True
        self.opts = {'alpha': .5, 'silhouette': True, 'age_window_s': 10}

    def open_file(self, packed=True, body=True, mask=False):
        def write():
            with h5py.File(self.path, 'w') as file:
                file.attrs['texture_size'] = 8
                file.create_dataset('frame_status', data=np.array([1, 2], np.uint8))
                file.create_dataset('uv_body_mask', data=np.ones((8, 8), np.uint8))
                if packed:
                    file.create_dataset('packed_uv_skin_rgb', data=np.stack([self.rgb, self.rgb]),
                                        chunks=(1, 4, 16, 3), compression='gzip')
                    if body:
                        file.create_dataset('packed_uv_body_mask', data=self.body)
                    if mask:
                        skin = self.rgb.any(-1)
                        skin[1, 3] = True
                        file.create_dataset('packed_uv_skin_mask_packed',
                                            data=np.packbits(np.stack([skin, skin]), axis=-1))

        app.h5call(write)
        src = app.SkinUVFile(self.path)
        self.addCleanup(src.close)
        session = SimpleNamespace(src=src, raw_future=self.raw_future)
        self.src = src
        return app.RowContext(session, 0)

    def raw_future(self, row, item):
        future = app.Future()
        future.set_result(self.src.read(item, row))
        return future

    def test_rgb_only_packed_atlas_and_silhouette(self):
        ctx = self.open_file()
        self.assertFalse(ctx.src.has('packed_uv_skin_mask'))
        image = app.render_view('packed_uv_skin_rgb', ctx, self.opts).image
        self.assertEqual(image.shape, (4, 16, 3))
        np.testing.assert_array_equal(image[1, 2], self.rgb[1, 2])
        np.testing.assert_array_equal(image[1, 3], [app.UV_EMPTY_LEVEL] * 3)
        np.testing.assert_array_equal(image[0, 0], [0, 0, 0])
        self.opts['silhouette'] = False
        np.testing.assert_array_equal(app.render_view('packed_uv_skin_rgb', ctx, self.opts).image, self.rgb)

    def test_optional_mask_preserves_observed_black_texels(self):
        ctx = self.open_file(mask=True)
        image = app.render_view('packed_uv_skin_rgb', ctx, self.opts).image
        np.testing.assert_array_equal(image[1, 3], [0, 0, 0])
        np.testing.assert_array_equal(image[2, 3], [app.UV_EMPTY_LEVEL] * 3)

    def test_rgb_without_body_mask(self):
        ctx = self.open_file(body=False)
        np.testing.assert_array_equal(app.render_view('packed_uv_skin_rgb', ctx, self.opts).image, self.rgb)

    def test_legacy_files_report_unavailable(self):
        ctx = self.open_file(packed=False)
        with self.assertRaisesRegex(app.Unavailable, 'packed_uv_skin_rgb is not in this file'):
            app.render_view('packed_uv_skin_rgb', ctx, self.opts)

    def test_failed_rows_report_unavailable(self):
        ctx = self.open_file()
        ctx.row = 1
        with self.assertRaisesRegex(app.Unavailable, 'row 1: no person detected'):
            app.render_view('packed_uv_skin_rgb', ctx, self.opts)

    def test_panel_selector_and_session_loading(self):
        ctx = self.open_file()
        session = app.Session.__new__(app.Session)
        session.src, session.views, session.video = ctx.src, ['packed_uv_skin_rgb'], None
        session.mesh = SimpleNamespace(state='disabled')
        self.assertEqual(session._items(), {'packed_uv_skin_rgb'})
        panel = app.ImagePanel('packed_uv_skin_rgb')
        try:
            self.assertEqual(panel.key, 'packed_uv_skin_rgb')
            panel.show_result(app.render_view(panel.key, ctx, self.opts))
            np.testing.assert_array_equal(panel._image[1, 2], self.rgb[1, 2])
            self.assertEqual(panel.combo.currentText(), 'Packed UV: skin RGB')
        finally:
            panel.close()


class StandardOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        app.MESH_PAN[:] = [0.0, 0.0]
        app.MESH_ORBIT[:] = [0.0, 0.0, 1.0]
        size = 8
        self.mask = np.zeros((size, size), bool)
        self.mask[2, 3] = True
        self.visible = self.mask.copy()
        self.visible[4, 5] = True
        self.rgb = np.full((size, size, 3), [220, 100, 50], np.uint8)
        ids = np.full((size, size), 13, np.uint8)
        colormap = np.zeros((256, 1, 3), np.uint8)
        colormap[13] = [252, 229, 92]
        self.src = SimpleNamespace(
            uv_bg=np.zeros_like(self.rgb), uv_body=np.ones((size, size), bool),
            uv_body_u8=np.ones((size, size), np.uint8), crop_size=size, fps=30,
            attrs={'pear_crop_size': size, 'render_size': size}, colormap=colormap,
            class_names=app.DEFAULT_CLASS_NAMES,
            small={name: np.array([[[1., 0., 0.], [0., 1., 0.]]])
                   for name in ('crop_to_frame', 'pear_crop_to_frame')},
            texel_tables=lambda: (np.arange(64), np.tile([1., 0., 0.], (64, 1))),
            accumulated_final=lambda: (self.rgb, self.mask))
        yy, xx = np.indices((size, size))
        mesh = {'screen': np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32),
                'faces': np.repeat(np.arange(64)[:, None], 3, axis=1)}
        self.data = {'uv_skin_rgb': self.rgb, 'uv_skin_mask': self.mask, 'uv_visible_mask': self.visible,
                     'uv_class_ids': ids, 'crop_skin_rgb': self.rgb, 'crop_skin_mask': self.mask,
                     'crop_class_ids': ids, 'mesh': mesh, 'standard_mesh': np.zeros((3, 3)),
                     'accum': (self.rgb, self.mask, np.zeros((size, size), np.int64))}
        self.renderer = SimpleNamespace(render_orbit=self.render_orbit)
        self.ctx = SimpleNamespace(src=self.src, row=0, get=self.data.__getitem__,
                                   session=SimpleNamespace(mesh=SimpleNamespace(model=self.renderer)))
        self.opts = {'alpha': .5, 'silhouette': True, 'age_window_s': 10}

    def render_orbit(self, verts, *orbit, **kwargs):
        self.texture = kwargs.get('texture')
        self.pan = kwargs.get('pan')
        return self.rgb.copy()

    def test_pan_gestures_preserve_rotation_and_reset(self):
        panel = app.ImagePanel('standard_uv_skin_rgb')
        panel.show_result(app.render_view(panel.key, self.ctx, self.opts))
        panel.resize(480, 640)
        panel.show()
        self.qt.processEvents()
        try:
            for button, modifiers, toggle in (
                    (QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier, True),
                    (QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.ShiftModifier, False),
                    (QtCore.Qt.MouseButton.MiddleButton, QtCore.Qt.KeyboardModifier.NoModifier, False),
                    (QtCore.Qt.MouseButton.RightButton, QtCore.Qt.KeyboardModifier.NoModifier, False)):
                panel.fit()
                panel.pan_button.setChecked(toggle)
                for kind, position in ((QtCore.QEvent.Type.MouseButtonPress, QtCore.QPointF(100, 100)),
                                       (QtCore.QEvent.Type.MouseMove, QtCore.QPointF(130, 160)),
                                       (QtCore.QEvent.Type.MouseButtonRelease, QtCore.QPointF(130, 160))):
                    event = QtGui.QMouseEvent(kind, position, position, button, button, modifiers)
                    self.assertTrue(panel.eventFilter(panel.gw.viewport(), event))
                self.assertEqual(app.MESH_ORBIT, [0.0, 0.0, 1.0])
                self.assertGreater(app.MESH_PAN[0], 0)
                self.assertGreater(app.MESH_PAN[1], 0)
                self.assertEqual(self.pan, tuple(app.MESH_PAN))
                panel.show_result(app.render_view(panel.key, self.ctx, self.opts))
                self.assertEqual(self.pan, tuple(app.MESH_PAN))
            panel.fit()
            self.assertEqual(app.MESH_PAN, [0.0, 0.0])
        finally:
            panel.close()
            app.MESH_PAN[:] = [0.0, 0.0]

    def test_all_views_keep_textures_during_orbit(self):
        for view in app.VIEWS:
            if not view.key.startswith('standard'):
                continue
            with self.subTest(view=view.key):
                result = app.render_view(view.key, self.ctx, self.opts)
                self.assertIsInstance(result, app.MeshResult)
                panel = app.ImagePanel(view.key)
                try:
                    panel.show_result(result)
                    panel._render_mesh()
                    self.assertEqual(panel.space, '3d')
                    self.assertIs(self.texture, result.texture)
                    self.assertLess(panel.minimumSizeHint().width(), 480)
                    self.assertEqual(panel.combo.toolTip(), view.label)
                finally:
                    panel.close()

    def test_skin_and_classes_respect_visibility(self):
        for key in ('standard_crop_skin_rgb', 'standard_uv_skin_rgb'):
            texture = app.standard_texture(key, self.ctx, self.opts)
            np.testing.assert_array_equal(texture[2, 3], self.rgb[2, 3])
            np.testing.assert_array_equal(texture[4, 5], [110, 110, 110])
        classes = app.standard_texture('standard_crop_classes', self.ctx, self.opts)
        np.testing.assert_array_equal(classes[4, 5], [252, 229, 92])
        np.testing.assert_array_equal(classes[0, 0], [110, 110, 110])

    def test_crop_affine_and_out_of_crop_texels(self):
        self.src.small['crop_to_frame'][0, 0, 2] = 1
        texture = app.standard_texture('standard_crop_skin_rgb', self.ctx, self.opts)
        np.testing.assert_array_equal(texture[2, 3], [110, 110, 110])
        self.visible[2, 4] = True
        texture = app.standard_texture('standard_crop_skin_rgb', self.ctx, self.opts)
        np.testing.assert_array_equal(texture[2, 4], self.rgb[2, 3])
        self.src.small['crop_to_frame'][0, 0, 2] = 100
        texture = app.standard_texture('standard_crop_classes', self.ctx, self.opts)
        self.assertTrue(np.all(texture == 110))

    def test_crop_mask_is_distinct_from_eroded_uv_mask(self):
        self.data['uv_skin_mask'] = np.zeros_like(self.mask)
        crop = app.standard_texture('standard_crop_skin_mask', self.ctx, self.opts)
        atlas = app.standard_texture('standard_uv_skin_mask', self.ctx, self.opts)
        np.testing.assert_array_equal(crop[2, 3], [60, 210, 90])
        np.testing.assert_array_equal(atlas[2, 3], [110, 110, 110])


class StandardGeometryTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch is an optional mesh dependency')
        self.torch = torch
        self.model = SmplxMesh.__new__(SmplxMesh)
        self.model.torch, self.model.device, self.model.lock = torch, torch.device('cpu'), threading.Lock()

    def test_neutral_pose_preserves_identity_and_inputs(self):
        torch = self.torch
        captured = {}

        def ehm(body, flame, **kwargs):
            captured.update(body=body, flame=flame)
            return {'vertices': torch.tensor([[[1., 2., 3.]]])}

        self.model.ehm = ehm
        body = {key: np.ones((count, 3, 3), np.float32)
                for key, count in (('global_pose', 1), ('body_pose', 21), ('left_hand_pose', 15),
                                   ('right_hand_pose', 15))}
        body.update(exp=np.ones(50), shape=np.array([2.]), head_scale=np.array([1.2]), hand_scale=np.array([.9]))
        flame = {'jaw_params': np.ones(3), 'shape_params': np.array([4.]), 'expression_params': np.ones(50)}
        np.testing.assert_array_equal(self.model.compute_standard(body, flame), [[-1., 2., -3.]])
        for key in ('global_pose', 'body_pose', 'left_hand_pose', 'right_hand_pose'):
            expected = torch.eye(3).expand_as(captured['body'][key])
            self.assertTrue(torch.equal(captured['body'][key], expected))
            self.assertTrue(np.all(body[key] == 1))
        for key in ('shape', 'head_scale', 'hand_scale'):
            np.testing.assert_allclose(captured['body'][key][0].numpy(), body[key])
        self.assertEqual(captured['flame']['shape_params'][0, 0], 4)
        self.assertEqual(torch.count_nonzero(captured['body']['exp']), 0)
        self.assertEqual(torch.count_nonzero(captured['flame']['jaw_params']), 0)

    def test_textured_raster_rotation_and_non_uv_faces(self):
        try:
            from pytorch3d.structures import Meshes
            from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
        except ImportError:
            self.skipTest('PyTorch3D is an optional mesh dependency')
        torch, model = self.torch, self.model
        if not torch.cuda.is_available():
            self.skipTest('The orbit renderer uses CUDA binned rasterization')
        model.device = torch.device('cuda:0')
        model.faces = torch.tensor([[0, 1, 2], [3, 4, 5]], device=model.device)
        model.face_uv = torch.tensor([[[0., 0.], [1., 0.], [0., 1.]]], device=model.device)
        model.uv_face_count = 1
        model._raster = (rasterize_meshes, Meshes)
        verts = np.array([[-1., -1., 0.], [0., -1., 0.], [-1., 1., 0.],
                          [0., -1., 0.], [1., -1., 0.], [1., 1., 0.]], np.float32)
        texture = np.full((32, 32, 3), [255, 0, 0], np.uint8)
        image = model.render_orbit(verts, size=128, texture=texture)
        self.assertGreater(int((image[..., 0] > 100).sum()), 100)
        gray = (image[..., 0] == image[..., 1]) & (image[..., 1] == image[..., 2]) & (image[..., 0] > 30)
        self.assertGreater(int(gray.sum()), 100)
        rotated = model.render_orbit(verts, yaw_deg=45, size=128, texture=texture)
        self.assertFalse(np.array_equal(image, rotated))
        translated = model.render_orbit(verts, size=128, texture=texture, pan=(0.1, 0.1))
        red = lambda frame: np.nonzero((frame[..., 0] > 100) & (frame[..., 1] < 20))
        before_y, before_x = red(image)
        after_y, after_x = red(translated)
        self.assertAlmostEqual(after_x.mean() - before_x.mean(), 12.8, delta=1)
        self.assertAlmostEqual(after_y.mean() - before_y.mean(), 12.8, delta=1)


if __name__ == '__main__':
    unittest.main()