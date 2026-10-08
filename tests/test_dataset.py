"""Dataset discovery, conversion plan, manifest and frame selection (no GPU needed)."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import cv2
import h5py
import numpy as np

from vhrm_studio import convert, dataset
from vhrm_studio.pipeline import frames


def touch(path, data=b''):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def fake_output(path, rows=4):
    """A finished pipeline file as far as the converter checks it."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.create_dataset('frame_status', data=np.ones(rows, np.uint8))
        f.attrs.update({'saved_frame_count': rows, 'effective_fps': 30.0, 'frame_selection_mode': 'all'})


class ImageTimestampTests(unittest.TestCase):
    def test_nanosecond_names(self):  # PURE: Image<ns>.png
        names = [f'Image{1392643993642815000 + round(k * 1e9 / 30)}.png' for k in range(10)]
        ts, source = frames.image_timestamps(names)
        self.assertIn('ns', source)
        np.testing.assert_allclose(ts, np.arange(10) / 30, atol=1e-6)

    def test_millisecond_and_microsecond_names(self):
        ts, source = frames.image_timestamps([f'{1700000000000 + 40 * k}.jpg' for k in range(5)])
        self.assertIn('ms', source)
        np.testing.assert_allclose(ts, np.arange(5) * 0.04)
        ts, source = frames.image_timestamps([f'f_{1000000 + 33333 * k}.png' for k in range(5)])
        self.assertIn('us', source)

    def test_frame_indices_keep_gaps(self):
        ts, source = frames.image_timestamps(['frame_0001.png', 'frame_0002.png', 'frame_0004.png'], 25.0)
        self.assertIn('index', source)
        np.testing.assert_allclose(ts, [0.0, 0.04, 0.12])

    def test_names_without_numbers_use_the_nominal_rate(self):
        ts, source = frames.image_timestamps(['a.png', 'b.png', 'c.png'], 20.0)
        self.assertIn('nominal', source)
        np.testing.assert_allclose(ts, [0.0, 0.05, 0.1])

    def test_natural_order(self):
        self.assertEqual(sorted(['f10.png', 'f2.png', 'f1.png'], key=frames.natural_key),
                         ['f1.png', 'f2.png', 'f10.png'])

    def test_read_image_rgb_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            bgr = np.zeros((4, 6, 3), np.uint8)
            bgr[..., 0] = 200  # blue
            cv2.imwrite(os.path.join(tmp, 'c.png'), bgr)
            cv2.imwrite(os.path.join(tmp, 'g16.png'), np.full((4, 6), 65535, np.uint16))
            self.assertEqual(tuple(frames.read_image_rgb(Path(tmp) / 'c.png')[0, 0]), (0, 0, 200))
            self.assertEqual(tuple(frames.read_image_rgb(Path(tmp) / 'g16.png')[0, 0]), (255, 255, 255))


class FrameSelectionTests(unittest.TestCase):
    def test_rules(self):
        cfr = lambda fps: np.arange(int(10 * fps)) / fps  # noqa: E731
        self.assertEqual(frames.select_frames(cfr(29.26), 29.26)['mode'], 'all')
        sel = frames.select_frames(cfr(60), 60)
        self.assertEqual((sel['mode'], sel['stride']), ('stride', 2))
        sel = frames.select_frames(cfr(50), 50)
        self.assertEqual(sel['mode'], 'timestamp')
        self.assertTrue((np.diff(sel['indices']) > 0).all())
        self.assertAlmostEqual(sel['effective_fps'], 30.0, delta=0.5)

    def test_select_filter(self):
        self.assertIsNone(frames.select_filter([0, 1, 2]))
        self.assertEqual(frames.select_filter([0, 2, 4]), 'select=not(mod(n\\,2))')
        self.assertIn('between(n,3,5)', frames.select_filter([0, 3, 4, 5]))


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = (Path(self.tmp.name) / 'data').resolve()
        touch(self.root / 'subject1' / 'vid.avi', b'video1')
        touch(self.root / 'subject1' / 'ground_truth.txt', b'1 2 3')
        touch(self.root / 'subject2' / 'vid.avi', b'video2')
        touch(self.root / 'subject2' / 'vid_skin_uv.h5')            # an earlier output: not a source
        touch(self.root / 'subject2' / 'thumb.png')                   # too few images: a companion file
        for k in range(35):                                           # PURE-like image sequence
            touch(self.root / '01-01' / '01-01' / f'Image{1000000000 + k * 33333333}.png')
        touch(self.root / '01-01' / '01-01.json', b'{}')
        touch(self.root / 'clash' / 'a.avi')
        touch(self.root / 'clash' / 'a.mp4')
        touch(self.root / 'readme.txt', b'readme')
        touch(self.root / '.hidden' / 'x.avi')

    def test_discover(self):
        items = dataset.discover(self.root)
        self.assertEqual([(i.kind, i.rel) for i in items],
                         [('images', '01-01/01-01'), ('video', 'clash/a.avi'), ('video', 'clash/a.mp4'),
                          ('video', 'subject1/vid.avi'), ('video', 'subject2/vid.avi')])
        self.assertEqual(len(items[0].files), 35)
        self.assertEqual(len(dataset.discover(self.root, min_images=100)), 4)

    def test_plan_in_place(self):
        items = dataset.discover(self.root)
        out = dataset.plan_outputs(items, self.root, 'in_place')
        self.assertEqual(out, self.root.resolve())
        self.assertEqual([i.output_rel for i in items],
                         ['01-01/01-01_skin_uv.h5', 'clash/a_skin_uv.h5', 'clash/a_mp4_skin_uv.h5',
                          'subject1/vid_skin_uv.h5', 'subject2/vid_skin_uv.h5'])
        self.assertEqual(items[3].output, self.root.resolve() / 'subject1' / 'vid_skin_uv.h5')

    def test_plan_mirror_and_companions(self):
        target = Path(self.tmp.name) / 'mirror'
        items = dataset.discover(self.root)
        out = dataset.plan_outputs(items, self.root, 'mirror', target)
        self.assertEqual(items[3].output, target.resolve() / 'subject1' / 'vid_skin_uv.h5')
        companions = {dataset.rel_posix(p, self.root) for p in dataset.companion_files(self.root, items)}
        self.assertEqual(companions, {'subject1/ground_truth.txt', 'subject2/thumb.png', '01-01/01-01.json',
                                      'readme.txt'})
        with self.assertRaises(ValueError):
            dataset.plan_outputs(items, self.root, 'mirror', self.root)
        self.assertEqual(out, target.resolve())

    def test_output_inside_the_source_is_excluded(self):
        inside = self.root / 'converted'
        touch(inside / 'subject1' / 'vid.avi')  # a copied source
        items = dataset.discover(self.root, exclude=inside)
        self.assertNotIn('converted/subject1/vid.avi', [i.rel for i in items])

    def test_manifest_round_trip_and_lookup(self):
        target = Path(self.tmp.name) / 'mirror'
        items = dataset.discover(self.root)
        dataset.plan_outputs(items, self.root, 'mirror', target)
        manifest = dataset.new_manifest(self.root, target, 'mirror')
        for item in items:
            dataset.manifest_item(manifest, item.rel).update(output=item.output_rel, kind=item.kind, status='done')
        dataset.save_manifest(manifest)
        loaded = dataset.load_manifest(target)
        self.assertEqual(len(loaded['items']), 5)
        self.assertTrue((target / dataset.MANIFEST_CSV).is_file())
        h5 = target / 'subject1' / 'vid_skin_uv.h5'
        touch(h5)
        self.assertEqual(dataset.source_from_manifest(h5), self.root.resolve() / 'subject1' / 'vid.avi')
        self.assertEqual(dataset.source_from_manifest(target / 'clash' / 'a_mp4_skin_uv.h5'),
                         self.root.resolve() / 'clash' / 'a.mp4')


class RunJobTests(unittest.TestCase):
    """The converter without the GPU: every output exists, so nothing is loaded."""

    def test_resume_copies_companions_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, target = (Path(tmp) / 'src').resolve(), (Path(tmp) / 'dst').resolve()
            touch(root / 's1' / 'vid.avi', b'v')
            touch(root / 's1' / 'gt.txt', b'signal')
            items = dataset.discover(root)
            dataset.plan_outputs(items, root, 'mirror', target)
            fake_output(items[0].output, rows=6)
            touch(str(items[0].output) + dataset.PARTIAL_SUFFIX)  # left by an interrupted run
            events = []
            job = {'source_root': str(root), 'output_root': str(target), 'mode': 'mirror',
                   'items': [i.to_job() for i in items], 'copy_companions': True, 'copy_sources': False,
                   'overwrite': False, 'assets_dir': '', 'settings': {}}
            self.assertEqual(convert.run_job(job, events.append), 'completed')
            self.assertEqual((target / 's1' / 'gt.txt').read_bytes(), b'signal')
            self.assertFalse((target / 's1' / 'vid.avi').exists())
            self.assertFalse(Path(str(items[0].output) + dataset.PARTIAL_SUFFIX).exists())
            done = [e for e in events if e['event'] == 'item_done']
            self.assertEqual(done[0]['status'], 'skipped')
            entry = dataset.load_manifest(target)['items'][0]
            self.assertEqual((entry['source'], entry['output'], entry['status'], entry['rows']),
                             ('s1/vid.avi', 's1/vid_skin_uv.h5', 'done', 6))
            self.assertEqual(events[-1]['event'], 'finished')
            self.assertTrue((target / convert.LOG_NAME).is_file())
            self.assertFalse((target / convert.LOCK_NAME).exists())

    def test_one_conversion_per_output_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / convert.LOCK_NAME).write_text(f'{os.getppid()}\n')  # a running process owns the folder
            events = []
            job = {'source_root': str(root), 'output_root': str(root), 'mode': 'in_place', 'items': []}
            self.assertEqual(convert.run_job(job, events.append), 'error')
            self.assertIn('another conversion', next(e['text'] for e in events if e['event'] == 'error'))
            (root / convert.LOCK_NAME).write_text('999999999\n')  # a stale lock is taken over
            self.assertEqual(convert.run_job(job, [].append), 'completed')
            self.assertFalse((root / convert.LOCK_NAME).exists())


if __name__ == '__main__':
    unittest.main()
