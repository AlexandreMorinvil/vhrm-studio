"""Background HDF5 writer of the skin-UV pipeline (notebook cell 9, unchanged apart from extra string datasets)."""
import queue
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import torch


class SkinUVH5Writer:
    """Background HDF5 writer: zlib in a thread pool (releases the GIL), write_direct_chunk into gzip-filtered,
    one-frame-per-chunk datasets. GPU outputs arrive through pinned slots, CPU outputs as numpy arrays.
    Small per-frame arrays are kept in memory and written at close."""

    STATUS_CODES = '0=not processed; 1=success; 2=no person detected; 3=invalid crop'

    def __init__(self, path, n_rows, frame_specs, pinned_names, batch_size, attrs, gzip_level=1, n_threads=4,
                 n_slots=3):
        self.gzip_level = gzip_level
        self.n_rows = n_rows
        self.pinned_names = list(pinned_names)
        self.slots = [{name: torch.empty((batch_size,) + frame_specs[name], dtype=torch.uint8, pin_memory=True)
                       for name in self.pinned_names} for _ in range(n_slots)]
        self.free = queue.Queue()
        for slot in range(n_slots):
            self.free.put(slot)
        self.jobs = queue.Queue()
        self.status = np.zeros(n_rows, dtype=np.uint8)
        self.small = {}
        self.error = None
        self.pool = ThreadPoolExecutor(n_threads)
        self.file = h5py.File(path, 'w')
        self.filters = {'compression': 'gzip', 'compression_opts': gzip_level} if gzip_level else {}
        self.datasets = {name: self.file.create_dataset(name, shape=(n_rows,) + shape, dtype=np.uint8,
                                                        chunks=(1,) + shape, **self.filters)
                         for name, shape in frame_specs.items()}
        self.file.attrs.update({**attrs, 'status_codes': self.STATUS_CODES})
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _check(self):
        if self.error is not None:
            raise RuntimeError('HDF5 writer failed') from self.error

    def acquire_slot(self):
        while True:
            self._check()
            try:
                return self.free.get(timeout=0.5)
            except queue.Empty:
                pass

    def submit(self, slot, rows, arrays, small):
        self._check()
        for name, value in small.items():
            value = np.asarray(value)
            if name not in self.small:
                self.small[name] = np.zeros((self.n_rows,) + value.shape[1:], dtype=value.dtype)
            self.small[name][rows] = value
        self.jobs.put(('frames', slot, rows, arrays))

    def set_status(self, rows, code):
        self.jobs.put(('status', rows, code))

    def _encode(self, array):
        return zlib.compress(array, self.gzip_level) if self.gzip_level else array.tobytes()

    def _run(self):
        try:
            while True:
                job = self.jobs.get()
                if job is None:
                    break
                if job[0] == 'status':
                    _, rows, code = job
                    self.status[rows] = code
                    continue
                _, slot, rows, arrays = job
                sources = {name: self.slots[slot][name].numpy() for name in self.pinned_names}
                sources.update(arrays)
                chunks = {name: [self.pool.submit(self._encode, src[i]) for i in range(len(rows))]
                          for name, src in sources.items()}
                for name, futures in chunks.items():
                    dataset = self.datasets[name]
                    offset = (0,) * (dataset.ndim - 1)
                    for row, future in zip(rows, futures):
                        dataset.id.write_direct_chunk((row,) + offset, future.result())
                self.status[rows] = 1
                self.free.put(slot)
        except BaseException as exc:
            self.error = exc

    def close(self, attrs=None, datasets=None, strings=None):
        """strings: {name: list of str} written as variable-length UTF-8 datasets."""
        self.jobs.put(None)
        self.thread.join()
        self.pool.shutdown()
        try:
            self.file.create_dataset('frame_status', data=self.status)
            for name, array in self.small.items():
                self.file.create_dataset(name, data=array, chunks=(min(self.n_rows, 1024),) + array.shape[1:],
                                         **self.filters)
            for name, array in (datasets or {}).items():
                array = np.asarray(array)
                chunks = (min(len(array), 1 << 16),) + array.shape[1:] if array.ndim == 1 else True
                self.file.create_dataset(name, data=array, chunks=chunks, **self.filters)
            for name, values in (strings or {}).items():
                self.file.create_dataset(name, data=np.asarray(values, dtype=object),
                                         dtype=h5py.string_dtype('utf-8'))
            self.file.attrs.update({**(attrs or {}), 'saved_frame_count': self.n_rows,
                                    'successful_frame_count': int((self.status == 1).sum())})
        finally:
            self.file.close()
        self._check()
