"""Dataset conversion: runs the skin-UV pipeline on every source of a plan.

    vhrm-studio-convert SOURCE_ROOT                         # HDF5 files next to the sources
    vhrm-studio-convert SOURCE_ROOT --out NEW_ROOT          # mirrored dataset (other files copied, not the videos)
    vhrm-studio-convert SOURCE_ROOT --out NEW_ROOT --copy-sources --overwrite --max-frames 240
    python -m vhrm_studio.convert --job job.json            # used by the GUI: JSON-line events on stdout,
                                                            # "stop" / "cancel" written to job.json.control

Finished outputs are skipped unless --overwrite (so an interrupted run resumes). Every file is written as
<name>.h5.partial and renamed when complete. The models load once for the whole dataset.
"""
import argparse
import json
import os
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

from . import __version__
from .dataset import (PARTIAL_SUFFIX, SourceItem, companion_files, copy_if_newer,
                      discover, load_manifest, manifest_item, manifest_path, new_manifest, now_iso, output_complete,
                      plan_outputs, probe_item, save_manifest)

LOG_NAME = 'vhrm_conversion_log.txt'
LOCK_NAME = 'vhrm_conversion.lock'


class Control:
    """stop: finish the current source, then stop. cancel: abandon the current source as soon as possible."""

    def __init__(self):
        self.stop = threading.Event()
        self.cancel = threading.Event()


def pid_alive(pid):
    if sys.platform == 'win32':
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x00100000, False, int(pid))  # SYNCHRONIZE
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == 0x102  # WAIT_TIMEOUT: still running
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def acquire_lock(output_root):
    """One conversion per output root: vhrm_conversion.lock holds the converting process id."""
    lock = Path(output_root) / LOCK_NAME
    try:
        owner = int(lock.read_text(encoding='utf-8').split()[0])
    except (OSError, ValueError, IndexError):
        owner = None
    if owner and owner != os.getpid() and pid_alive(owner):
        raise RuntimeError(f'another conversion (process {owner}) is writing to {output_root}; wait for it or stop it '
                           f'(delete {lock.name} if that process is not a conversion)')
    lock.write_text(f'{os.getpid()}\n', encoding='utf-8')
    return lock


def _h5_summary(path):
    import h5py
    import numpy as np
    with h5py.File(path, 'r') as f:
        status = f['frame_status'][:]
        return {'rows': int(len(status)), 'ok_rows': int((status == 1).sum()),
                'effective_fps': float(f.attrs.get('effective_fps', np.nan)),
                'frame_selection_mode': str(f.attrs.get('frame_selection_mode', ''))}


def run_job(job, emit, control=None):
    """job: dict (see ConvertWindow.job / main). emit(event dict). Returns the final status."""
    control = control or Control()
    source_root, output_root, mode = Path(job['source_root']).resolve(), Path(job['output_root']).resolve(), job['mode']
    items = job['items']
    output_root.mkdir(parents=True, exist_ok=True)
    log_file = open(output_root / LOG_NAME, 'a', encoding='utf-8')

    def log(text):
        stamp = time.strftime('%H:%M:%S')
        for line in str(text).splitlines() or ['']:
            log_file.write(f'{stamp} {line}\n')
            emit({'event': 'log', 'text': line})
        log_file.flush()

    def stage(text):
        log(text)
        emit({'event': 'stage', 'text': text})

    summary = {'done': 0, 'skipped': 0, 'failed': 0}
    status = 'completed'
    lock = None
    try:
        log(f'=== vHRM Studio {__version__}: {len(items)} sources, {mode} | {source_root} -> {output_root} ===')
        lock = acquire_lock(output_root)
        manifest = load_manifest(manifest_path(output_root)) or new_manifest(source_root, output_root, mode)
        manifest.update({'mode': mode, 'source_root': str(source_root.resolve()),
                         'output_root': str(output_root.resolve()),
                         'settings': {**job.get('settings', {}), 'vhrm_studio_version': __version__}})

        if mode == 'mirror' and job.get('copy_companions', True):
            stage('Copying the other dataset files...')
            sources = [SourceItem(i['kind'], i['path'], i['rel']) for i in job.get('all_items', items)]
            files = companion_files(source_root, sources, exclude=output_root)
            copied = 0
            for k, path in enumerate(files):
                if control.cancel.is_set():
                    break
                if copy_if_newer(path, output_root / os.path.relpath(path, source_root)):
                    copied += 1
                emit({'event': 'copy', 'done': k + 1, 'total': len(files), 'text': path.name})
            manifest['companion_files'] = len(files)
            log(f'{len(files)} other files ({copied} copied, {len(files) - copied} already up to date)')
        save_manifest(manifest)

        todo = []
        for index, item in enumerate(items):
            output = Path(item['output'])
            entry = manifest_item(manifest, item['rel'])
            entry.update({'output': item['output_rel'], 'kind': item['kind']})
            Path(str(output) + PARTIAL_SUFFIX).unlink(missing_ok=True)  # left by an interrupted run
            if not job.get('overwrite') and output_complete(output):
                if entry.get('status') != 'done':
                    entry.update(status='done', error='', **_h5_summary(output))
                summary['skipped'] += 1
                emit({'event': 'item_done', 'index': index, 'status': 'skipped', 'rows': entry.get('rows'),
                      'ok': entry.get('ok_rows')})
            else:
                entry.setdefault('status', 'pending')
                todo.append(index)
        save_manifest(manifest)
        if summary['skipped']:
            log(f'{summary["skipped"]} sources already converted (skipped; use overwrite to redo them)')
        if not todo:
            stage('Nothing to convert.')
            return status

        stage('Loading the models (about a minute)...')
        from .pipeline.runner import Cancelled, SkinUVPipeline
        pipeline = SkinUVPipeline(job['assets_dir'], checkpoint=job.get('checkpoint') or None,
                                  yolo_weights=job.get('yolo_weights') or None,
                                  compile_backbone=job.get('compile_backbone', True), log=log)
        settings = {**job.get('settings', {}), 'ffmpeg_dir': job.get('ffmpeg_dir') or None}
        for position, index in enumerate(todo):
            if control.stop.is_set() or control.cancel.is_set():
                status = 'cancelled' if control.cancel.is_set() else 'stopped'
                break
            item = items[index]
            output = Path(item['output'])
            partial = Path(str(output) + PARTIAL_SUFFIX)
            entry = manifest_item(manifest, item['rel'])
            stage(f'[{position + 1}/{len(todo)}] {item["rel"]}')
            emit({'event': 'item_start', 'index': index})
            last = [0.0]

            def progress(state, index=index):
                now = time.perf_counter()
                if now - last[0] >= 0.25 or state['done'] >= state['rows']:
                    last[0] = now
                    emit({'event': 'item_progress', 'index': index, **state})

            output.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            try:
                extra = {'source_relative_path': item['rel'], 'dataset_source_root': str(source_root.resolve())}
                stats = pipeline.run_video((item['kind'], item['path']), partial, settings, progress=progress,
                                           should_stop=control.cancel.is_set, extra_attrs=extra)
                os.replace(partial, output)
                entry.update(status='done', error='', seconds=round(time.perf_counter() - t0, 1), finished=now_iso(),
                             **_h5_summary(output))
                if mode == 'mirror' and job.get('copy_sources'):
                    target = output_root / item['rel']
                    log(f'copying the source to {target}')
                    if item['kind'] == 'images':
                        shutil.copytree(item['path'], target, dirs_exist_ok=True, copy_function=shutil.copy2)
                    else:
                        copy_if_newer(item['path'], target)
                    entry['source_copied'] = True
                summary['done'] += 1
                emit({'event': 'item_done', 'index': index, 'status': 'done', 'rows': entry['rows'],
                      'ok': entry['ok_rows'], 'seconds': entry['seconds'], 'fps': stats['fps']})
            except Cancelled:
                partial.unlink(missing_ok=True)
                log(f'{item["rel"]}: cancelled')
                emit({'event': 'item_done', 'index': index, 'status': 'cancelled'})
                status = 'cancelled'
                break
            except Exception as exc:
                partial.unlink(missing_ok=True)
                error = f'{type(exc).__name__}: {exc}'
                log(f'{item["rel"]}: FAILED {error}\n{traceback.format_exc()}')
                entry.update(status='failed', error=error, finished=now_iso())
                summary['failed'] += 1
                emit({'event': 'item_done', 'index': index, 'status': 'failed', 'error': error})
                import torch
                torch.cuda.empty_cache()
            finally:
                save_manifest(manifest)
        return status
    except Exception as exc:
        status = 'error'
        log(f'FAILED: {type(exc).__name__}: {exc}\n{traceback.format_exc()}')
        emit({'event': 'error', 'text': f'{type(exc).__name__}: {exc}'})
        return status
    finally:
        if lock is not None:
            lock.unlink(missing_ok=True)
        log(f'=== {status}: {summary["done"]} converted, {summary["skipped"]} skipped, {summary["failed"]} failed ===')
        emit({'event': 'finished', 'status': status, **summary})
        log_file.close()


# ---------------------------------------------------------------- GUI worker (subprocess)
def worker_main(job_path):
    """JSON-line events on stdout; 'stop' / 'cancel' are written by the GUI into <job>.control. Anything else
    printed goes to stderr. (stdin is not used: on Windows a thread blocked reading a pipe makes DLL loading hang.)"""
    proto = os.fdopen(os.dup(1), 'w', buffering=1, encoding='utf-8')
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    lock = threading.Lock()

    def emit(event):
        with lock:
            proto.write(json.dumps(event, default=str) + '\n')
            proto.flush()

    control = Control()
    control_file = Path(str(job_path) + '.control')
    job = json.loads(Path(job_path).read_text(encoding='utf-8'))
    parent = job.get('parent_pid')

    def poll_commands():
        while not control.cancel.is_set():
            try:
                commands = control_file.read_text(encoding='utf-8').split()
            except OSError:
                commands = []
            if 'stop' in commands:
                control.stop.set()
            if 'cancel' in commands or (parent and not pid_alive(parent)):  # the window was closed or crashed
                control.cancel.set()
            time.sleep(0.3)

    threading.Thread(target=poll_commands, daemon=True).start()
    status = run_job(job, emit, control)
    return 0 if status in ('completed', 'stopped', 'cancelled') else 1


# ---------------------------------------------------------------- command line
def _console_emitter():
    state = {'line': False}

    def emit(event):
        kind = event['event']
        if kind == 'item_progress':
            done, rows = event['done'], max(event['rows'], 1)
            sys.stdout.write(f'\r  row {done}/{rows} ({100 * done / rows:5.1f}%)  {event["fps"]:5.1f} fps  ')
            sys.stdout.flush()
            state['line'] = True
            return
        if state['line']:
            sys.stdout.write('\n')
            state['line'] = False
        if kind == 'log':
            print(event['text'])
        elif kind == 'copy' and event['done'] == event['total']:
            print(f'  copied {event["total"]} files')

    return emit


def main(argv=None):
    from .assets import find_assets_dir
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source_root', nargs='?', type=Path, help='dataset folder (searched recursively)')
    parser.add_argument('--job', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--out', type=Path, default=None,
                        help='new dataset root mirroring the source tree (default: HDF5 files next to the sources)')
    parser.add_argument('--no-copy-other', action='store_true', help='mirror: do not copy the other dataset files')
    parser.add_argument('--copy-sources', action='store_true', help='mirror: also copy the source videos / images')
    parser.add_argument('--overwrite', action='store_true', help='redo sources whose HDF5 file already exists')
    parser.add_argument('--assets', type=Path, default=None, help='SMPL-X / FLAME assets folder (SMPLX/, FLAME/)')
    parser.add_argument('--checkpoint', type=Path, default=None, help='PEAR ehm_model_stage1.pt (default: download)')
    parser.add_argument('--yolo', type=Path, default=None, help='yolov8x.pt (default: download)')
    parser.add_argument('--ffmpeg-dir', type=Path, default=None, help='folder of ffmpeg / ffprobe (default: PATH)')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-frames', type=int, default=None, help='rows per source (tests)')
    parser.add_argument('--target-fps', type=float, default=30.0)
    parser.add_argument('--image-fps', type=float, default=30.0,
                        help='frame rate of image sequences whose file names hold no time stamps')
    parser.add_argument('--no-packed-uv', action='store_true', help='do not store the packed semantic UV atlas')
    parser.add_argument('--no-compile', action='store_true', help='do not torch.compile the backbone')
    parser.add_argument('--min-images', type=int, default=30, help='images that make a folder an image sequence')
    parser.add_argument('--dry-run', action='store_true', help='list the sources and outputs, convert nothing')
    args = parser.parse_args(argv)
    if args.job:
        return worker_main(args.job)
    if args.source_root is None:
        parser.error('source_root is required')
    source_root = args.source_root.resolve()
    mode = 'mirror' if args.out else 'in_place'
    items = discover(source_root, min_images=args.min_images, exclude=args.out)
    output_root = plan_outputs(items, source_root, mode, args.out)
    print(f'{len(items)} sources in {source_root} -> {output_root} ({mode})')
    for item in items:
        print(f'  {item.kind:6} {item.rel}  ->  {item.output_rel}')
    if args.dry_run or not items:
        return 0
    assets = args.assets or find_assets_dir()
    if assets is None:
        parser.error('SMPL-X / FLAME assets not found: pass --assets or set VHRM_ASSETS_DIR')
    for item in items:
        probe_item(item, args.ffmpeg_dir)
    job = {
        'source_root': str(source_root), 'output_root': str(output_root), 'mode': mode,
        'items': [item.to_job() for item in items], 'copy_companions': not args.no_copy_other,
        'copy_sources': args.copy_sources, 'overwrite': args.overwrite, 'assets_dir': str(assets),
        'checkpoint': str(args.checkpoint or ''), 'yolo_weights': str(args.yolo or ''),
        'ffmpeg_dir': str(args.ffmpeg_dir or ''), 'compile_backbone': not args.no_compile,
        'settings': {'batch_size': args.batch_size, 'max_frames': args.max_frames, 'target_fps': args.target_fps,
                     'image_fps': args.image_fps, 'packed_uv': not args.no_packed_uv},
    }
    control = Control()
    try:
        status = run_job(job, _console_emitter(), control)
    except KeyboardInterrupt:
        return 130
    return 0 if status == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
