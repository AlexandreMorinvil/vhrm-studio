"""Dataset conversion window: choose a dataset folder, review the sources found, choose where the HDF5 files go,
then convert them in a worker process (vhrm_studio.convert) with per-source and overall progress."""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

from . import dataset
from .assets import ASSETS_HELP, missing_assets

STATUS_COLORS = {'done': '#4a4', 'converted': '#4a4', 'skipped': '#4a4', 'failed': '#d55', 'cancelled': '#da4',
                 'running': '#59f', 'exists': '#8a8', 'probe error': '#d55'}
COLUMNS = ['Source', 'Type', 'Frames', 'FPS', 'Duration', 'Size', 'Output', 'Status']
C_SOURCE, C_TYPE, C_FRAMES, C_FPS, C_DURATION, C_SIZE, C_OUTPUT, C_STATUS = range(len(COLUMNS))


def hms(seconds):
    if seconds is None or seconds != seconds or seconds == float('inf'):
        return '--:--'
    seconds = int(round(seconds))
    h, rest = divmod(seconds, 3600)
    return f'{h}:{rest // 60:02d}:{rest % 60:02d}' if h else f'{rest // 60:d}:{rest % 60:02d}'


class ScanThread(QtCore.QThread):
    """discover() then probe every source (ffprobe headers / first image)."""
    listed = Signal(list)
    probed = Signal(int)

    def __init__(self, root, min_images, exclude, ffmpeg_dir):
        super().__init__()
        self.root, self.min_images, self.exclude, self.ffmpeg_dir = root, min_images, exclude, ffmpeg_dir
        self.items = []
        self.error = None

    def run(self):
        try:
            self.items = dataset.discover(self.root, min_images=self.min_images, exclude=self.exclude,
                                          should_stop=self.isInterruptionRequested)
            self.listed.emit(self.items)
            for i, item in enumerate(self.items):
                if self.isInterruptionRequested():
                    return
                dataset.probe_item(item, self.ffmpeg_dir)
                self.probed.emit(i)
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'


class SortItem(QtWidgets.QTreeWidgetItem):
    def __lt__(self, other):
        col = self.treeWidget().sortColumn()
        a, b = self.data(col, Qt.ItemDataRole.UserRole), other.data(col, Qt.ItemDataRole.UserRole)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a < b
        return dataset.natural_key(self.text(col)) < dataset.natural_key(other.text(col))


class ConvertWindow(QtWidgets.QWidget):
    """config() returns the model / tool paths of the application settings; openFolder / openFile ask the viewer
    to show results; settingsRequested opens the application settings."""
    openFolder = Signal(str)
    openFile = Signal(str)
    settingsRequested = Signal()

    def __init__(self, settings, config, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle('vHRM Studio - Convert a dataset')
        self.settings, self.config = settings, config
        self.items = []
        self.tree_items = []
        self.scan = None
        self.process = None
        self.run = None
        self._stdout = b''
        self._build()
        self._load_settings()
        self._update_mode()
        self.refresh_models()
        self.resize(1250, 900)
        geometry = self.settings.value('convert/geometry')
        if geometry is not None:
            self.restoreGeometry(geometry)

    # ------------------------------------------------------------------ UI
    def _build(self):
        def browse_button(edit, title):
            button = QtWidgets.QToolButton(text='Browse...')
            button.clicked.connect(lambda: self._browse(edit, title))
            return button

        # source
        self.source_edit = QtWidgets.QLineEdit(placeholderText='dataset folder, e.g. C:\\dataset\\UBFC_DATASET\\DATASET_2')
        self.source_edit.returnPressed.connect(self.start_scan)
        self.scan_button = QtWidgets.QPushButton('Scan')
        self.scan_button.setToolTip('Find the videos and image sequences in the folder and its subfolders')
        self.scan_button.clicked.connect(self.start_scan)
        self.min_images = QtWidgets.QSpinBox(minimum=2, maximum=100000, value=dataset.MIN_SEQUENCE_IMAGES)
        self.min_images.setToolTip('A folder holding at least this many images is one image sequence (PURE, ...)')
        source_row = QtWidgets.QHBoxLayout()
        source_row.addWidget(self.source_edit, 1)
        source_row.addWidget(browse_button(self.source_edit, 'Dataset folder'))
        source_row.addWidget(self.scan_button)
        options_row = QtWidgets.QHBoxLayout()
        options_row.addWidget(QtWidgets.QLabel('Image sequence: a folder with at least'))
        options_row.addWidget(self.min_images)
        options_row.addWidget(QtWidgets.QLabel('images'))
        options_row.addSpacing(24)
        self.scan_label = QtWidgets.QLabel()
        options_row.addWidget(self.scan_label, 1)
        source_box = QtWidgets.QGroupBox('1. Dataset')
        source_layout = QtWidgets.QVBoxLayout(source_box)
        source_layout.addLayout(source_row)
        source_layout.addLayout(options_row)

        # sources table
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(COLUMNS)
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSortingEnabled(True)
        self.tree.sortByColumn(C_SOURCE, Qt.SortOrder.AscendingOrder)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.itemDoubleClicked.connect(self._open_item)
        self.tree.itemChanged.connect(lambda *_: self._update_summary())
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._tree_menu)
        select_all = QtWidgets.QToolButton(text='All', toolTip='Convert every source')
        select_all.clicked.connect(lambda: self._check_all(True))
        select_none = QtWidgets.QToolButton(text='None', toolTip='Convert no source')
        select_none.clicked.connect(lambda: self._check_all(False))
        select_todo = QtWidgets.QToolButton(text='Not converted', toolTip='Only the sources without an output yet')
        select_todo.clicked.connect(self._check_todo)
        self.summary_label = QtWidgets.QLabel()
        table_bar = QtWidgets.QHBoxLayout()
        table_bar.addWidget(QtWidgets.QLabel('Convert:'))
        for b in (select_all, select_none, select_todo):
            table_bar.addWidget(b)
        table_bar.addSpacing(24)
        table_bar.addWidget(self.summary_label, 1)
        table_widget = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout(table_widget)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addWidget(self.tree, 1)
        table_layout.addLayout(table_bar)

        # output
        self.in_place_radio = QtWidgets.QRadioButton('Next to each source (same folder as the video / image folder)')
        self.mirror_radio = QtWidgets.QRadioButton('In a new dataset folder with the same structure:')
        self.mode_group = QtWidgets.QButtonGroup(self)
        self.mode_group.addButton(self.in_place_radio)
        self.mode_group.addButton(self.mirror_radio)
        self.mode_group.buttonToggled.connect(lambda *_: self._update_mode())
        self.output_edit = QtWidgets.QLineEdit(placeholderText='new dataset root, e.g. C:\\dataset\\UBFC_DATASET\\DATASET_2_skin_uv')
        self.output_edit.editingFinished.connect(self._replan)
        self.output_browse = browse_button(self.output_edit, 'New dataset folder')
        self.copy_other = QtWidgets.QCheckBox('Copy the other files of the dataset (ground-truth signals, metadata, ...)')
        self.copy_sources = QtWidgets.QCheckBox('Also copy the source videos / image folders')
        self.overwrite = QtWidgets.QCheckBox('Redo sources that are already converted (overwrite their HDF5 file)')
        self.overwrite.toggled.connect(lambda _: self._update_summary())
        mirror_row = QtWidgets.QHBoxLayout()
        mirror_row.addWidget(self.output_edit, 1)
        mirror_row.addWidget(self.output_browse)
        mirror_options = QtWidgets.QVBoxLayout()
        mirror_options.setContentsMargins(24, 0, 0, 0)
        mirror_options.addLayout(mirror_row)
        mirror_options.addWidget(self.copy_other)
        mirror_options.addWidget(self.copy_sources)
        output_box = QtWidgets.QGroupBox('2. Output')
        output_layout = QtWidgets.QVBoxLayout(output_box)
        output_layout.addWidget(self.in_place_radio)
        output_layout.addWidget(self.mirror_radio)
        output_layout.addLayout(mirror_options)
        output_layout.addWidget(self.overwrite)
        output_layout.addWidget(QtWidgets.QLabel(
            f'Each source gives <i>&lt;name&gt;{dataset.H5_SUFFIX}</i>; <i>{dataset.MANIFEST_NAME}</i> (and a CSV copy) in '
            'the output root lists which source every file comes from. Interrupted runs resume where they stopped.',
            wordWrap=True))

        # processing settings
        self.batch_size = QtWidgets.QSpinBox(minimum=1, maximum=64, value=8)
        self.batch_size.setToolTip('Frames per GPU batch (8 needs ~10 GB of GPU memory). The person box is updated '
                                   'once per batch.')
        self.max_frames = QtWidgets.QSpinBox(minimum=0, maximum=10_000_000, value=0, specialValueText='all')
        self.max_frames.setToolTip('Rows per source (for quick tests); "all" converts the whole source')
        self.target_fps = QtWidgets.QDoubleSpinBox(minimum=1, maximum=240, value=30.0, decimals=2)
        self.target_fps.setToolTip('Faster sources are subsampled to about this rate (exact strides when possible)')
        self.image_fps = QtWidgets.QDoubleSpinBox(minimum=1, maximum=1000, value=30.0, decimals=3)
        self.image_fps.setToolTip('Frame rate of image sequences whose file names hold no time stamps '
                                  '(PURE names hold nanosecond time stamps, which are used instead)')
        self.packed_uv = QtWidgets.QCheckBox('Packed semantic UV atlas', checked=True)
        self.packed_uv.setToolTip('Also store packed_uv_skin_rgb (every body part unrolled to its own rectangle)')
        self.accumulate = QtWidgets.QCheckBox('Accumulated skin texture', checked=True)
        self.save_png = QtWidgets.QCheckBox('Save it as PNG too', checked=False)
        self.stabilize = QtWidgets.QCheckBox('Temporal stabilization', checked=True)
        self.stabilize.setToolTip('Person box, body shape, camera and pose filters of the pipeline (One Euro / EMA)')
        self.compile = QtWidgets.QCheckBox('torch.compile the backbone', checked=True)
        self.compile.setToolTip('Faster with Triton installed (first compile ~1 min); ignored without Triton')
        grid = QtWidgets.QGridLayout()
        for k, (label, widget) in enumerate((('Batch size', self.batch_size), ('Rows per source', self.max_frames),
                                             ('Target fps', self.target_fps), ('Image sequence fps', self.image_fps))):
            grid.addWidget(QtWidgets.QLabel(label), k // 2, 2 * (k % 2))
            grid.addWidget(widget, k // 2, 2 * (k % 2) + 1)
        checks = QtWidgets.QGridLayout()
        for k, w in enumerate((self.packed_uv, self.accumulate, self.save_png, self.stabilize, self.compile)):
            checks.addWidget(w, k // 3, k % 3)
        self.models_label = QtWidgets.QLabel(wordWrap=True)
        self.models_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        models_button = QtWidgets.QToolButton(text='Model settings...')
        models_button.clicked.connect(self.settingsRequested.emit)
        models_row = QtWidgets.QHBoxLayout()
        models_row.addWidget(self.models_label, 1)
        models_row.addWidget(models_button)
        settings_box = QtWidgets.QGroupBox('3. Processing')
        settings_layout = QtWidgets.QVBoxLayout(settings_box)
        settings_layout.addLayout(grid)
        settings_layout.addLayout(checks)
        settings_layout.addLayout(models_row)

        # run
        self.start_button = QtWidgets.QPushButton('Start conversion')
        self.start_button.setStyleSheet('font-weight: bold;')
        self.start_button.clicked.connect(self.start)
        self.stop_button = QtWidgets.QPushButton('Stop after this source')
        self.stop_button.setToolTip('Finish the source being converted, then stop (resume later with Start)')
        self.stop_button.clicked.connect(lambda: self._send('stop'))
        self.cancel_button = QtWidgets.QPushButton('Cancel now')
        self.cancel_button.setToolTip('Abandon the source being converted (its partial file is deleted)')
        self.cancel_button.clicked.connect(self.cancel)
        self.view_button = QtWidgets.QPushButton('Open the output in the viewer')
        self.view_button.clicked.connect(lambda: self.openFolder.emit(str(self._output_root() or '')))
        self.stage_label = QtWidgets.QLabel('Idle')
        self.stage_label.setStyleSheet('font-weight: bold;')
        self.overall_bar = QtWidgets.QProgressBar(textVisible=True)
        self.overall_label = QtWidgets.QLabel(' ')
        self.item_bar = QtWidgets.QProgressBar(textVisible=True)
        self.item_label = QtWidgets.QLabel(' ')
        buttons = QtWidgets.QHBoxLayout()
        for b in (self.start_button, self.stop_button, self.cancel_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(self.view_button)
        bars = QtWidgets.QGridLayout()
        bars.addWidget(QtWidgets.QLabel('Dataset'), 0, 0)
        bars.addWidget(self.overall_bar, 0, 1)
        bars.addWidget(self.overall_label, 1, 1)
        bars.addWidget(QtWidgets.QLabel('Source'), 2, 0)
        bars.addWidget(self.item_bar, 2, 1)
        bars.addWidget(self.item_label, 3, 1)
        run_box = QtWidgets.QGroupBox('4. Conversion')
        run_layout = QtWidgets.QVBoxLayout(run_box)
        run_layout.addLayout(buttons)
        run_layout.addWidget(self.stage_label)
        run_layout.addLayout(bars)

        self.log = QtWidgets.QPlainTextEdit(readOnly=True)
        self.log.setMaximumBlockCount(20000)
        self.log.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))

        top = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(source_box)
        top_layout.addWidget(table_widget, 1)
        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        options = QtWidgets.QHBoxLayout()
        options.addWidget(output_box, 1)
        options.addWidget(settings_box, 1)
        controls_layout.addLayout(options)
        controls_layout.addWidget(run_box)
        splitter = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        for k, (widget, stretch) in enumerate(((top, 3), (controls, 0), (self.log, 2))):
            splitter.addWidget(widget)
            splitter.setStretchFactor(k, stretch)
            splitter.setCollapsible(k, False)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(splitter)
        self._set_running(False)

    def _browse(self, edit, title):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, title, edit.text())
        if folder:
            edit.setText(folder)
            if edit is self.source_edit:
                self.start_scan()
            else:
                self._replan()

    # ------------------------------------------------------------------ settings
    SETTINGS = ('source_root', 'output_root', 'mode', 'copy_other', 'copy_sources', 'overwrite', 'min_images',
                'batch_size', 'max_frames', 'target_fps', 'image_fps', 'packed_uv', 'accumulate', 'save_png',
                'stabilize', 'compile')

    def _load_settings(self):
        s, prefix = self.settings, 'convert/'
        self.source_edit.setText(s.value(prefix + 'source_root', '') or '')
        self.output_edit.setText(s.value(prefix + 'output_root', '') or '')
        (self.mirror_radio if s.value(prefix + 'mode', 'mirror') == 'mirror' else self.in_place_radio).setChecked(True)
        for name in ('copy_other', 'copy_sources', 'overwrite', 'packed_uv', 'accumulate', 'save_png', 'stabilize',
                     'compile'):
            widget = getattr(self, name)
            widget.setChecked(s.value(prefix + name, widget.isChecked(), type=bool))
        if s.value(prefix + 'copy_other') is None:
            self.copy_other.setChecked(True)
        for name in ('min_images', 'batch_size', 'max_frames'):
            widget = getattr(self, name)
            widget.setValue(int(s.value(prefix + name, widget.value())))
        for name in ('target_fps', 'image_fps'):
            widget = getattr(self, name)
            widget.setValue(float(s.value(prefix + name, widget.value())))

    def _save_settings(self):
        s, prefix = self.settings, 'convert/'
        s.setValue(prefix + 'source_root', self.source_edit.text())
        s.setValue(prefix + 'output_root', self.output_edit.text())
        s.setValue(prefix + 'mode', self._mode())
        for name in ('copy_other', 'copy_sources', 'overwrite', 'packed_uv', 'accumulate', 'save_png', 'stabilize',
                     'compile'):
            s.setValue(prefix + name, getattr(self, name).isChecked())
        for name in ('min_images', 'batch_size', 'max_frames', 'target_fps', 'image_fps'):
            s.setValue(prefix + name, getattr(self, name).value())
        s.setValue('convert/geometry', self.saveGeometry())

    def refresh_models(self):
        cfg = self.config()
        missing = missing_assets(cfg.get('assets_dir'))
        if not cfg.get('assets_dir'):
            text = '<span style="color:#d55">Body-model assets folder not set.</span> ' + ASSETS_HELP
        elif missing:
            text = (f'<span style="color:#d55">Body-model assets incomplete</span> ({cfg["assets_dir"]}): missing '
                    f'{", ".join(missing[:4])}{" ..." if len(missing) > 4 else ""}')
        else:
            text = f'<span style="color:#4a4">Body-model assets OK</span>: {cfg["assets_dir"]}'
        text += ('<br>PEAR checkpoint: ' + (cfg.get('checkpoint') or 'automatic (weights folder next to the assets, '
                                            'else downloaded once from Hugging Face)'))
        text += '<br>YOLO weights: ' + (cfg.get('yolo_weights') or 'automatic (downloaded once)')
        self.models_label.setText(text)
        return not missing

    # ------------------------------------------------------------------ scan / plan
    def _mode(self):
        return 'mirror' if self.mirror_radio.isChecked() else 'in_place'

    def _output_root(self):
        if self._mode() == 'in_place':
            text = self.source_edit.text().strip()
        else:
            text = self.output_edit.text().strip()
        return Path(text).resolve() if text else None

    def _update_mode(self):
        mirror = self._mode() == 'mirror'
        for w in (self.output_edit, self.output_browse, self.copy_other, self.copy_sources):
            w.setEnabled(mirror and self.run is None)
        self._replan()

    def start_scan(self):
        if self.run is not None:
            return
        root = self.source_edit.text().strip()
        if not root or not Path(root).is_dir():
            self.scan_label.setText('<span style="color:#d55">Choose an existing dataset folder</span>')
            return
        if self.scan is not None:
            self.scan.requestInterruption()
            self.scan.wait()
        self.tree.clear()
        self.items, self.tree_items = [], []
        out = self._output_root() if self._mode() == 'mirror' else None
        self.scan_label.setText('scanning...')
        self.scan_button.setEnabled(False)
        self.scan = scan = ScanThread(Path(root).resolve(), self.min_images.value(), out,
                                      self.config().get('ffmpeg_dir') or None)
        scan.listed.connect(lambda items: self.scan is scan and self._listed(items))
        scan.probed.connect(lambda i: self.scan is scan and self._probed(i))
        scan.finished.connect(lambda: self.scan is scan and self._scan_finished())
        scan.start()

    def _listed(self, items):
        self.items = items
        self.tree.setSortingEnabled(False)
        self.tree.blockSignals(True)
        for item in items:
            row = SortItem()
            row.setFlags(row.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            row.setCheckState(C_SOURCE, Qt.CheckState.Checked)
            row.setText(C_SOURCE, item.rel)
            row.setToolTip(C_SOURCE, str(item.path))
            row.setText(C_TYPE, 'video' if item.kind == 'video' else f'images ({len(item.files)})')
            self.tree.addTopLevelItem(row)
            self.tree_items.append(row)
        self.tree.blockSignals(False)
        self.tree.setSortingEnabled(True)
        self._replan()
        self.scan_label.setText(f'probing {len(items)} sources...')

    def _probed(self, i):
        item, row = self.items[i], self.tree_items[i]
        self.tree.blockSignals(True)
        if item.probe_error:
            row.setText(C_STATUS, 'probe error')
            row.setToolTip(C_STATUS, item.probe_error)
            row.setForeground(C_STATUS, QtGui.QBrush(QtGui.QColor(STATUS_COLORS['probe error'])))
        values = ((C_FRAMES, item.frames, f'{item.frames:,}' if item.frames else '?'),
                  (C_FPS, item.fps, f'{item.fps:.2f}' if item.fps else '?'),
                  (C_DURATION, item.duration, hms(item.duration) if item.duration else '?'),
                  (C_SIZE, item.size[0] * item.size[1] if item.size else 0,
                   f'{item.size[0]}x{item.size[1]}' if item.size else '?'))
        for col, value, text in values:
            row.setText(col, text)
            row.setData(col, Qt.ItemDataRole.UserRole, value or 0)
        self.tree.blockSignals(False)
        self.scan_label.setText(f'probing {i + 1} / {len(self.items)}...')

    def _scan_finished(self):
        self.scan_button.setEnabled(self.run is None)
        if self.scan.error:
            self.scan_label.setText(f'<span style="color:#d55">{self.scan.error}</span>')
            return
        videos = sum(item.kind == 'video' for item in self.items)
        self.scan_label.setText(f'{len(self.items)} sources: {videos} videos, {len(self.items) - videos} image '
                                f'sequences')
        for col in range(1, len(COLUMNS)):
            self.tree.resizeColumnToContents(col)
        self.tree.setColumnWidth(C_SOURCE, max(self.tree.columnWidth(C_SOURCE), 220))
        self._update_summary()

    def _replan(self):
        if not self.items or self.run is not None:
            self._update_summary()
            return
        root = Path(self.source_edit.text().strip() or '.').resolve()
        out = self._output_root() if self._mode() == 'mirror' else None
        try:
            dataset.plan_outputs(self.items, root, self._mode(), out)
            error = None
        except ValueError as exc:
            error = str(exc)
        self.tree.blockSignals(True)
        for item, row in zip(self.items, self.tree_items):
            if error:
                row.setText(C_OUTPUT, '')
                continue
            row.setText(C_OUTPUT, item.output_rel)
            row.setToolTip(C_OUTPUT, str(item.output))
            if row.text(C_STATUS) in ('', 'exists', 'converted'):
                exists = item.output.is_file()
                self._set_status(row, 'exists' if exists else '')
        self.tree.blockSignals(False)
        self._update_summary(error)

    def _set_status(self, row, status, tooltip=None):
        row.setText(C_STATUS, status)
        row.setForeground(C_STATUS, QtGui.QBrush(QtGui.QColor(STATUS_COLORS.get(status.split(' ')[0], '#ccc'))))
        if tooltip is not None:
            row.setToolTip(C_STATUS, tooltip)

    def _checked(self):
        return [i for i, row in enumerate(self.tree_items) if row.checkState(C_SOURCE) == Qt.CheckState.Checked]

    def _check_all(self, on):
        self.tree.blockSignals(True)
        for row in self.tree_items:
            row.setCheckState(C_SOURCE, Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self.tree.blockSignals(False)
        self._update_summary()

    def _check_todo(self):
        self.tree.blockSignals(True)
        for item, row in zip(self.items, self.tree_items):
            done = item.output is not None and item.output.is_file()
            row.setCheckState(C_SOURCE, Qt.CheckState.Unchecked if done else Qt.CheckState.Checked)
        self.tree.blockSignals(False)
        self._update_summary()

    def _update_summary(self, error=None):
        if error:
            self.summary_label.setText(f'<span style="color:#d55">{error}</span>')
            return
        checked = self._checked()
        frames = sum(self.items[i].estimated_rows(self.target_fps.value()) for i in checked)
        exists = sum(1 for i in checked if self.items[i].output is not None and self.items[i].output.is_file())
        text = f'{len(checked)} / {len(self.items)} selected, ~{frames:,} rows'
        if exists:
            text += f' ({exists} already converted: ' + ('redone)' if self.overwrite.isChecked() else 'skipped)')
        self.summary_label.setText(text)

    def _tree_menu(self, pos):
        row = self.tree.itemAt(pos)
        if row is None:
            return
        item = self.items[self.tree_items.index(row)]
        menu = QtWidgets.QMenu(self)
        if item.output is not None and item.output.is_file():
            menu.addAction('Open in the viewer', lambda: self.openFile.emit(str(item.output)))
        menu.addAction('Show the source folder', lambda: QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(str(item.path if item.kind == 'images' else item.path.parent))))
        menu.addSeparator()
        menu.addAction('Convert the selected rows only', self._check_selected)
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _check_selected(self):
        selected = set(map(id, self.tree.selectedItems()))
        self.tree.blockSignals(True)
        for row in self.tree_items:
            row.setCheckState(C_SOURCE, Qt.CheckState.Checked if id(row) in selected else Qt.CheckState.Unchecked)
        self.tree.blockSignals(False)
        self._update_summary()

    def _open_item(self, row, _col):
        item = self.items[self.tree_items.index(row)]
        if item.output is not None and item.output.is_file():
            self.openFile.emit(str(item.output))

    # ------------------------------------------------------------------ run
    def job(self, indices):
        cfg = self.config()
        source_root = Path(self.source_edit.text().strip()).resolve()
        stabilize = self.stabilize.isChecked()
        from .pipeline.config import TEMPORAL_CONFIG
        temporal = {**TEMPORAL_CONFIG, 'bbox_enabled': stabilize, 'shape_enabled': stabilize,
                    'camera_enabled': stabilize, 'pose_enabled': stabilize}
        return {
            'source_root': str(source_root), 'output_root': str(self._output_root()), 'mode': self._mode(),
            'items': [self.items[i].to_job() for i in indices],
            'all_items': [item.to_job() for item in self.items],
            'copy_companions': self.copy_other.isChecked(), 'copy_sources': self.copy_sources.isChecked(),
            'overwrite': self.overwrite.isChecked(),
            'assets_dir': cfg.get('assets_dir') or '', 'checkpoint': cfg.get('checkpoint') or '',
            'yolo_weights': cfg.get('yolo_weights') or '', 'ffmpeg_dir': cfg.get('ffmpeg_dir') or '',
            'compile_backbone': self.compile.isChecked(), 'parent_pid': os.getpid(),
            'settings': {'batch_size': self.batch_size.value(), 'max_frames': self.max_frames.value() or None,
                         'target_fps': self.target_fps.value(), 'image_fps': self.image_fps.value(),
                         'packed_uv': self.packed_uv.isChecked(), 'accumulate': self.accumulate.isChecked(),
                         'save_accumulated_png': self.save_png.isChecked(), 'temporal': temporal},
        }

    def start(self):
        if self.run is not None:
            return
        self._replan()
        indices = sorted(self._checked())
        problem = None
        if not self.items:
            problem = 'Scan a dataset folder first.'
        elif not indices:
            problem = 'No source is selected.'
        elif self._mode() == 'mirror' and not self.output_edit.text().strip():
            problem = 'Choose the new dataset folder (or "Next to each source").'
        elif not self.refresh_models():
            problem = 'The SMPL-X / FLAME body-model assets are missing or incomplete (Model settings...).'
        elif any(self.items[i].output is None for i in indices):
            problem = self.summary_label.text()
        if problem:
            QtWidgets.QMessageBox.warning(self, 'Cannot start', problem)
            return
        out = self._output_root()
        if self._mode() == 'mirror' and dataset._inside(out, self.source_edit.text().strip()):
            answer = QtWidgets.QMessageBox.question(
                self, 'Output inside the dataset', f'{out}\nis inside the source dataset. It is excluded from '
                'the conversion; continue?')
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        self._save_settings()
        job = self.job(indices)
        fd, job_path = tempfile.mkstemp(prefix='vhrm_job_', suffix='.json')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(job, f)
        target_fps = self.target_fps.value()
        max_rows = self.max_frames.value() or None
        self.run = {
            'indices': indices, 'job_path': job_path, 'current': None, 'finished': None, 't0': time.perf_counter(),
            't_first': None, 'rows': {i: min(self.items[i].estimated_rows(target_fps) or 1, max_rows or 1 << 62)
                                      for i in indices},
            'done_rows': {}, 'statuses': {},
        }
        for i in indices:
            self._set_status(self.tree_items[i], 'queued')
        self.log.appendPlainText(f'--- {time.strftime("%Y-%m-%d %H:%M:%S")} starting ({len(indices)} sources) ---')
        self.process = process = QtCore.QProcess(self)
        process.setProgram(self._python())
        process.setArguments(['-m', 'vhrm_studio.convert', '--job', job_path])
        env = QtCore.QProcessEnvironment.systemEnvironment()
        package_root = str(Path(__file__).resolve().parent.parent)
        env.insert('PYTHONPATH', os.pathsep.join(p for p in (package_root, env.value('PYTHONPATH')) if p))
        env.insert('PYTHONUNBUFFERED', '1')
        env.insert('PYTHONIOENCODING', 'utf-8')
        process.setProcessEnvironment(env)
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.finished.connect(self._process_finished)
        self._stdout = b''
        self._set_running(True)
        self.stage_label.setText('Starting the worker...')
        self._update_progress()
        process.start()
        process.closeWriteChannel()

    @staticmethod
    def _python():
        exe = Path(sys.executable)
        if exe.name.lower() == 'pythonw.exe' and exe.with_name('python.exe').is_file():
            return str(exe.with_name('python.exe'))
        return str(exe)

    def _send(self, command):
        if self.process is not None and self.process.state() == QtCore.QProcess.ProcessState.Running:
            with open(self.run['job_path'] + '.control', 'a', encoding='utf-8') as f:
                f.write(f'{command}\n')
            self.stage_label.setText({'stop': 'Stopping after the current source...',
                                      'cancel': 'Cancelling...'}.get(command, command))
            if command == 'stop':
                self.stop_button.setEnabled(False)

    def cancel(self):
        if self.process is None:
            return
        self._send('cancel')
        QtCore.QTimer.singleShot(60000, lambda p=self.process: p is self.process and p.state() ==
                                 QtCore.QProcess.ProcessState.Running and p.kill())

    def _set_running(self, running):
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.cancel_button.setEnabled(running)
        self.scan_button.setEnabled(not running)
        for w in (self.source_edit, self.in_place_radio, self.mirror_radio, self.overwrite, self.min_images,
                  self.batch_size, self.max_frames, self.target_fps, self.image_fps, self.packed_uv, self.accumulate,
                  self.save_png, self.stabilize, self.compile):
            w.setEnabled(not running)
        mirror = self._mode() == 'mirror'
        for w in (self.output_edit, self.output_browse, self.copy_other, self.copy_sources):
            w.setEnabled(mirror and not running)

    def _read_stderr(self):
        text = bytes(self.process.readAllStandardError()).decode('utf-8', errors='replace')
        for line in text.splitlines():
            if line.strip():
                self.log.appendPlainText(line.rstrip())

    def _read_stdout(self):
        self._stdout += bytes(self.process.readAllStandardOutput())
        *lines, self._stdout = self._stdout.split(b'\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                self.log.appendPlainText(line.decode('utf-8', errors='replace'))
                continue
            self._event(event)

    def _event(self, event):
        run = self.run
        if run is None:
            return
        kind = event['event']
        if kind == 'log':
            self.log.appendPlainText(event['text'])
        elif kind == 'stage':
            self.stage_label.setText(event['text'])
        elif kind == 'copy':
            self.item_bar.setRange(0, max(event['total'], 1))
            self.item_bar.setValue(event['done'])
            self.item_label.setText(f'copying the other files: {event["done"]} / {event["total"]}  {event["text"]}')
        elif kind == 'item_start':
            i = run['indices'][event['index']]
            run['current'] = i
            run['statuses'][i] = 'running'
            if run['t_first'] is None:
                run['t_first'] = time.perf_counter()
            self._set_status(self.tree_items[i], 'running')
            self.tree.scrollToItem(self.tree_items[i])
            self.item_bar.setRange(0, 0)
            self.item_label.setText(f'{self.items[i].rel}: preparing (frame selection, warm-up)...')
        elif kind == 'item_progress':
            i = run['indices'][event['index']]
            run['rows'][i] = event['rows']
            run['done_rows'][i] = event['done']
            self.item_bar.setRange(0, max(event['rows'], 1))
            self.item_bar.setValue(event['done'])
            pct = 100.0 * event['done'] / max(event['rows'], 1)
            self._set_status(self.tree_items[i], f'running {pct:.0f}%')
            self.item_label.setText(
                f'{self.items[i].rel}: row {event["done"]:,} / {event["rows"]:,} ({pct:.1f}%) | {event["fps"]:.1f} fps '
                f'| {hms(event["elapsed"])} elapsed, {hms((event["rows"] - event["done"]) / event["fps"]) if event["fps"] else "--:--"} left'
                + (f' | {event["done"] - event["ok"]} rows without a person' if event['done'] > event['ok'] else ''))
        elif kind == 'item_done':
            i = run['indices'][event['index']]
            status = event['status']
            run['statuses'][i] = status
            row = self.tree_items[i]
            if status in ('done', 'skipped'):
                if event.get('rows'):
                    run['rows'][i] = run['done_rows'][i] = event['rows']
                text = 'converted' if status == 'done' else 'skipped (exists)'
                tip = (f'{event.get("ok", "?")} / {event.get("rows", "?")} rows with a person'
                       + (f', {event["seconds"]:.0f} s' if event.get('seconds') else ''))
                self._set_status(row, text, tip)
            elif status == 'failed':
                self._set_status(row, 'failed', event.get('error', ''))
                run['done_rows'][i] = run['rows'][i]
            else:
                self._set_status(row, status)
            if status == 'skipped':  # not part of the timed work
                run['rows'].pop(i, None)
                run['done_rows'].pop(i, None)
        elif kind == 'error':
            self.log.appendPlainText(f'ERROR: {event["text"]}')
            self.stage_label.setText(f'Error: {event["text"]}')
        elif kind == 'finished':
            run['finished'] = event
        self._update_progress()

    def _update_progress(self):
        run = self.run
        if run is None:
            return
        total = sum(run['rows'].values())
        done = sum(run['done_rows'].values())
        finished = sum(1 for s in run['statuses'].values() if s in ('done', 'failed', 'skipped'))
        self.overall_bar.setRange(0, max(total, 1))
        self.overall_bar.setValue(min(done, total))
        elapsed = time.perf_counter() - run['t0']
        rate = None
        if run['t_first'] is not None and done > 0:
            rate = done / max(time.perf_counter() - run['t_first'], 1e-6)
        eta = (total - done) / rate if rate else None
        failed = sum(1 for s in run['statuses'].values() if s == 'failed')
        self.overall_label.setText(
            f'{finished} / {len(run["indices"])} sources | {done:,} / {total:,} rows | '
            + (f'{rate:.1f} rows/s | ' if rate else '') + f'elapsed {hms(elapsed)} | remaining ~{hms(eta)}'
            + (f' | {failed} failed' if failed else ''))

    def _process_finished(self, code, _status):
        self._read_stdout()
        self._read_stderr()
        run, self.process = self.run, None
        finished = run.get('finished') if run else None
        if run is not None and run['current'] is not None and run['statuses'].get(run['current']) == 'running':
            i = run['current']
            self._set_status(self.tree_items[i], 'failed', f'the worker process ended (exit code {code})')
        for i in (run or {}).get('indices', []):
            if self.tree_items[i].text(C_STATUS) == 'queued':
                self._set_status(self.tree_items[i], '')
        if finished:
            text = (f'{finished["status"]}: {finished["done"]} converted, {finished["skipped"]} skipped, '
                    f'{finished["failed"]} failed')
        else:
            text = f'The worker process ended unexpectedly (exit code {code}); see the log.'
        self.stage_label.setText(text.capitalize())
        self.item_bar.setRange(0, 1)
        self.item_bar.setValue(0)
        self.log.appendPlainText(f'--- {time.strftime("%H:%M:%S")} {text} ---')
        if run is not None:
            Path(run['job_path']).unlink(missing_ok=True)
            Path(run['job_path'] + '.control').unlink(missing_ok=True)
        self.run = None
        self._set_running(False)
        self._update_summary()
        QtWidgets.QApplication.alert(self)

    def is_running(self):
        return self.run is not None

    def shutdown(self):
        """Cancel a running conversion without asking (the application quits), then close."""
        if self.process is not None:
            self._send('cancel')
            if not self.process.waitForFinished(30000):
                self.process.kill()
                self.process.waitForFinished(5000)
        self.run = None
        self.close()

    def closeEvent(self, event):
        if self.run is not None:
            answer = QtWidgets.QMessageBox.question(
                self, 'Conversion running', 'A conversion is running. Cancel it and close this window?')
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._send('cancel')
            if self.process is not None and not self.process.waitForFinished(30000):
                self.process.kill()
        if self.scan is not None:
            self.scan.requestInterruption()
            self.scan.wait()
        self._save_settings()
        super().closeEvent(event)
