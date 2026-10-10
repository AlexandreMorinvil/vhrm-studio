"""Default settings of the skin-UV pipeline (notebook cell 12); no torch import, so the GUI can use them."""

# Temporal stabilization (notebook cell 12): tunable starting values, not a calibrated rPPG setting.
TEMPORAL_CONFIG = {
    'bbox_enabled': True, 'bbox_alpha': 0.0,
    'bbox_deadband': 0.03,          # a still person keeps a still crop (fraction of the box size / log-size)
    'shape_enabled': True, 'shape_alpha': 0.99, 'shape_reference_fps': 30.0, 'shape_warmup_frames': 0,
    'camera_enabled': True, 'pose_enabled': True,
    'camera_d_cutoff': 1.0, 'pose_d_cutoff': 1.0,
    'diagnostics': False,
    'profiles': {
        'camera': (1.5, 0.15), 'global': (1.5, 0.25), 'body': (1.5, 0.30), 'hand': (1.0, 0.30),
        'head': (2.0, 0.30), 'face': (3.0, 0.30), 'expression': (5.0, 0.05), 'eye': (8.0, 0.30),
        'eyelid': (8.0, 0.05),
    },
}

# Per-video settings (notebook cell 12 defaults).
RUN_DEFAULTS = {
    'max_frames': None,             # processed-frame cap for tests; None = whole video
    'batch_size': 8,                # frames per GPU batch (also the DeepLab batch); bbox updated once per batch
    'target_fps': 30.0,             # frame selection target
    'image_fps': 30.0,              # image sequences whose file names hold no time stamps
    'overlap_deeplab': True,        # DeepLab + refinement on a worker thread / own stream, concurrent with PEAR
    'io_binding': True,             # keep DeepLab input/output on the GPU
    'accumulate': True,             # running skin texture (visible-and-skin texels), saved once per video
    'packed_uv': True,              # also store every frame's atlases in the packed semantic layout (packed_uv_*)
    'packed_uv_outputs': ['skin_rgb'],
    'texel_chunk_frames': 2,        # frames per per-texel pass: bounds its GPU memory (VRAM spill is far slower)
    'gzip_level': 1,
    'compress_threads': 4,
    'person_bbox_pad': 0.02,        # per side, fraction of the person's visible extent
    'save_accumulated_png': False,  # <stem>_accumulated_skin_uv.png next to the HDF5 file
    'profile_stages': False,        # synchronizes at stage boundaries; lowers throughput
    'temporal': TEMPORAL_CONFIG,
}
