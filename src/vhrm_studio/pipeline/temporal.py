"""One Euro / EMA temporal stabilization of the PEAR parameters (temporal_stabilization.py of the skin-UV pipeline),
with two additions:

* the camera translation is filtered in a fixed reference camera instead of the crop camera. PEAR predicts it
  relative to the crop, so every crop move is a step of the raw translation that exactly compensates the move; a
  low-pass filter on crop-relative values lags behind that step and drags the mesh along with the crop (~70% of the
  move, decaying over ~6 frames). Re-expressed in a crop-independent camera, a crop move is no motion at all.
* the bbox filter has a dead band, so that a still person keeps a still crop (the crop is re-estimated from the
  segmentation of the crop itself, a feedback loop that can otherwise oscillate between two crops)."""
import math

import numpy as np
import torch
from pytorch3d.transforms import (
    axis_angle_to_quaternion,
    matrix_to_quaternion,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
)


class OneEuroFilterTensor:
    """Device-resident One Euro filter; quaternion mode uses angular speed and shortest-path nlerp."""

    def __init__(self, min_cutoff, beta, d_cutoff=1.0, quaternion=False):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.quaternion = quaternion
        self.reset()

    def reset(self):
        self.previous_raw = None
        self.filtered = None
        self.derivative = None

    @staticmethod
    def _alpha(cutoff, dt):
        return 1.0 - 1.0 / (1.0 + 2.0 * math.pi * cutoff * dt)

    def update(self, value, dt):
        if self.filtered is None:
            self.previous_raw = value.clone()
            self.filtered = value.clone()
            self.derivative = torch.zeros_like(value[..., :1] if self.quaternion else value)
            return self.filtered
        dt = max(float(dt), 1e-6)
        if self.quaternion:
            valid = torch.isfinite(value).all(dim=-1, keepdim=True)
            value = torch.where(valid, value, self.previous_raw)
            value = torch.where((value * self.previous_raw).sum(-1, keepdim=True) < 0, -value, value)
            speed = 4.0 * torch.atan2(
                torch.linalg.vector_norm(value - self.previous_raw, dim=-1, keepdim=True),
                torch.linalg.vector_norm(value + self.previous_raw, dim=-1, keepdim=True),
            ) / dt
        else:
            value = torch.where(torch.isfinite(value), value, self.previous_raw)
            speed = (value - self.previous_raw) / dt
        derivative_alpha = self._alpha(self.d_cutoff, dt)
        self.derivative.lerp_(speed, derivative_alpha)
        alpha = self._alpha(self.min_cutoff + self.beta * self.derivative.abs(), dt)
        if self.quaternion:
            value = torch.where((value * self.filtered).sum(-1, keepdim=True) < 0, -value, value)
            self.filtered.lerp_(value, alpha)
            self.filtered.div_(torch.linalg.vector_norm(self.filtered, dim=-1, keepdim=True).clamp_min(1e-12))
        else:
            self.filtered.lerp_(value, alpha)
        self.previous_raw.copy_(value)
        return self.filtered


class BBoxTemporalFilter:
    """EMA of center and log-size on the existing four-value CPU bbox path, without extra padding.

    deadband: the box stays put while the center moves by less than this fraction of the box size and the log-size
    by less than this value; larger changes are filtered as usual (0 disables the dead band)."""

    def __init__(self, enabled=True, alpha=0.65, nominal_interval=8 / 30, deadband=0.0):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError('bbox alpha must be in [0, 1]')
        if deadband < 0:
            raise ValueError('bbox deadband must be nonnegative')
        self.enabled = enabled
        self.alpha = alpha
        self.nominal_interval = nominal_interval
        self.deadband = deadband
        self.reset()

    def reset(self):
        self.state = None
        self.previous_timestamp = None
        self.diagnostic = np.full((2, 4), np.nan, dtype=np.float32)

    def update(self, bbox, width, height, timestamp):
        raw = None
        if bbox is not None:
            candidate = np.asarray(bbox, dtype=np.float64)
            if candidate.shape == (4,) and np.isfinite(candidate).all() and (candidate[2:] > 1).all():
                raw = np.r_[candidate[:2] + candidate[2:] * 0.5, np.log(candidate[2:])]
        if raw is not None:
            if self.state is None or not self.enabled:
                self.state = raw.copy()
            elif (np.abs(raw[:2] - self.state[:2]) < self.deadband * np.exp(self.state[2:])).all() and \
                    (np.abs(raw[2:] - self.state[2:]) < self.deadband).all():
                pass
            else:
                dt = max(float(timestamp) - self.previous_timestamp, 1e-6)
                retention = self.alpha ** (dt / self.nominal_interval)
                self.state += (1.0 - retention) * (raw - self.state)
            self.previous_timestamp = float(timestamp)
        if self.state is None:
            return None
        size = np.clip(np.exp(self.state[2:]), 2.0, [width - 1, height - 1])
        center = np.clip(self.state[:2], size * 0.5, np.array([width - 1, height - 1]) - size * 0.5)
        self.state = np.r_[center, np.log(size)]
        self.diagnostic = np.stack([raw if raw is not None else np.full(4, np.nan), self.state]).astype(np.float32)
        return np.r_[center - size * 0.5, size].astype(np.float32)


class TemporalParameterStabilizer:
    """Filter PEAR's exact parameter schema, packing channels to avoid a filter/kernel loop per joint."""

    STATIC_KEYS = (('body_param', 'shape'), ('body_param', 'head_scale'),
                   ('body_param', 'hand_scale'), ('flame_param', 'shape_params'))
    ROTATION_KEYS = (
        ('body_param', 'global_pose', 'matrix', 'global'),
        ('body_param', 'body_pose', 'matrix', 'body'),
        ('body_param', 'left_hand_pose', 'matrix', 'hand'),
        ('body_param', 'right_hand_pose', 'matrix', 'hand'),
        ('flame_param', 'pose_params', 'axis_angle', 'head'),
        ('flame_param', 'jaw_params', 'axis_angle', 'face'),
        ('flame_param', 'eye_pose_params', 'axis_angle', 'eye'),
    )
    EXPRESSION_KEYS = (('body_param', 'exp', 'expression'),
                       ('flame_param', 'expression_params', 'expression'),
                       ('flame_param', 'eyelid_params', 'eyelid'))

    def __init__(self, config, fps, batch_size):
        if fps <= 0 or batch_size < 1:
            raise ValueError('fps and batch_size must be positive')
        if not 0.0 <= config['shape_alpha'] <= 1.0 or config['shape_warmup_frames'] < 0:
            raise ValueError('shape alpha must be in [0, 1] and warmup frames must be nonnegative')
        if min(config['shape_reference_fps'], config['camera_d_cutoff'], config['pose_d_cutoff']) <= 0:
            raise ValueError('reference fps and derivative cutoffs must be positive')
        if any(minimum <= 0 or adaptive < 0 for minimum, adaptive in config['profiles'].values()):
            raise ValueError('minimum cutoffs must be positive and beta must be nonnegative')
        self.config = dict(config)
        self.fps = float(fps)
        self.bbox_filter = BBoxTemporalFilter(config['bbox_enabled'], config['bbox_alpha'], batch_size / fps,
                                              config.get('bbox_deadband', 0.0))
        self.reset_temporal_state()

    def reset_temporal_state(self):
        self.bbox_filter.reset()
        self.previous_timestamp = None
        self.linear_filter = self.rotation_filter = None
        self.camera_reference = None
        self.shape_estimate = None
        self.shape_count = 0
        self.previous_diagnostics = {}
        self.previous_projected = None

    def stabilize_bbox(self, bbox, width, height, timestamp):
        return self.bbox_filter.update(bbox, width, height, timestamp)

    @staticmethod
    def crop_camera(crop_to_frame, crop_size, focal_length, device):
        """Frame-pixel principal point and focal lengths (2,) of PEAR's camera for an axis-aligned crop whose pixels
        map to frame pixels by crop_to_frame (2, 3): NDC xy projects to crop pixel (1 - ndc) * crop_size / 2."""
        affine = torch.as_tensor(np.asarray(crop_to_frame, dtype=np.float64)[:, :2], dtype=torch.float32,
                                 device=device)
        shift = torch.as_tensor(np.asarray(crop_to_frame, dtype=np.float64)[:, 2], dtype=torch.float32, device=device)
        half = 0.5 * crop_size
        return affine.sum(1) * half + shift, affine.diagonal() * (focal_length * half)

    @staticmethod
    def _change_camera(translation, source, target):
        """Translations (n, 3) of the camera `source` re-expressed in the camera `target` (both (center, focal)):
        the model origin keeps its frame-pixel projection and its projected size."""
        (source_center, source_focal), (target_center, target_focal) = source, target
        pixel = source_center - source_focal * translation[:, :2] / translation[:, 2:]
        depth = translation[:, 2:] * (target_focal.mean() / source_focal.mean())
        return torch.cat([depth * (target_center - pixel) / target_focal, depth], dim=1)

    def _coefficients(self, values, specs):
        cutoff, beta, derivative_cutoff = [], [], []
        for value, profile in zip(values, specs):
            minimum, adaptive = self.config['profiles'][profile]
            cutoff.append(torch.full_like(value[0, ..., :1] if profile in ('global', 'body', 'hand', 'head', 'face', 'eye')
                                          else value[0], minimum))
            beta.append(torch.full_like(cutoff[-1], adaptive))
            derivative_cutoff.append(torch.full_like(cutoff[-1], self.config[
                'camera_d_cutoff' if profile == 'camera' else 'pose_d_cutoff']))
        return torch.cat(cutoff), torch.cat(beta), torch.cat(derivative_cutoff)

    def _diagnose(self, name, raw, filtered, rotation=False):
        previous = self.previous_diagnostics.get(name)
        changes = []
        for label, value, old in zip(('raw', 'filtered'), (raw, filtered), previous or (None, None)):
            lagged = torch.cat([value[:1] if old is None else old[None], value[:-1]])
            if rotation:
                aligned = torch.where((value * lagged).sum(-1, keepdim=True) < 0, -value, value)
                delta = 4 * torch.atan2(torch.linalg.vector_norm(aligned - lagged, dim=-1),
                                       torch.linalg.vector_norm(aligned + lagged, dim=-1))
                change = delta.square().mean(-1).sqrt()
            else:
                change = (value - lagged).square().mean(-1).sqrt()
            changes.append((f'diagnostics/{name}_{label}_change', change))
        self.previous_diagnostics[name] = (raw[-1].clone(), filtered[-1].clone())
        return dict(changes)

    def stabilize_model_outputs(self, outputs, timestamps, crop_camera=None):
        """crop_camera: (center, focal) of the batch's crop (see crop_camera()). When given, the camera translation
        is filtered in the camera of the first crop since the last reset, so that crop moves are not filtered as
        motion; without it, the translation is filtered relative to the crop."""
        config = self.config
        result = {**outputs, 'body_param': dict(outputs['body_param']), 'flame_param': dict(outputs['flame_param'])}
        camera = outputs['pd_cam']
        count = camera.shape[0]
        if len(timestamps) != count:
            raise ValueError('one source timestamp is required per prediction')
        if camera.shape[1:] != (4, 4):
            raise ValueError('PEAR pd_cam must be (N, 4, 4), with fixed rotation and predicted translation')
        diagnostics = config['diagnostics']
        static_specs, static_values = [], []
        rotation_specs, rotation_values = [], []
        linear_specs, linear_values, profiles = [], [], []
        if config['shape_enabled']:
            for group, key in self.STATIC_KEYS:
                value = outputs[group].get(key)
                if value is not None:
                    static_specs.append((group, key, value.shape, value[0].numel()))
                    static_values.append(value.reshape(count, -1))
        if config['camera_enabled']:
            translation = camera[:, :3, 3]
            if crop_camera is not None:
                if self.camera_reference is None:
                    self.camera_reference = crop_camera
                translation = self._change_camera(translation, crop_camera, self.camera_reference)
            linear_specs.append(('pd_cam', None, (count, 3), 3))
            linear_values.append(translation)
            profiles.append('camera')
        if config['pose_enabled'] or diagnostics:
            for group, key, representation, profile in self.ROTATION_KEYS:
                value = outputs[group].get(key)
                if value is not None:
                    quaternion = (matrix_to_quaternion(value.reshape(count, -1, 3, 3)) if representation == 'matrix'
                                  else axis_angle_to_quaternion(value.reshape(count, -1, 3)))
                    rotation_specs.append((group, key, representation, value.shape, quaternion.shape[1], profile))
                    rotation_values.append(quaternion)
        if config['pose_enabled']:
            for group, key, profile in self.EXPRESSION_KEYS:
                value = outputs[group].get(key)
                if value is not None:
                    linear_specs.append((group, key, value.shape, value[0].numel()))
                    linear_values.append(value.reshape(count, -1))
                    profiles.append(profile)
        raw_static = torch.cat(static_values, dim=1) if static_values else None
        raw_linear = torch.cat(linear_values, dim=1) if linear_values else None
        raw_rotation = torch.cat(rotation_values, dim=1) if rotation_values else None
        filtered_static = torch.empty_like(raw_static) if raw_static is not None else None
        filtered_linear = torch.empty_like(raw_linear) if raw_linear is not None else None
        filtered_rotation = (torch.empty_like(raw_rotation) if config['pose_enabled'] and raw_rotation is not None
                             else raw_rotation)
        if raw_linear is not None and self.linear_filter is None:
            minimum, beta, derivative_cutoff = self._coefficients(linear_values, profiles)
            self.linear_filter = OneEuroFilterTensor(minimum, beta, derivative_cutoff)
        if config['pose_enabled'] and raw_rotation is not None and self.rotation_filter is None:
            minimum, beta, derivative_cutoff = self._coefficients(rotation_values, [spec[-1] for spec in rotation_specs])
            self.rotation_filter = OneEuroFilterTensor(minimum, beta, derivative_cutoff, quaternion=True)
        for index, timestamp in enumerate(timestamps):
            timestamp = float(timestamp)
            dt = 1.0 / self.fps if self.previous_timestamp is None else timestamp - self.previous_timestamp
            if dt < 0:
                raise ValueError('source timestamps must be nondecreasing')
            dt = max(dt, 1e-6)
            self.previous_timestamp = timestamp
            if raw_static is not None:
                value = raw_static[index]
                if self.shape_estimate is None:
                    self.shape_estimate = value.clone()
                else:
                    value = torch.where(torch.isfinite(value), value, self.shape_estimate)
                    warmup = config['shape_warmup_frames']
                    retention = (self.shape_count / (self.shape_count + 1) if self.shape_count < warmup else
                                 config['shape_alpha'] ** (dt * config['shape_reference_fps']))
                    self.shape_estimate.lerp_(value, 1.0 - retention)
                self.shape_count += 1
                filtered_static[index].copy_(self.shape_estimate)
            if raw_linear is not None:
                filtered_linear[index].copy_(self.linear_filter.update(raw_linear[index], dt))
            if config['pose_enabled'] and raw_rotation is not None:
                filtered_rotation[index].copy_(self.rotation_filter.update(raw_rotation[index], dt))
        for specs, packed in ((static_specs, filtered_static), (linear_specs, filtered_linear)):
            offset = 0
            for group, key, shape, length in specs:
                value = packed[:, offset:offset + length].reshape(shape)
                if group == 'pd_cam':
                    result['pd_cam'] = camera.clone()
                    result['pd_cam'][:, :3, 3] = (value if crop_camera is None else
                                                  self._change_camera(value, self.camera_reference, crop_camera))
                else:
                    result[group][key] = value
                offset += length
        extra = {}
        offset = 0
        for group, key, representation, shape, length, profile in rotation_specs:
            quaternion = filtered_rotation[:, offset:offset + length]
            if config['pose_enabled']:
                value = quaternion_to_matrix(quaternion) if representation == 'matrix' else quaternion_to_axis_angle(quaternion)
                result[group][key] = value.reshape(shape)
            if diagnostics:
                extra.update(self._diagnose(f'{group}_{key}', raw_rotation[:, offset:offset + length], quaternion, True))
            offset += length
        if diagnostics:
            extra['diagnostics/camera_raw'] = camera[:, :3, 3]
            extra['diagnostics/camera_filtered'] = result['pd_cam'][:, :3, 3]
            for group, key in self.STATIC_KEYS + tuple((group, key) for group, key, _ in self.EXPRESSION_KEYS):
                raw = outputs[group].get(key)
                if raw is not None:
                    filtered = result[group][key]
                    extra.update(self._diagnose(f'{group}_{key}', raw.reshape(count, -1), filtered.reshape(count, -1)))
                    if (group, key) in self.STATIC_KEYS:
                        extra[f'diagnostics/{group}_{key}_raw'] = raw
                        extra[f'diagnostics/{group}_{key}_filtered'] = filtered
        return result, extra

    def diagnose_geometry(self, frame_vertices, rows, batch_index):
        """Observational only: median corresponding-vertex displacement, in original-frame pixels."""
        lagged = torch.cat([frame_vertices[:1] if self.previous_projected is None else self.previous_projected[None],
                            frame_vertices[:-1]])
        displacement = torch.linalg.vector_norm(frame_vertices - lagged, dim=-1).median(dim=-1).values
        if self.previous_projected is None:
            displacement[0] = torch.nan
        self.previous_projected = frame_vertices[-1].clone()
        count = len(rows)
        boundary = np.zeros(count, dtype=np.uint8)
        boundary[0] = 1
        return {
            'diagnostics/projected_vertex_median_displacement_px': displacement,
            'diagnostics/batch_start': boundary,
            'diagnostics/batch_index': np.full(count, batch_index, dtype=np.int64),
            'diagnostics/bbox_raw_center_log_size': np.broadcast_to(self.bbox_filter.diagnostic[0], (count, 4)),
            'diagnostics/bbox_filtered_center_log_size': np.broadcast_to(self.bbox_filter.diagnostic[1], (count, 4)),
        }