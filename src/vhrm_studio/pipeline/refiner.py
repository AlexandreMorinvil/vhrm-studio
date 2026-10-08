"""CIHP classes and the temporal Mahalanobis skin refinement (pear_deeplab_skin_uv_video.ipynb cells 6-7, unchanged)."""
import cv2
import numpy as np

CLASS_NAMES = [
    'Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'UpperClothes', 'Dress', 'Coat', 'Socks', 'Pants',
    'Torso-skin', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe',
]
COLORMAP = np.array([
    [0, 0, 0], [31, 119, 180], [44, 160, 44], [44, 127, 125], [52, 225, 143],
    [217, 222, 163], [254, 128, 37], [130, 162, 128], [121, 7, 166], [136, 183, 248],
    [85, 1, 76], [22, 23, 62], [159, 50, 15], [101, 93, 152], [252, 229, 92],
    [167, 173, 17], [218, 252, 252], [238, 126, 197], [116, 157, 140], [214, 220, 252],
], dtype=np.uint8)
SKIN_CLASS_IDS = np.array([10, 13, 14, 15, 16, 17], dtype=np.int64)  # Torso-skin, Face, arms, legs
SKIN_LUT_U8 = np.zeros((256, 1), dtype=np.uint8)
SKIN_LUT_U8[SKIN_CLASS_IDS, 0] = 255

# Skin-colour refinement: same values as deeplabv3p_semantic_segmentation.ipynb.
SKIN_COLOR_MAHALANOBIS_THRESHOLD = 3.5
SKIN_COLOR_COV_REGULARIZATION = 25.0
SKIN_COLOR_REFERENCE_DISTANCE_PX = 3.0
SKIN_COLOR_MIN_REFERENCE_PIXELS = 200
SKIN_COLOR_MAX_REFERENCE_SAMPLES = 20_000
SKIN_COLOR_EVAL_SCALE = 0.5
ENABLE_TEMPORAL_SKIN_COLOR_MODEL = True
SKIN_COLOR_TEMPORAL_ALPHA = 0.95
SKIN_COLOR_MODEL_EVERY = 10
SKIN_MASK_ERODE_PX = 1    # 3x3 erosion iterations before the UV lookup (1 crop px is ~4-5 frame px here)


class TemporalSkinRefiner:
    """MahalanobisSkinRefiner video path (refine_bgr_u8) from deeplabv3p_semantic_segmentation.ipynb.

    Same statistics, temporal smoothing and thresholds; the LAB conversion code is a parameter because the crops
    here are RGB. Removes colour outliers from the CIHP skin classes and never adds pixels.
    """

    def __init__(self, mahalanobis_threshold=SKIN_COLOR_MAHALANOBIS_THRESHOLD,
                 covariance_regularization=SKIN_COLOR_COV_REGULARIZATION,
                 reference_distance_px=SKIN_COLOR_REFERENCE_DISTANCE_PX,
                 min_reference_pixels=SKIN_COLOR_MIN_REFERENCE_PIXELS,
                 max_reference_samples=SKIN_COLOR_MAX_REFERENCE_SAMPLES,
                 temporal_enabled=ENABLE_TEMPORAL_SKIN_COLOR_MODEL,
                 temporal_alpha=SKIN_COLOR_TEMPORAL_ALPHA,
                 model_every=SKIN_COLOR_MODEL_EVERY,
                 eval_scale=SKIN_COLOR_EVAL_SCALE):
        self.mahalanobis_threshold = float(mahalanobis_threshold)
        self.covariance_regularization = float(covariance_regularization)
        self.reference_distance_px = float(reference_distance_px)
        self.min_reference_pixels = int(min_reference_pixels)
        self.max_reference_samples = int(max_reference_samples)
        self.temporal_enabled = bool(temporal_enabled)
        self.temporal_alpha = float(temporal_alpha)
        self.model_every = max(1, int(model_every))
        self.eval_scale = float(eval_scale)
        if not 0.0 < self.eval_scale <= 1.0:
            raise ValueError('eval_scale must be in (0, 1]')
        self.reset()

    def reset(self):
        """Discard temporal state (once per video)."""
        self.running_center_lab = None
        self.running_covariance_lab = None
        self.temporal_frame_index = 0
        self.model_fit_count = 0
        self.undersized_refits = 0   # refits skipped for < min_reference_pixels samples
        self._whitening_affine = None

    def _reference_samples(self, lab_roi, initial_roi):
        distance = cv2.distanceTransform(initial_roi.astype(np.uint8) * 255, cv2.DIST_L2, 3)
        reference = initial_roi & (distance >= self.reference_distance_px)
        if int(reference.sum()) < self.min_reference_pixels:
            reference = initial_roi.copy()
        samples = lab_roi[reference].astype(np.float32, copy=False)
        if len(samples) < self.min_reference_pixels:
            return reference, samples, distance
        if len(samples) > self.max_reference_samples:
            indices = np.linspace(0, len(samples) - 1, self.max_reference_samples, dtype=np.intp)
            samples = samples[indices]
        median = np.median(samples, axis=0)
        mad = np.median(np.abs(samples - median), axis=0)
        robust_scale = np.maximum(1.4826 * mad, 1.0)
        inliers = np.all(np.abs(samples - median) <= 3.5 * robust_scale, axis=1)
        if int(inliers.sum()) >= self.min_reference_pixels:
            samples = samples[inliers]
        return reference, samples, distance

    def _estimate_model(self, samples):
        center = np.median(samples, axis=0).astype(np.float32)
        centered = samples - center
        covariance = (centered.T @ centered) / max(len(samples) - 1, 1)
        covariance = np.asarray(covariance, dtype=np.float32)
        covariance = 0.5 * (covariance + covariance.T)
        covariance += np.eye(3, dtype=np.float32) * self.covariance_regularization
        return center, covariance

    def _active_model(self, center, covariance):
        if self.temporal_enabled:
            if self.running_center_lab is None:
                self.running_center_lab = center.copy()
                self.running_covariance_lab = covariance.copy()
            else:
                alpha = self.temporal_alpha ** self.model_every
                self.running_center_lab = alpha * self.running_center_lab + (1.0 - alpha) * center
                self.running_covariance_lab = alpha * self.running_covariance_lab + (1.0 - alpha) * covariance
                self.running_covariance_lab = 0.5 * (self.running_covariance_lab + self.running_covariance_lab.T)
            covariance = self.running_covariance_lab + np.eye(3, dtype=np.float32) * self.covariance_regularization
            center = self.running_center_lab
        return center, covariance

    def _should_refit(self, frame_index):
        return (not self.temporal_enabled or self.running_center_lab is None
                or frame_index % self.model_every == 0)

    @staticmethod
    def _make_whitening(center, covariance):
        try:
            inverse_covariance = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            inverse_covariance = np.linalg.pinv(covariance)
        whitening = np.linalg.cholesky(inverse_covariance).T.astype(np.float32, copy=False)
        return np.concatenate((whitening, (-whitening @ center).reshape(3, 1)), axis=1).astype(np.float32, copy=False)

    def refine_u8(self, image, class_ids, color_code=cv2.COLOR_RGB2LAB):
        """image (H, W, 3) uint8, class_ids (H, W) uint8 -> (H, W) uint8 0/255 refined skin mask."""
        skin = cv2.LUT(class_ids, SKIN_LUT_U8)
        x, y, width, height = cv2.boundingRect(skin)
        if width == 0 or height == 0:
            return skin

        skin_roi = skin[y:y + height, x:x + width]
        scale = self.eval_scale
        eval_width = max(1, int(round(width * scale)))
        eval_height = max(1, int(round(height * scale)))
        roi = image[y:y + height, x:x + width]
        eval_img = roi if scale == 1.0 else cv2.resize(roi, (eval_width, eval_height), interpolation=cv2.INTER_AREA)
        eval_lab = cv2.cvtColor(eval_img, color_code).astype(np.float32, copy=False)

        frame_index = self.temporal_frame_index
        self.temporal_frame_index += 1
        if self._should_refit(frame_index):
            eval_skin = skin_roi if scale == 1.0 else cv2.resize(
                skin_roi, (eval_width, eval_height), interpolation=cv2.INTER_NEAREST)
            _, samples, _ = self._reference_samples(eval_lab, eval_skin.astype(bool, copy=False))
            if len(samples) >= self.min_reference_pixels:
                center, covariance = self._estimate_model(samples)
                center, covariance = self._active_model(center, covariance)
                self.model_fit_count += 1
                self._whitening_affine = self._make_whitening(center, covariance)
            else:
                self.undersized_refits += 1

        if self._whitening_affine is None:
            return skin

        whitened = cv2.transform(eval_lab, self._whitening_affine)
        distances_sq = cv2.transform(cv2.multiply(whitened, whitened), np.ones((1, 3), dtype=np.float32))
        keep = cv2.compare(distances_sq, float(self.mahalanobis_threshold ** 2), cv2.CMP_LE)
        if scale != 1.0:
            keep = cv2.resize(keep, (width, height), interpolation=cv2.INTER_NEAREST)
        cv2.bitwise_and(skin_roi, keep, dst=skin_roi)
        return skin
