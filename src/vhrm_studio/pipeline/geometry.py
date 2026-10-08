"""Crops, projection, rasterization and UV texel geometry of the skin-UV pipeline.

From pear_deeplab_skin_uv_video.ipynb (cells 4 and 9) and PEAR's inference_images.py, with the same arithmetic;
the notebook's module globals (DEVICE, PROJ_MATRIX) are taken from the inputs / the backprojector instead.
"""
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

RENDER_SIZE = 1024
RASTER_BIN_SIZE = 32
FOCAL_LENGTH = 24.0
PERSON_MERGE_PX = 12       # DeepLab pixels bridged when joining the parts of one person (arms, hair, ...)
PERSON_MIN_PIXELS = 1000   # smaller blobs count as no person (-> YOLO re-detection)
_PERSON_MERGE_KERNEL = np.ones((2 * PERSON_MERGE_PX + 1,) * 2, dtype=np.uint8)


# ---------------------------------------------------------------- PEAR crop helpers (inference_images.py)
def sanitize_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w * h > 0 and x2 > x1 and y2 > y1:
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])
    else:
        bbox = None
    return bbox


def process_bbox(bbox, img_width, img_height, input_img_shape, ratio=1.25):
    bbox = sanitize_bbox(bbox, img_width, img_height)
    if bbox is None:
        return bbox
    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    aspect_ratio = input_img_shape[1] / input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * ratio
    bbox[3] = h * ratio
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.
    return bbox.astype(np.float32)


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)
    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans.astype(np.float32)


def get_proj_matrix(tanfov, device, z_near=0.01, z_far=100, z_sign=1.0):
    """PEAR utils/graphics_utils.get_proj_matrix."""
    tanHalfFovY = tanfov
    tanHalfFovX = tanfov
    top = tanHalfFovY * z_near
    bottom = -top
    right = tanHalfFovX * z_near
    left = -right
    z_sign = 1.0
    proj_matrix = torch.zeros(4, 4).float().to(device)
    proj_matrix[0, 0] = 2.0 * z_near / (right - left)
    proj_matrix[1, 1] = 2.0 * z_near / (top - bottom)
    proj_matrix[0, 2] = (right + left) / (right - left)
    proj_matrix[1, 2] = (top + bottom) / (top - bottom)
    proj_matrix[3, 2] = z_sign
    proj_matrix[2, 2] = z_sign * z_far / (z_far - z_near)
    proj_matrix[2, 3] = -(z_far * z_near) / (z_far - z_near)
    return proj_matrix


# ---------------------------------------------------------------- SMPL-X UV topology and atlas tables
class SmplxTopology(torch.nn.Module):
    """faces / uvfaces / raw_uvcoords of smplx_tex.obj, the same buffers as PEAR's body renderer (Renderer2)."""

    def __init__(self, smplx_assets_dir):
        super().__init__()
        from pytorch3d.io import load_obj
        _, faces, aux = load_obj(str(Path(smplx_assets_dir) / 'smplx_tex.obj'), load_textures=False)
        self.register_buffer('faces', faces.verts_idx[None, ...])
        self.register_buffer('uvfaces', faces.textures_idx[None, ...])
        self.register_buffer('raw_uvcoords', aux.verts_uvs[None, ...])


class PearUVBackprojector:
    """Static UV-atlas tables of PearUVBackprojector (pear_SMPLX_surface_mapping_fast.ipynb, __init__ unchanged)."""

    def __init__(self, body_renderer, device, texture_size=1024, raster_size=512, render_size=1024, crop_size=256,
                 depth_tol_frac=0.01, focal_length=FOCAL_LENGTH):
        self.device = device
        self.texture_size = texture_size
        self.raster_size = raster_size
        self.render_size = render_size
        self.crop_size = crop_size
        self.depth_tol_frac = depth_tol_frac
        self.front_sign = None  # learned from the first batch's z-buffer winners
        self.proj_matrix = get_proj_matrix(1.0 / focal_length, device)

        faces = body_renderer.faces
        uvfaces = body_renderer.uvfaces
        uvcoords = body_renderer.raw_uvcoords
        faces = (faces.squeeze(0) if faces.ndim == 3 else faces).long()
        uvfaces = (uvfaces.squeeze(0) if uvfaces.ndim == 3 else uvfaces).long()
        uvcoords = (uvcoords.squeeze(0) if uvcoords.ndim == 3 else uvcoords).float()
        self.faces = faces.to(device)                      # (F, 3)

        uv_np = uvcoords.cpu().numpy()
        uvf_np = uvfaces.cpu().numpy()
        uv_px = np.empty_like(uv_np, dtype=np.float32)
        uv_px[:, 0] = uv_np[:, 0] * (texture_size - 1)
        uv_px[:, 1] = (1.0 - uv_np[:, 1]) * (texture_size - 1)
        face_id_tex = np.full((texture_size, texture_size), -1, dtype=np.int32)
        for fid in range(uvf_np.shape[0]):
            tri = uv_px[uvf_np[fid]].astype(np.int32).reshape(-1, 1, 2)
            cv2.fillConvexPoly(face_id_tex, tri, int(fid))

        self.uv_body_mask = torch.from_numpy(face_id_tex >= 0).to(device)
        ty, tx = np.where(face_id_tex >= 0)
        tex_fids = face_id_tex[ty, tx].astype(np.int64)

        v0 = uv_px[uvf_np[tex_fids, 0]].astype(np.float64)
        v1 = uv_px[uvf_np[tex_fids, 1]].astype(np.float64)
        v2 = uv_px[uvf_np[tex_fids, 2]].astype(np.float64)
        p = np.stack([tx, ty], axis=1).astype(np.float64)
        e0, e1, ep = v1 - v0, v2 - v0, p - v0
        d00 = (e0 * e0).sum(1); d01 = (e0 * e1).sum(1); d11 = (e1 * e1).sum(1)  # noqa: E702
        d20 = (ep * e0).sum(1); d21 = (ep * e1).sum(1)  # noqa: E702
        det = d00 * d11 - d01 * d01
        det[np.abs(det) < 1e-10] = 1e-10
        b1 = (d11 * d20 - d01 * d21) / det
        b2 = (d00 * d21 - d01 * d20) / det
        b0 = 1.0 - b1 - b2
        bary = np.stack([b0, b1, b2], axis=1).astype(np.float32)

        self.tex_y = torch.from_numpy(ty.astype(np.int64)).to(device)
        self.tex_x = torch.from_numpy(tx.astype(np.int64)).to(device)
        self.tex_flat = self.tex_y * texture_size + self.tex_x       # texel order = np.flatnonzero(uv_body_mask)
        self.texel_fids = torch.from_numpy(tex_fids).to(device)      # (T,)
        self.texel_tri = self.faces[self.texel_fids]                  # (T, 3)
        self.texel_bary = torch.from_numpy(bary).to(device)           # (T, 3)
        self.texel_csr = None


def mesh_digest(faces, faces_uv, vt):
    """packed_uv/semantic_layout.py: identifies the mesh a layout was built for."""
    payload = faces.astype(np.int64).tobytes() + faces_uv.astype(np.int64).tobytes() + np.round(vt, 6).tobytes()
    return hashlib.sha1(payload).hexdigest()


def load_layout(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


class UVTexelLayout:
    """Texel tables of another UV layout of the renderer's mesh, saved by packed_uv/semantic_layout.py
    (texel order = np.flatnonzero(uv_body_mask), like PearUVBackprojector)."""

    def __init__(self, path, bp, renderer):
        self.path = Path(path)
        self.tables = load_layout(self.path)
        flat = lambda t: (t.squeeze(0) if t.ndim == 3 else t).cpu().numpy()  # noqa: E731
        digest = mesh_digest(flat(renderer.faces), flat(renderer.uvfaces), flat(renderer.raw_uvcoords).astype(np.float64))
        if digest != str(self.tables['mesh_digest']):
            raise ValueError(f'{self.path}: the layout was built for a different mesh than the renderer')
        mask = self.tables['uv_body_mask'].astype(bool)
        self.uv_body_mask = torch.from_numpy(mask).to(bp.device)
        self.tex_flat = torch.from_numpy(np.flatnonzero(mask)).to(bp.device)
        self.texel_fids = torch.from_numpy(self.tables['texel_fids'].astype(np.int64)).to(bp.device)
        self.texel_tri = bp.faces[self.texel_fids]
        self.texel_bary = torch.from_numpy(self.tables['texel_bary'].astype(np.float32)).to(bp.device)
        self.texel_csr = None


# ---------------------------------------------------------------- geometry helpers (exact FP32)
def pixel_grid(xy, height, width):
    """Pixel coordinates -> grid_sample coordinates for align_corners=True."""
    return torch.stack((xy[..., 0] * (2.0 / (width - 1)) - 1.0, xy[..., 1] * (2.0 / (height - 1)) - 1.0), dim=-1)


def apply_affine(xy, affine):
    """xy (..., 2) through one (2, 3) affine. Elementwise, so TF32 cannot perturb pixel coordinates."""
    return xy[..., :1] * affine[:, 0] + xy[..., 1:2] * affine[:, 1] + affine[:, 2]


def apply_affine_batch(xy, affines):
    """xy (n, T, 2) through per-frame (n, 2, 3) affines; same arithmetic as apply_affine."""
    return xy[..., :1] * affines[:, None, :, 0] + xy[..., 1:2] * affines[:, None, :, 1] + affines[:, None, :, 2]


_crop_pixel_grids = {}


def crop_batch(frames, inv_trans, size, rounding):
    """GPU affine crop (n, 3, size, size) of planar float frames at inv_trans @ [x, y, 1].

    rounding='floor' matches generate_patch_image + astype(uint8) (PEAR); 'round' matches cv2.warpAffine on uint8
    frames (segmentation notebook)."""
    device = frames.device
    if (size, device) not in _crop_pixel_grids:
        _crop_pixel_grids[size, device] = torch.stack(torch.meshgrid(
            torch.arange(size, device=device, dtype=torch.float32),
            torch.arange(size, device=device, dtype=torch.float32), indexing='xy'), dim=-1)
    affine = torch.as_tensor(np.asarray(inv_trans), dtype=torch.float32, device=device)
    grid = pixel_grid(apply_affine(_crop_pixel_grids[size, device], affine), *frames.shape[-2:])
    out = F.grid_sample(frames, grid[None].expand(frames.shape[0], -1, -1, -1), mode='bilinear',
                        padding_mode='zeros', align_corners=True)
    return out.floor_() if rounding == 'floor' else out.round_()


def project_batch(verts, rt, proj_matrix, render_size=RENDER_SIZE):
    """Same math as GS_Camera (exact FP32): view depth (n, V), NDC xy (n, V, 2), 1024-screen xy (n, V, 2)."""
    view = (rt[:, None, :3, :3] * verts[..., None, :]).sum(-1) + rt[:, None, :3, 3]
    view_h = torch.cat([view, torch.ones_like(view[..., :1])], dim=-1)
    clip = (proj_matrix * view_h[..., None, :]).sum(-1)
    ndc_xy = clip[..., :2] / (clip[..., 3:] + 1e-7)
    return view[..., 2], ndc_xy, (1.0 - ndc_xy) * (render_size / 2)


def faces_per_bin_max(v2d_raster, faces, image_size, bin_size):
    """Max number of face bounding boxes overlapping any coarse bin; PyTorch3D silently drops faces beyond
    max_faces_per_bin, so the limit is sized from this count on every batch."""
    n = v2d_raster.shape[0]
    nb = 1 + (image_size - 1) // bin_size
    fxy = v2d_raster[:, faces]
    lo, hi = fxy.amin(2), fxy.amax(2)
    inside = ((hi >= 0) & (lo < image_size)).all(-1).float()
    b0 = (lo / bin_size).floor().clamp(0, nb - 1).long()
    b1 = (hi / bin_size).floor().clamp(0, nb - 1).long() + 1
    stride = nb + 1
    base = (torch.arange(n, device=v2d_raster.device) * stride * stride)[:, None]
    diff = torch.zeros(n * stride * stride, device=v2d_raster.device)
    for y, x, sign in ((b0[..., 1], b0[..., 0], 1.0), (b0[..., 1], b1[..., 0], -1.0),
                       (b1[..., 1], b0[..., 0], -1.0), (b1[..., 1], b1[..., 0], 1.0)):
        diff.index_add_(0, (base + y * stride + x).reshape(-1), (inside * sign).reshape(-1))
    return diff.view(n, stride, stride).cumsum(1).cumsum(2)[:, :nb, :nb].amax()


def texel_interp_matrix(bp, n_verts):
    """Sparse (T, V) barycentric matrix: one SpMM gives texel screen x, y and depth for the whole batch."""
    if getattr(bp, 'texel_csr', None) is None or bp.texel_csr.shape[1] != n_verts:
        n_texels = bp.texel_tri.shape[0]
        rows = torch.arange(n_texels, device=bp.texel_tri.device).repeat_interleave(3)
        coo = torch.sparse_coo_tensor(torch.stack([rows, bp.texel_tri.reshape(-1)]), bp.texel_bary.reshape(-1),
                                      (n_texels, n_verts))
        bp.texel_csr = coo.coalesce().to_sparse_csr()
    return bp.texel_csr


def raster_batch(bp, verts, rt):
    """Per-frame mesh state of backproject_batch (pear_SMPLX_surface_mapping_fast.ipynb) shared by every UV layout:
    projected vertices, z-buffer, depth tolerance, front-facing and z-buffer-winning faces."""
    from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
    from pytorch3d.structures import Meshes
    n, n_verts = verts.shape[:2]
    rs = bp.raster_size
    n_faces = bp.faces.shape[0]
    z_view, ndc_xy, screen = project_batch(verts, rt, bp.proj_matrix, bp.render_size)
    v2d = screen * (rs / bp.render_size)

    max_faces = min(n_faces, int(faces_per_bin_max(v2d, bp.faces, rs, RASTER_BIN_SIZE)) + 512)
    meshes = Meshes(verts=torch.cat([ndc_xy, z_view[..., None]], dim=-1), faces=bp.faces[None].expand(n, -1, -1))
    pix_to_face, zbuf, _, _ = rasterize_meshes(
        meshes, image_size=rs, blur_radius=0.0, faces_per_pixel=1,
        bin_size=RASTER_BIN_SIZE, max_faces_per_bin=max_faces)
    pix_to_face, zbuf = pix_to_face[..., 0], zbuf[..., 0]  # packed ids: frame b owns [b*F, (b+1)*F)

    occupied = zbuf > 0
    zmin = zbuf.masked_fill(~occupied, torch.inf).amin(dim=(1, 2))
    zmax = zbuf.masked_fill(~occupied, -torch.inf).amax(dim=(1, 2))
    depth_tol = (bp.depth_tol_frac * (zmax - zmin).clamp_min(0)).clamp_min(1e-6)  # per frame

    winner = torch.zeros(n * n_faces + 1, dtype=torch.bool, device=verts.device)
    winner[pix_to_face.reshape(-1) + 1] = True
    winner = winner[1:].view(n, n_faces)
    fv = v2d[:, bp.faces]
    ea, eb = fv[:, :, 1] - fv[:, :, 0], fv[:, :, 2] - fv[:, :, 0]
    signed_area = ea[..., 0] * eb[..., 1] - ea[..., 1] * eb[..., 0]
    if bp.front_sign is None:
        bp.front_sign = 1.0 if float(torch.sign(signed_area[0][winner[0]]).sum()) >= 0 else -1.0
    return {'vert_vals': torch.cat([v2d, z_view[..., None]], dim=-1).permute(1, 0, 2).reshape(n_verts, -1),
            'zbuf_max': F.max_pool2d(zbuf[:, None], 3, stride=1, padding=1), 'depth_tol': depth_tol,
            'front': signed_area * bp.front_sign > 0, 'winner': winner, 'screen': screen}


def texel_geometry(bp, raster, layout, pear_crop_to_frame, frame_hw):
    """valid (n, T) bool (visible and inside the frame) and src_orig (n, T, 2) frame pixel coordinates of the
    texels of one layout (bp itself or a UVTexelLayout). pear_crop_to_frame: (n, 2, 3) float32 CUDA."""
    rs = bp.raster_size
    vert_vals = raster['vert_vals']
    n = vert_vals.shape[1] // 3
    texel_xyz = (texel_interp_matrix(layout, vert_vals.shape[0]) @ vert_vals).view(-1, n, 3).permute(1, 0, 2)
    src, tdepth = texel_xyz[..., :2], texel_xyz[..., 2]
    z_at = F.grid_sample(raster['zbuf_max'], pixel_grid(src, rs, rs)[:, :, None], mode='nearest',
                         align_corners=True)[:, 0, :, 0]
    visible = (z_at <= 0) | (tdepth <= z_at + raster['depth_tol'][:, None])
    valid = (visible & raster['front'][:, layout.texel_fids]) | raster['winner'][:, layout.texel_fids]

    src_orig = apply_affine_batch(src * (bp.crop_size / rs), pear_crop_to_frame)
    height, width = frame_hw
    valid &= ((src_orig[..., 0] >= 0) & (src_orig[..., 0] <= width - 1)
              & (src_orig[..., 1] >= 0) & (src_orig[..., 1] <= height - 1))
    return valid, src_orig


def raster_frames(raster, start, stop):
    """Frames start:stop of a raster_batch result."""
    return {'vert_vals': raster['vert_vals'][:, 3 * start:3 * stop].contiguous(),
            **{k: raster[k][start:stop] for k in ('zbuf_max', 'depth_tol', 'front', 'winner', 'screen')}}


def sample_frame_colors(frames, src_orig):
    """Bilinear RGB samples (n, T, 3) uint8 of planar float frames at src_orig."""
    height, width = frames.shape[-2:]
    colors = F.grid_sample(frames, pixel_grid(src_orig, height, width)[:, :, None], mode='bilinear',
                           align_corners=True)[..., 0]
    return colors.round_().clamp_(0, 255).to(torch.uint8).transpose(1, 2)


def lookup_nearest(maps, xy, size):
    """Nearest-pixel values of (n, size*size) maps at (n, T, 2) pixel coordinates, plus an inside-the-map mask."""
    ix, iy = xy[..., 0].round(), xy[..., 1].round()
    inside = (ix >= 0) & (ix <= size - 1) & (iy >= 0) & (iy <= size - 1)
    flat = (iy.clamp(0, size - 1) * size + ix.clamp(0, size - 1)).long()
    return torch.gather(maps, 1, flat), inside


def texels_to_atlas(values, layout):
    """(n, T, ...) per-texel values -> (n, H, W, ...) atlas of the layout, zeros outside the body texels."""
    n, (height, width) = values.shape[0], layout.uv_body_mask.shape
    out = values.new_zeros((n, height * width) + tuple(values.shape[2:]))
    out[:, layout.tex_flat] = values
    return out.view((n, height, width) + tuple(values.shape[2:]))


_PACK_WEIGHTS = {}


def pack_mask_bits(mask):
    """GPU equivalent of np.packbits(mask, axis=-1) for widths divisible by 8."""
    if mask.device not in _PACK_WEIGHTS:
        _PACK_WEIGHTS[mask.device] = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=mask.device)
    n, h, w = mask.shape
    bits = mask.view(n, h, w // 8, 8).to(torch.uint8) * _PACK_WEIGHTS[mask.device]
    return bits.sum(dim=-1, dtype=torch.int16).to(torch.uint8)


def bbox_from_class_ids(class_ids, crop_to_frame, img_w, img_h, pad):
    """Frame-space xywh of the largest person blob (non-background CIHP pixels) of one skin crop, or None."""
    person = (class_ids > 0).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(cv2.dilate(person, _PERSON_MERGE_KERNEL), connectivity=8)
    counts = np.bincount(labels[person > 0], minlength=n_labels)
    best = int(counts.argmax())
    if counts[best] < PERSON_MIN_PIXELS:
        return None
    ys, xs = np.nonzero((labels == best) & (person > 0))
    corners = np.array([[xs.min(), ys.min()], [xs.max(), ys.max()]], dtype=np.float32)
    lo, hi = np.sort(corners @ crop_to_frame[:, :2].T + crop_to_frame[:, 2], axis=0)
    lo, hi = lo - pad * (hi - lo), hi + pad * (hi - lo)
    limit = np.array([img_w - 1, img_h - 1], dtype=np.float32)
    lo, hi = np.clip(lo, 0, limit), np.clip(hi, 0, limit)
    return np.r_[lo, hi - lo].astype(np.float32)
