"""SMPL-X mesh rebuilt from the PEAR parameters stored in a skin-UV HDF5 file (used by the viewer).

Projection and rasterization follow the pipeline (project_batch / uv_geometry_batch). Needs torch, PyTorch3D and the
SMPL-X / FLAME assets (see vhrm_studio.assets); the PEAR model code is bundled (vhrm_studio.third_party.pear).
Without a working rasterizer only vertices / projections are available.
"""
import math
import sys
import threading
from pathlib import Path

import numpy as np

RASTER_BIN_SIZE = 32


def faces_per_bin_max(v2d_raster, faces, image_size, bin_size):
    """Max number of face bounding boxes overlapping any coarse bin (from pear_deeplab_skin_uv_video.ipynb).
    PyTorch3D silently drops faces beyond max_faces_per_bin, and its cost grows with that limit."""
    import torch
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


class SmplxMesh:
    """EHM_v2 (SMPL-X body + FLAME head) on one row of stored parameters, plus a shaded raster in PEAR render space."""

    def __init__(self, assets_dir, device=None):
        try:
            import torch
            from pytorch3d.structures import Meshes
        except ImportError as exc:
            raise RuntimeError(
                'Mesh views require torch and PyTorch3D (also needed to build the body model). '
                'requirements.txt does not install them: install the "pipeline" requirements (see README). '
                f'Python: {sys.executable}. Import error: {exc}'
            ) from exc
        from .assets import ASSETS_HELP, missing_assets
        from .third_party.pear import build_ehm

        missing = missing_assets(assets_dir)
        if missing:
            raise RuntimeError(f'SMPL-X / FLAME assets incomplete in {assets_dir} (missing {missing[0]}, ...). '
                               + ASSETS_HELP)
        self.torch = torch
        assets = Path(assets_dir).resolve()
        ehm = build_ehm(assets)
        self.device = torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.ehm = ehm.to(self.device)
        self.ehm.smplx.smplx2flame_ind = torch.as_tensor(np.asarray(ehm.smplx.smplx2flame_ind), device=self.device)
        self.faces = ehm.smplx.faces_tensor.to(self.device).long()
        self.faces_np = self.faces.cpu().numpy().astype(np.int32)
        from pytorch3d.io import load_obj
        _, topology, atlas = load_obj(str(assets / 'SMPLX' / 'smplx_tex.obj'), load_textures=False)
        self.uv_face_count = len(topology.verts_idx)
        if not np.array_equal(topology.verts_idx.numpy(), self.faces_np[:self.uv_face_count]):
            raise RuntimeError('SMPL-X UV topology does not match the EHM mesh faces')
        self.face_uv = atlas.verts_uvs[topology.textures_idx].to(self.device)
        try:
            from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
            self._raster = (rasterize_meshes, Meshes)
        except ImportError:
            self._raster = None
        self.lock = threading.Lock()

    @property
    def can_rasterize(self):
        return self._raster is not None

    def compute(self, body, flame, rt, focal_length=24.0, render_size=1024, raster_size=1024):
        """body / flame: dicts of per-row arrays (no batch axis); rt: (4, 4) pd_cam. See compute_batch."""
        return self.compute_batch([body], [flame], [rt], focal_length, render_size, raster_size)[0]

    def compute_batch(self, bodies, flames, rts, focal_length=24.0, render_size=1024, raster_size=1024):
        """Several rows at once (one EHM call). Per row: verts (V, 3) model space, view (V, 3) camera space
        (x left, y up, z away from the camera), screen (V, 2) render-space pixels and, with PyTorch3D,
        shade / coverage (raster_size, raster_size) float32.
        Raster pixel (i, j) covers render-space pixel ((j + 0.5), (i + 0.5)) * render_size / raster_size.
        """
        torch = self.torch
        with self.lock, torch.inference_mode():
            dev = self.device
            stack = lambda dicts: {k: torch.as_tensor(np.stack([np.asarray(d[k], dtype=np.float32) for d in dicts]),
                                                      device=dev) for k in dicts[0]}
            verts = self.ehm(stack(bodies), stack(flames), pose_type='aa')['vertices'].float()
            rt = torch.as_tensor(np.stack(rts).astype(np.float32), device=dev)
            view = torch.einsum('nij,nvj->nvi', rt[:, :3, :3], verts) + rt[:, None, :3, 3]
            ndc_xy = view[..., :2] * float(focal_length) / (view[..., 2:] + 1e-7)
            screen = (1.0 - ndc_xy) * (render_size / 2)
            n = verts.shape[0]
            verts_np, view_np, screen_np = verts.cpu().numpy(), view.cpu().numpy(), screen.cpu().numpy()
            outs = [{'verts': verts_np[i], 'view': view_np[i], 'screen': screen_np[i], 'faces': self.faces_np}
                    for i in range(n)]
            if self._raster is not None:
                shade, coverage = self._shade_raster(ndc_xy, view[..., 2], view, int(raster_size))
                shade, coverage = shade.cpu().numpy(), coverage.cpu().numpy()
                for i in range(n):
                    outs[i]['shade'], outs[i]['coverage'] = shade[i], coverage[i]
        return outs

    def compute_standard(self, body, flame):
        """Rest pose with the row's identity and scales, but no recorded pose or expression."""
        torch = self.torch
        with self.lock, torch.inference_mode():
            body = {key: torch.as_tensor(np.asarray(value, dtype=np.float32), device=self.device)[None].clone()
                    for key, value in body.items()}
            flame = {key: torch.as_tensor(np.asarray(value, dtype=np.float32), device=self.device)[None].clone()
                     for key, value in flame.items()}
            for key in ('global_pose', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'eye_pose'):
                if key in body:
                    value = body[key]
                    body[key] = (torch.eye(3, device=self.device).expand_as(value).clone()
                                 if value.ndim >= 4 and value.shape[-2:] == (3, 3) else torch.zeros_like(value))
            for key in ('exp', 'joints_offset'):
                if key in body:
                    body[key] = torch.zeros_like(body[key])
            for key in ('pose_params', 'jaw_params', 'eye_pose_params', 'expression_params', 'eyelid_params'):
                if key in flame:
                    flame[key] = torch.zeros_like(flame[key])
            verts = self.ehm(body, flame, pose_type='aa')['vertices'][0].float()
            verts = verts * verts.new_tensor([-1.0, 1.0, -1.0])
            return verts.cpu().numpy()

    def _shade_raster(self, ndc_xy, depth, eye_space, image_size, texture=None):
        """Rasterize n meshes (ndc_xy (n, V, 2) in this module's convention, depth (n, V)) and shade each face by
        |cos| between its normal and the viewing ray (eye_space (n, V, 3), camera at the origin).
        Returns shade (in [0.25, 1], 0 where empty) and coverage, both (n, image_size, image_size)."""
        torch = self.torch
        rasterize_meshes, Meshes = self._raster
        n, n_faces = ndc_xy.shape[0], int(self.faces.shape[0])
        bin_size = RASTER_BIN_SIZE
        while -(-image_size // bin_size) >= 22:  # PyTorch3D limit on bins per side
            bin_size *= 2
        v2d = (1.0 - ndc_xy) * (image_size / 2)
        max_faces = min(n_faces, int(faces_per_bin_max(v2d, self.faces, image_size, bin_size)) + 512)
        meshes = Meshes(verts=list(torch.cat([ndc_xy, depth[..., None]], dim=-1)), faces=[self.faces] * n)
        pix_to_face, _, bary, _ = rasterize_meshes(meshes, image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
                                                bin_size=bin_size, max_faces_per_bin=max_faces)
        p2f = pix_to_face[..., 0]  # packed ids: mesh b owns [b * F, (b + 1) * F)
        fv = eye_space[:, self.faces]
        normal = torch.cross(fv[:, :, 1] - fv[:, :, 0], fv[:, :, 2] - fv[:, :, 0], dim=-1)
        ray = fv.mean(dim=2)
        cos = (normal * ray).sum(-1).abs() / (normal.norm(dim=-1) * ray.norm(dim=-1)).clamp_min(1e-12)
        face_shade = torch.cat([(0.25 + 0.75 * cos).reshape(-1), cos.new_zeros(1)])
        shade = face_shade[p2f]
        if texture is not None:
            import torch.nn.functional as functional
            local_faces = p2f.clamp_min(0) % n_faces
            uv = (self.face_uv[local_faces.clamp_max(self.uv_face_count - 1)] * bary[..., 0, :, None]).sum(-2)
            grid = torch.stack((uv[..., 0] * 2 - 1, 1 - uv[..., 1] * 2), dim=-1)
            image = torch.as_tensor(np.ascontiguousarray(texture), dtype=torch.float32, device=self.device)
            image = image.permute(2, 0, 1)[None].expand(n, -1, -1, -1)
            colors = functional.grid_sample(image, grid, mode='nearest', align_corners=True).permute(0, 2, 3, 1)
            colors = torch.where((local_faces < self.uv_face_count)[..., None], colors, 110.0)
            shade = colors * shade[..., None]
        return shade, (p2f >= 0).float()

    def render_orbit(self, view_verts, yaw_deg=0.0, pitch_deg=0.0, zoom=1.0, size=640,
                     color=(219, 184, 158), background=(22, 22, 26), texture=None, pan=(0.0, 0.0)):
        """(size, size, 3) uint8 image of a camera-space mesh seen from an orbit camera around its bounding-box
        centre. yaw = pitch = 0 looks along the original camera axis (same left / up as the video)."""
        if self._raster is None:
            raise RuntimeError('PyTorch3D is needed for the 3D view')
        torch = self.torch
        with self.lock, torch.inference_mode():
            v = torch.as_tensor(np.asarray(view_verts, dtype=np.float32), device=self.device)
            lo, hi = v.amin(0), v.amax(0)
            extent = float((hi - lo).max())
            y, p = math.radians(yaw_deg), math.radians(pitch_deg)
            rot_y = torch.tensor([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
            rot_x = torch.tensor([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
            q = (v - (lo + hi) / 2) @ (rot_x @ rot_y).T.to(v)
            distance = 3.0 * extent
            q[:, 2] += distance
            ndc = q[:, :2] * (1.6 * distance / extent * zoom) / q[:, 2:]
            ndc -= ndc.new_tensor(pan) * 2.0
            shade, coverage = self._shade_raster(ndc[None], q[None, :, 2], q[None], int(size), texture=texture)
            bg = torch.tensor(background, dtype=torch.float32, device=self.device)
            fg = torch.tensor(color, dtype=torch.float32, device=self.device)
            surface = fg * shade[0, ..., None] if texture is None else shade[0]
            img = bg + (surface - bg) * coverage[0, ..., None]
            return img.round().clamp(0, 255).to(torch.uint8).cpu().numpy()
