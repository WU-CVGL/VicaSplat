from typing import Dict, Literal, Tuple
import time

import numpy as np
import torch
from torch import Tensor
import viser
import viser.transforms as vtf
import nerfview
from nerfview.viewer import Viewer
from plyfile import PlyData
from gsplat.rendering import rasterization

# from datasets.colmap import Dataset


class PoseViewer(Viewer):

    def init_scene(
        self, 
        pil_images,
        c2ws,
        fov_deg=50,
    ):
        self.camera_handles: Dict[int, viser.CameraFrustumHandle] = {}
        self.original_c2w: Dict[int, np.ndarray] = {}

        total_num = len(pil_images)
        # NOTE: not constraining the maximum number of camera frustums shown
        image_indices = np.linspace(0, total_num - 1, total_num, dtype=np.int32).tolist()
        for idx in image_indices:
            image_uint8 = np.asarray(pil_images[idx].resize((100, 100)))
            R = vtf.SO3.from_matrix(c2ws[idx][:3, :3])
            # NOTE: not understand why this is needed in nerfstudio viewer, but comment it out make ours work
            # probably because gsplat uses OpenCV convention, whereas nerfstudio use the Blender / OpenGL convention
            # R = R @ vtf.SO3.from_x_radians(np.pi)
 
            camera_handle = self.server.add_camera_frustum(
                name=f"/cameras/camera_{idx:05d}",
                fov=np.deg2rad(fov_deg),
                scale=0.05,  # hardcode this scale for now
                aspect=1,
                image=image_uint8,
                wxyz=R.wxyz,
                position=c2ws[idx][:3, 3],
                # NOTE: not multiplied by VISER_NERFSTUDIO_SCALE_RATIO, this should also be used in get_camera_state
            )

            @camera_handle.on_click
            def _(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
                with event.client.atomic():
                    event.client.camera.position = event.target.position
                    event.client.camera.wxyz = event.target.wxyz

            self.camera_handles[idx] = camera_handle
            self.original_c2w[idx] = c2ws[idx]

        self.state.status = "test"
        # self.train_util = 0.9

class GaussianRenderer:

    def __init__(self, ply_path, port=12025):
        self.load_ply(ply_path)

        # viewer
        self.server = viser.ViserServer(port=port, verbose=False)
        self.viewer = PoseViewer(
            server=self.server,
            render_fn=self._viewer_render_fn,
        )

    def load_ply(self, path):
        path = path.replace('\\', "/")
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))

        if len(extra_f_names) == 0:
            self.max_sh_degree = 0
        if len(extra_f_names) == 9:
            self.max_sh_degree = 1
        if len(extra_f_names) == 24:
            self.max_sh_degree = 2
        if len(extra_f_names) == 45:
            self.max_sh_degree = 3
        if len(extra_f_names) == 72:
            self.max_sh_degree = 4
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self.splats = dict(
            means=torch.tensor(xyz, dtype=torch.float, device="cuda"),
            quats=torch.tensor(rots, dtype=torch.float, device="cuda"),
            scales=torch.tensor(scales, dtype=torch.float, device="cuda"),
            opacities=torch.tensor(opacities[..., 0], dtype=torch.float, device="cuda"),
            sh0=torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous(),
            shN=torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        )

        self.active_sh_degree = self.max_sh_degree


    def set_cameras(self, pil_images, c2ws, fov_deg=50):
        self.viewer.init_scene(pil_images, c2ws, fov_deg)

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]
        
        rasterize_mode = "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=True,
            absgrad=False,
            sparse_grad=False,
            rasterize_mode=rasterize_mode,
            backgrounds=torch.ones(1, 3).to(means),
            **kwargs,
        )
        return render_colors, render_alphas, info

    @torch.no_grad()
    def _viewer_render_fn(self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]):
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().cuda()
        K = torch.from_numpy(K).float().cuda()

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.active_sh_degree,  # active all SH degrees
            radius_clip=0.1,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()
    
    # def show(self):




if __name__ == "__main__":
    import os
    import json
    import argparse
    from pathlib import Path
    from PIL import Image
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=str, required=True)
    parser.add_argument("--meta_file", type=str, default=None)
    args = parser.parse_args()


    master = GaussianRenderer(args.ply)
    if args.meta_file is not None and os.path.exists(args.meta_file):
        with open(args.meta_file, "r") as f:
            transforms = json.load(f)

        data_dir = Path(os.path.dirname(args.meta_file))
        images = []
        c2ws = []
        for frame in transforms:
            image_path = data_dir / frame["file_path"]
            image = Image.open(image_path)
            images.append(image)

            c2w = np.array(frame["transform_matrix"])
            c2ws.append(c2w)
        
        master.set_cameras(images, c2ws)

    print("Viewer running... Ctrl+C to exit.")
    time.sleep(1000000)