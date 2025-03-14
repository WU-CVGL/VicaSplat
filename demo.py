import os
import json
import gradio as gr
import functools
import imageio
from pathlib import Path
from colorama import Fore
from PIL import Image
from PIL.ImageOps import exif_transpose

import torch
import torchvision.transforms as tvf
from torchvision.utils import make_grid
import numpy as np

import hydra
import trimesh
import matplotlib.pyplot as pl
from scipy.spatial.transform import Rotation

from src.visualization.camera_trajectory.interpolation import interpolate_extrinsics, interpolate_intrinsics
from src.visualization.dust3r_viz import add_scene_cam, OPENGL, CAM_COLORS
from src.misc.cam_utils import simple_intrin_matrix_from_fov
from src.model.decoder.cuda_splatting import render_cuda
from src.model.ply_export import export_ply
from src.model.encoder import get_encoder
from src.misc.image_io import prep_image, save_image
from src.config import load_typed_root_config
from src.misc.weight_modify import checkpoint_filter_fn
from omegaconf import DictConfig

try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def todevice(batch, device, callback=None, non_blocking=False):
    ''' Transfer some variables to another device (i.e. GPU, CPU:torch, CPU:numpy).

    batch: list, tuple, dict of tensors or other things
    device: pytorch device or 'numpy'
    callback: function that would be called on every sub-elements.
    '''
    if callback:
        batch = callback(batch)

    if isinstance(batch, dict):
        return {k: todevice(v, device) for k, v in batch.items()}

    if isinstance(batch, (tuple, list)):
        return type(batch)(todevice(x, device) for x in batch)

    x = batch
    if device == 'numpy':
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    elif x is not None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if torch.is_tensor(x):
            x = x.to(device, non_blocking=non_blocking)
    return x


to_device = todevice  # alias

def to_numpy(x): return todevice(x, 'numpy')
def to_cpu(x): return todevice(x, 'cpu')
def to_cuda(x): return todevice(x, 'cuda')

def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = Image.LANCZOS
    elif S <= long_edge_size:
        interp = Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)


def load_images(folder_or_list, size=256, square_ok=False, verbose=True):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', sorted(folder_or_list, key=lambda x: x.split('/')[-1])

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(Image.open(os.path.join(root, path))).convert('RGB')
        W1, H1 = img.size

        # resize short side to 256 (then crop)
        img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))

        W, H = img.size
        cx, cy = W//2, H//2
        half = min(cx, cy)
        img = img.crop((cx-half, cy-half, cx+half, cy+half))

        W2, H2 = img.size
        if verbose:
            print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')

        imgs.append(ImgNorm(img))

    assert imgs, 'no images foud at '+ root
    if verbose:
        print(f' (Found {len(imgs)} images)')

    imgs = torch.stack(imgs, dim=0)
    return imgs   # (V, C, H, W)


def _convert_scene_output_to_glb(outdir, imgs, pts3d, focals, cams2world, cam_size=0.05):
    assert len(pts3d) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    pct = trimesh.PointCloud(pts3d.reshape(-1, 3), colors=imgs.reshape(-1, 3))
    scene.add_geometry(pct)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        camera_edge_color = CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, 'xyz_and_camera.glb')

    print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile


def get_3D_model_from_scene(outdir, scene, cam_size=0.05):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """

    # get optimized values from scene
    rgbimg = scene["imgs"][0].permute(0, 2, 3, 1).cpu()
    intrins = scene["camera_intrins"][0].cpu()
    focals = intrins[:, 0, 0] * rgbimg.shape[1]
    cams2world = scene["camera_poses"][0].cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = scene["pts3d"][0]
    return _convert_scene_output_to_glb(outdir, rgbimg, pts3d, focals, cams2world, 
                                        cam_size=cam_size)

@torch.no_grad()
def inference(model, imgs, fovx_deg=None, fovy_deg=None):
    inputs = {"image": imgs}

    if model.backbone.config.use_intrinsic_embedding:
        assert fovx_deg > 0 or fovy_deg > 0, "need to provide valid fovx and fovy"
        fovx = np.deg2rad(fovx_deg) if fovx_deg > 0 else None
        fovy = np.deg2rad(fovy_deg) if fovy_deg > 0 else None
        fov = torch.as_tensor([fovx or fovy, fovy or fovx], dtype=torch.float)
        intrins = simple_intrin_matrix_from_fov(fov[None])
        inputs["intrinsics"] = intrins
    else:
        intrins = None

    model_outs = model(inputs, compute_viewspace_depth=False)

    return dict(
        imgs=imgs*0.5+0.5,
        gaussians=model_outs["gaussians"],
        pts3d=model_outs["gaussian_centers"],
        camera_poses=model_outs["gaussian_camera_extrins"],
        camera_intrins=model_outs["gaussian_camera_intrins"] if intrins is None else intrins,
    )

@torch.no_grad()
def render_video_interpolation(device, gaussians, camera_poses, camera_intrins, n_interp_per_interv=10, near=0.01, far=100.0):

    def trajectory_fn(t_per_interv):
        extrinsics = interpolate_extrinsics(
            initial=camera_poses[:-1],
            final=camera_poses[1:],
            t=t_per_interv
        ).reshape(-1, 4, 4)
        intrinsics = interpolate_intrinsics(
            initial=camera_intrins[:-1],
            final=camera_intrins[1:],
            t=t_per_interv
        ).reshape(-1, 3, 3)
        return extrinsics, intrinsics
    
    t = torch.linspace(0, 1, n_interp_per_interv, dtype=torch.float, device=device)
    # # more smooth traj
    # t = (torch.cos(torch.pi * (t + 1)) + 1) / 2
    extrinsics, intrinsics = trajectory_fn(t)
    v = extrinsics.shape[0]

    images, depth = render_cuda(
        extrinsics,
        intrinsics,
        torch.as_tensor([near]*v, dtype=torch.float, device=device),
        torch.as_tensor([far]*v, dtype=torch.float, device=device),
        (256, 256),
        background_color=torch.zeros(v, 3, device=device),
        gaussian_means=gaussians.means.reshape(-1, 3),
        gaussian_covariances=gaussians.covariances.reshape(-1, 3, 3),
        gaussian_sh_coefficients=gaussians.harmonics.flatten(0, 3),
        gaussian_opacities=gaussians.opacities.flatten(),
        scale_invariant=False,
    )

    # reverse loop
    images = torch.cat([images, images.flip(dims=(0,))], dim=0)

    return images   # (2*v, C, H, W)


def get_reconstructed_scene(outdir, model, device, filelist, fovx_deg, fovy_deg, cam_size=0.05,):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """

    imgs = load_images(filelist, size=256, verbose=False)
    imgs = imgs.to(device)

    outs = inference(model, imgs[None], fovx_deg, fovy_deg)

    rendered_video_frames = render_video_interpolation(
        device, outs["gaussians"], outs["camera_poses"][0], outs["camera_intrins"][0]
    )

    # Gaussian centers and camera visualization
    out_scene = get_3D_model_from_scene(outdir, outs, cam_size)

    # save video
    video_save_path = os.path.join(outdir, "video_interpolation.mp4")
    video_frames = to_numpy(rendered_video_frames.permute(0, 2, 3, 1))
    video_frames = (video_frames.clip(0, 1) * 255).astype(np.uint8)
    imageio.mimwrite(video_save_path, video_frames, fps=30)

    # export gaussians (.ply)
    gaussian_save_path = os.path.join(outdir, "gaussians.ply")
    gaussians = outs["gaussians"]
    export_ply(
        0,
        means=gaussians.means.flatten(0, 3),
        scales=gaussians.scales.flatten(0, 3),
        rotations=gaussians.rotations.flatten(0, 3),
        harmonics=gaussians.harmonics.flatten(0, 3),
        opacities=gaussians.opacities.flatten(),
        path=Path(gaussian_save_path),
        save_sh_dc_only=False,
    )
    
    # save predicted camera poses and input images
    frames = []
    for i in range(imgs.shape[0]):
        save_image(imgs[i]*0.5+0.5, os.path.join(outdir, "context", f"{i}.png"))
        frame = {
            "file_path": f"context/{i}.png",
            "transform_matrix": outs["camera_poses"][0][i].cpu().numpy().tolist()
        }
        frames.append(frame)
    with open(os.path.join(outdir, "transforms.json"), "w") as f:
        json.dump(frames, f, indent=4)

    # also visualize the input images as a grid
    context = make_grid(imgs*0.5+0.5, nrow=4, padding=2)
    context = to_numpy(context.permute(1, 2, 0))

    return outs, out_scene, context, video_save_path, gaussian_save_path



# gradio UI
def main_demo(tmpdir, model, device, server_name='0.0.0.0', server_port=None):
    _TITLE = '''VicaSplat Demo'''

    _DESCRIPTION = '''
    <div>
    <a style="display:inline-block; margin-left: .5em" href="https://github.com/WU-CVGL/VicaSplat"><img src='https://img.shields.io/github/stars/WU-CVGL/VicaSplat?style=social'/></a>
    </div>

    * Input should be a directory containing images.
    '''

    block = gr.Blocks(title=_TITLE).queue()
    with block:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('# ' + _TITLE)
        gr.Markdown(_DESCRIPTION)
        # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
        state = gr.State(None)

        with gr.Row(variant='panel'):
            with gr.Column(scale=1):
                # input image dir
                input_files = gr.File(file_count="directory")
                # input_files = gr.File(file_count="multiple") # images
                # (Optional) fovx degree
                input_fovx_deg = gr.Number(label="fovx degree", minimum=0, maximum=120, step=0.1, value=None)
                # (Optional) fovy degree
                input_fovy_deg = gr.Number(label="fovy degree", minimum=0, maximum=120, step=0.1, value=None)
                # adjust the camera size in the output pointcloud
                cam_size = gr.Slider(label="cam_size", value=0.1, minimum=0.001, maximum=0.2, step=0.001)
                # run button
                button_run = gr.Button("Run")
                # ply file
                output_ply = gr.File(label="ply")

            with gr.Column(scale=2):
                # Input images
                output_context = gr.Image(label="input images")
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Tab("Rendered video along predicted camera trajectory"):
                            output_video = gr.Video(label="video")
                    with gr.Column(scale=2):
                        with gr.Tab("Viz of Gaussian centers and camera"):
                            output_scene = gr.Model3D()

        recon_fn = functools.partial(get_reconstructed_scene, tmpdir, model, device)
        button_run.click(
            recon_fn, 
            inputs=[input_files, input_fovx_deg, input_fovy_deg, cam_size], 
            outputs=[state, output_scene, output_context, output_video, output_ply]
        )
        
    block.launch(server_name=server_name, server_port=server_port, share=False)


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


def load_model(cfg):
    encoder, _ = get_encoder(cfg.model.encoder)

    # Load the encoder weights.
    if cfg.model.encoder.pretrained_weights and cfg.mode == "train":
        weight_path = cfg.model.encoder.pretrained_weights
        print(cyan(f"Init model from pretrained weights {weight_path}."))
        ckpt_weights = torch.load(weight_path, map_location='cpu')
        if 'model' in ckpt_weights:
            ckpt_weights = ckpt_weights['model']
            ckpt_weights = checkpoint_filter_fn(ckpt_weights, encoder)
            missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)
        elif 'state_dict' in ckpt_weights:
            ckpt_weights = ckpt_weights['state_dict']
            ckpt_weights = {k[8:]: v for k, v in ckpt_weights.items() if k.startswith('encoder.')}
            if ckpt_weights['downstream_head1.dpt.head.4.bias'].shape[0] == 4 and not encoder.cfg.predict_conf:
                ckpt_weights['downstream_head1.dpt.head.4.weight'] = ckpt_weights['downstream_head1.dpt.head.4.weight'][0:3]
                ckpt_weights['downstream_head1.dpt.head.4.bias'] = ckpt_weights['downstream_head1.dpt.head.4.bias'][0:3]
            missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)
        else:
            raise ValueError(f"Invalid checkpoint format: {weight_path}")
        
    return encoder
    
@hydra.main(
    version_base=None, config_path="./config", config_name="main"
)
def main(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    print(cyan(f"Saving outputs to {output_dir}."))

    model = load_model(cfg)
    model.eval()
    model.to(device)

    main_demo(tmpdir=output_dir, model=model, device=device,)

if __name__ == "__main__":
    main()