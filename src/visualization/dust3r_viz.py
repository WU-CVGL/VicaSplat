# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Visualization utilities using trimesh
# --------------------------------------------------------
import PIL.Image
import numpy as np
from scipy.spatial.transform import Rotation
import torch

try:
    import trimesh
except ImportError:
    print('/!\\ module trimesh is not installed, cannot visualize results /!\\')


def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


def pts3d_to_trimesh(img, pts3d, valid=None):
    H, W, THREE = img.shape
    assert THREE == 3
    assert img.shape == pts3d.shape

    vertices = pts3d.reshape(-1, 3)

    # make squares: each pixel == 2 triangles
    idx = np.arange(len(vertices)).reshape(H, W)
    idx1 = idx[:-1, :-1].ravel()  # top-left corner
    idx2 = idx[:-1, +1:].ravel()  # right-left corner
    idx3 = idx[+1:, :-1].ravel()  # bottom-left corner
    idx4 = idx[+1:, +1:].ravel()  # bottom-right corner
    faces = np.concatenate((
        np.c_[idx1, idx2, idx3],
        np.c_[idx3, idx2, idx1],  # same triangle, but backward (cheap solution to cancel face culling)
        np.c_[idx2, idx3, idx4],
        np.c_[idx4, idx3, idx2],  # same triangle, but backward (cheap solution to cancel face culling)
    ), axis=0)

    # prepare triangle colors
    face_colors = np.concatenate((
        img[:-1, :-1].reshape(-1, 3),
        img[:-1, :-1].reshape(-1, 3),
        img[+1:, +1:].reshape(-1, 3),
        img[+1:, +1:].reshape(-1, 3)
    ), axis=0)

    # remove invalid faces
    if valid is not None:
        assert valid.shape == (H, W)
        valid_idxs = valid.ravel()
        valid_faces = valid_idxs[faces].all(axis=-1)
        faces = faces[valid_faces]
        face_colors = face_colors[valid_faces]

    assert len(faces) == len(face_colors)
    return dict(vertices=vertices, face_colors=face_colors, faces=faces)


def cat_meshes(meshes):
    vertices, faces, colors = zip(*[(m['vertices'], m['faces'], m['face_colors']) for m in meshes])
    n_vertices = np.cumsum([0]+[len(v) for v in vertices])
    for i in range(len(faces)):
        faces[i][:] += n_vertices[i]

    vertices = np.concatenate(vertices)
    colors = np.concatenate(colors)
    faces = np.concatenate(faces)
    return dict(vertices=vertices, face_colors=colors, faces=faces)


def add_scene_cam(scene, pose_c2w, edge_color, image=None, focal=None, imsize=None, 
                  screen_width=0.03, marker=None):
    if image is not None:
        image = np.asarray(image)
        H, W, THREE = image.shape
        assert THREE == 3
        if image.dtype != np.uint8:
            image = np.uint8(255*image)
    elif imsize is not None:
        W, H = imsize
    elif focal is not None:
        H = W = focal / 1.1
    else:
        H = W = 1

    if isinstance(focal, np.ndarray):
        focal = focal[0]
    if not focal:
        focal = min(H,W) * 1.1 # default value

    # create fake camera
    height = max( screen_width/10, focal * screen_width / H )
    width = screen_width * 0.5**0.5
    rot45 = np.eye(4)
    rot45[:3, :3] = Rotation.from_euler('z', np.deg2rad(45)).as_matrix()
    rot45[2, 3] = -height  # set the tip of the cone = optical center
    aspect_ratio = np.eye(4)
    aspect_ratio[0, 0] = W/H
    transform = pose_c2w @ OPENGL @ aspect_ratio @ rot45
    cam = trimesh.creation.cone(width, height, sections=4)  # , transform=transform)

    # this is the image
    if image is not None:
        vertices = geotrf(transform, cam.vertices[[4, 5, 1, 3]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]])
        img = trimesh.Trimesh(vertices=vertices, faces=faces)
        uv_coords = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        img.visual = trimesh.visual.TextureVisuals(uv_coords, image=PIL.Image.fromarray(image))
        scene.add_geometry(img)

    # this is the camera mesh
    rot2 = np.eye(4)
    rot2[:3, :3] = Rotation.from_euler('z', np.deg2rad(2)).as_matrix()
    vertices = np.r_[cam.vertices, 0.95*cam.vertices, geotrf(rot2, cam.vertices)]
    vertices = geotrf(transform, vertices)
    faces = []
    for face in cam.faces:
        if 0 in face:
            continue
        a, b, c = face
        a2, b2, c2 = face + len(cam.vertices)
        a3, b3, c3 = face + 2*len(cam.vertices)

        # add 3 pseudo-edges
        faces.append((a, b, b2))
        faces.append((a, a2, c))
        faces.append((c2, b, c))

        faces.append((a, b, b3))
        faces.append((a, a3, c))
        faces.append((c3, b, c))

    # no culling
    faces += [(c, b, a) for a, b, c in faces]

    cam = trimesh.Trimesh(vertices=vertices, faces=faces)
    cam.visual.face_colors[:, :3] = edge_color
    scene.add_geometry(cam)

    if marker == 'o':
        marker = trimesh.creation.icosphere(3, radius=screen_width/4)
        marker.vertices += pose_c2w[:3,3]
        marker.visual.face_colors[:,:3] = edge_color
        scene.add_geometry(marker)


OPENGL = np.array([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])


CAM_COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (255, 204, 0), (0, 204, 204),
              (128, 255, 255), (255, 128, 255), (255, 255, 128), (0, 0, 0), (128, 128, 128)]