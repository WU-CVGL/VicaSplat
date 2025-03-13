import json
import subprocess
import sys
import os
from pathlib import Path
from typing import Literal, TypedDict, NamedTuple

import math
import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from torch import Tensor
from tqdm import tqdm
from PIL import Image

from src.scripts.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text

# INPUT_IMAGE_DIR = Path(".")
INPUT_IMAGE_DIR = Path("/run/determined/workdir/data/MVImgNet")
OUTPUT_DIR = Path("/run/determined/workdir/SSD/lzq-data/MVImgNet-processed")


# Target 100 MB per chunk.
TARGET_BYTES_PER_CHUNK = int(1e8)

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    fl_y_normed: float
    FovX: np.array
    fl_x_normed: float
    # image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        # R = np.transpose(qvec2rotmat(extr.qvec))
        R = qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec)

        # if intr.model=="SIMPLE_PINHOLE":
        if intr.model.startswith("SIMPLE"):
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            fl_x_normed = focal_length_x / width
            fl_y_normed = focal_length_x / height
        # elif intr.model=="PINHOLE":
        else:
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            fl_x_normed = focal_length_x / width
            fl_y_normed = focal_length_y / height
        # else:
        #     assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        # image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, # image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              fl_y_normed=fl_y_normed, fl_x_normed=fl_x_normed)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapSceneInfo(path):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images"
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    timestamps = []
    cameras = []
    images = []

    for cam_info in cam_infos:
        frame_id = int(cam_info.image_name)
        timestamps.append(frame_id)
        # intrinsic array
        intrinsic = [cam_info.fl_x_normed, cam_info.fl_y_normed, 0.5, 0.5, 0.0, 0.0]
        intrinsic = np.array(intrinsic, dtype=np.float32)

        # opencv w2c
        w2c = np.zeros((3, 4))
        w2c[:3, :3] = cam_info.R
        w2c[:3, 3] = cam_info.T
        w2c = w2c.flatten().astype(np.float32)
        camera = np.concatenate([intrinsic, w2c])
        cameras.append(camera)

        # load image
        img_raw = load_raw(cam_info.image_path)
        images.append(img_raw)

    timestamps = torch.tensor(timestamps, dtype=torch.int64)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)
    
    example = {
        "url": "",
        "timestamps": timestamps,
        "cameras": cameras,
        "images": images
    }
    return example

def get_example_keys() -> list[str]:
    subsets = np.arange(268)
    subsets = [str(i) for i in subsets]
    keys = []
    for subset in subsets:
        subdir = INPUT_IMAGE_DIR / subset
        # iterate through all the subdirectories
        if os.path.exists(subdir):
            for key in subdir.iterdir():
                if key.is_dir():
                    item = key.name.split('/')[-1]
                    item = '/'.join([subset, item])
                    keys.append(item)
        else:
            print(f"Direcotry {subdir} does not exist.")

    keys.sort()
    return keys


def get_size(path: Path) -> int:
    """Get file or folder size in bytes."""
    return int(subprocess.check_output(["du", "-b", path, "--exclude", "*_bg_removed.png"]).split()[0].decode("utf-8"))


def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    try:
        raw = torch.tensor(np.memmap(path, dtype="uint8", mode="r"))
    except:
        raise ValueError(f"Invalid path '{path}'")
    return raw

def load_images(example_path: Path) -> dict[str, UInt8[Tensor, "..."]]:
    """Load JPG images as raw bytes (do not decode)."""

    return {path.stem: load_raw(path) for path in example_path.iterdir()}


class Metadata(TypedDict):
    url: str
    timestamps: Int[Tensor, " camera"]
    cameras: Float[Tensor, "camera entry"]


class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]


def opengl_c2w_to_opencv_w2c(c2w: np.ndarray) -> np.ndarray:
    c2w = c2w.copy()
    c2w[2, :] *= -1
    c2w = c2w[np.array([1, 0, 2, 3]), :]
    c2w[0:3, 1:3] *= -1
    w2c_opencv = np.linalg.inv(c2w)
    return w2c_opencv


def load_metadata(file_path: Path) -> Metadata:
    return readColmapSceneInfo(file_path)


if __name__ == "__main__":
    # for stage in ("train", "test"):
    for stage in ["train"]:
        keys_ordered = get_example_keys()

        # Shuffle the keys
        randperm_indices = torch.randperm(len(keys))
        keys = [keys_ordered[x] for x in randperm_indices]

        chunk_size = 0
        chunk_index = 0
        chunk: list[Example] = []

        def save_chunk():
            global chunk_size
            global chunk_index
            global chunk

            chunk_key = f"{chunk_index:0>6}"
            print(
                f"Saving chunk {chunk_key} of {len(keys)} ({chunk_size / 1e6:.2f} MB)."
            )
            dir = OUTPUT_DIR / stage
            dir.mkdir(exist_ok=True, parents=True)
            torch.save(chunk, dir / f"{chunk_key}.torch")

            # Reset the chunk.
            chunk_size = 0
            chunk_index += 1
            chunk = []

        for key in keys:
            # chunk_key = f"{chunk_index:0>6}"
            # chunk_save_path = OUTPUT_DIR / stage / f"{chunk_key}.torch"
            # if os.path.exists(chunk_save_path):
            #     print(f"Skip existing chunk {chunk_save_path}.")
            #     # Reset the chunk.
            #     chunk_size = 0
            #     chunk_index += 1
            #     chunk = []
            #     continue
              
            image_dir = INPUT_IMAGE_DIR / key / 'images'
            colmap_dir = INPUT_IMAGE_DIR / key

            if not image_dir.exists() or not colmap_dir.exists():
                print(f"Skipping {key} because it is missing.")
                continue

            num_bytes = get_size(image_dir)     # exclude *_bg_removed.png
            # Read images and metadata.
            try:
                example = load_metadata(colmap_dir)
            except:
                print(f"Skipping {key} because of missing metadata.")
                continue

            assert len(example["images"]) == len(example["timestamps"]), f"len(example['images'])={len(example['images'])}, len(example['timestamps'])={len(example['timestamps'])}"

            # Add the key to the example.
            example["key"] = key

            print(f"    Added {key} to chunk ({num_bytes / 1e6:.2f} MB).")
            chunk.append(example)
            chunk_size += num_bytes

            if chunk_size >= TARGET_BYTES_PER_CHUNK:
                save_chunk()

        if chunk_size > 0:
            save_chunk()

        # generate index
        print("Generate key:torch index...")
        index = {}
        stage_path = OUTPUT_DIR / stage
        for chunk_path in tqdm(list(stage_path.iterdir()), desc=f"Indexing {stage_path.name}"):
            if chunk_path.suffix == ".torch":
                chunk = torch.load(chunk_path)
                for example in chunk:
                    index[example["key"]] = str(chunk_path.relative_to(stage_path))
        with (stage_path / "index.json").open("w") as f:
            json.dump(index, f)
