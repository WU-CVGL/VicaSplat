import os
from multiprocessing import Pool
import shutil

import numpy as np


origin_root = "/run/determined/workdir/data/scannet_freesplat/scannet/train"

output_root = "/run/determined/workdir/data/scannet_freesplat/new_scan/test"
raw_root = "/run/determined/workdir/data/scannet/scans"
count = 0

input_scenes = sorted(os.listdir(origin_root))

raw_scenes = sorted(os.listdir(raw_root))

scenes = []

# def copy_file(task):
#     img_src, img_dst, depth_src, depth_dst = task
#     shutil.copyfile(img_src, img_dst)
#     shutil.copyfile(de)

# while count < 100:
#     for scene in raw_scenes[1:]:
#         if scene not in input_scenes:
#             image_folder = os.path.join(raw_root, scene, "extract", "color")
#             depth_folder = os.path.join(raw_root, scene, "extract", "depth")
#             pose_folder = os.path.join(raw_root, scene, "extract", "pose")
#             intrincis_folder = os.path.join(raw_root, scene, "extract", "intrinsic")
            
#             # check input data
#             if not (os.path.exists(image_folder) and os.path.exists(depth_folder) and os.path.join(pose_folder) and os.path.join(intrincis_folder)):
#                 continue
#             img_num = len(os.listdir(image_folder))
#             depth_num = len(os.listdir(depth_folder))
#             pose_num = len(os.listdir(pose_folder))
#             if not (img_num == depth_num and depth_num == pose_num and img_num <= 1500):
#                 continue
#             poses = [np.loadtxt(os.path.join(pose_folder, f"{i}.txt")) for i in range(pose_num)]
#             poses = np.stack(poses)
#             if np.isnan(poses).any() or np.isinf(poses).any():
#                 continue
            
#             # pass check save to output folder
#             output_folder = os.path.join(output_root, scene)
#             output_img_folder = os.path.join(output_folder, "color")
#             output_depth_folder = os.path.join(output_folder, "depth")
#             output_intrinsic_folder = os.path.join(output_folder, "intrinsic")
#             os.makedirs(output_folder, exist_ok=True)
#             # os.makedirs(output_img_folder, exist_ok=True)
#             # os.makedirs(output_depth_folder, exist_ok=True)
            
#             # copy images and depth
#             copy_pairs = []
#             # print("copying")
#             # for i in range(img_num):
#             #     copy_pairs.append([os.path.join(image_folder, f"{i}.jpg"), os.path.join(output_img_folder, f"{i}.jpg"), os.path.join(depth_folder, f"{i}.png"), os.path.join(output_depth_folder, f"{i}.png")])
#             #     # shutil.copyfile(os.path.join(image_folder, f"{i}.jpg"),
#             #     #                 os.path.join(output_img_folder, f"{i}.jpg"))
#             #     # shutil.copyfile(os.path.join(depth_folder, f"{i}.png"),
#             #     #                 os.path.join(output_depth_folder, f"{i}.png"))
#             # with Pool(processes=50) as pool:
#             #     pool.map(copy_file, copy_pairs)
#             # shutil.copytree(image_folder, output_img_folder)
#             # shutil.copytree(depth_folder, output_depth_folder)
#             os.symlink(image_folder, output_img_folder)
#             os.symlink(depth_folder, output_depth_folder)
#             shutil.copytree(intrincis_folder, output_intrinsic_folder)
#             # print("copied")
            
#             np.save(os.path.join(output_folder, "extrinsics.npy"), poses)
#             count += 1
#             scenes.append(scene)
#             print(count)
#         else:
#             continue

scenes = sorted(os.listdir(output_root))

with open(os.path.join(output_root, "test_idx.txt"), "w") as f:
    for scene in scenes:
        f.write(f"{scene}\n")
