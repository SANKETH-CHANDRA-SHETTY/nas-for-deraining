import os
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.io import read_image, write_png
import numpy as np

# Configs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dir = Path("/home/dgx-s-user1/NAS_DR/datasets/test/Test2800/input")
target_dir = Path("/home/dgx-s-user1/NAS_DR/datasets/test/Test2800/target")
output_base = Path("/home/dgx-s-user1/NAS_DR/patched_datasets/test/Test2800")

patch_size = 128
stride = 112
min_variance = 5.0
max_pad_ratio = 0.25

# Output dirs
(input_dir_out := output_base / "input").mkdir(parents=True, exist_ok=True)
(target_dir_out := output_base / "target").mkdir(parents=True, exist_ok=True)

# Helper: pad using reflect
def reflect_pad(img):
    h, w = img.shape[-2:]
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    pad = (0, pad_w, 0, pad_h)  # left, right, top, bottom
    return F.pad(img, pad, mode='reflect'), h, w

# Helper: patch extraction
def extract_patches(img, patch_size, stride):
    patches = img.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    B, H, W, ph, pw = patches.shape
    patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, 3, ph, pw)
    return patches

# Main loop
def get_image_files(folder):
    return sorted([
        f for f in folder.iterdir()
        if f.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ])

input_files = get_image_files(input_dir)
target_files = get_image_files(target_dir)


assert len(input_files) == len(target_files)

total_patches = 0
total_skipped_patches = 0
skipped_images = 0
patches_per_image = []


for in_path, tgt_path in tqdm(zip(input_files, target_files), total=len(input_files), desc=" CUDA patching"):
    input_img = read_image(str(in_path)).float().to(device)
    target_img = read_image(str(tgt_path)).float().to(device)

    if input_img.shape != target_img.shape:
        print(f"Skipping {in_path.name} due to mismatch.")
        skipped_images += 1
        continue

    input_pad, h, w = reflect_pad(input_img)
    target_pad, _, _ = reflect_pad(target_img)

    in_patches = extract_patches(input_pad, patch_size, stride)
    tgt_patches = extract_patches(target_pad, patch_size, stride)

    # Variance filter
    gray_patches = 0.299 * in_patches[:, 0] + 0.587 * in_patches[:, 1] + 0.114 * in_patches[:, 2]
    variances = gray_patches.view(gray_patches.shape[0], -1).var(dim=1)
    mask = variances >= min_variance

    filtered_in = in_patches[mask]
    filtered_tgt = tgt_patches[mask]

    total_patches += filtered_in.shape[0]
    total_skipped_patches += (~mask).sum().item()
    patches_per_image.append(filtered_in.shape[0])

    for idx, (in_patch, tgt_patch) in enumerate(zip(filtered_in, filtered_tgt)):
        ext = in_path.suffix
        name = f"{in_path.stem}_patch{idx+1:03d}{ext}"
        write_png(in_patch.byte().cpu(), input_dir_out / name)
        write_png(tgt_patch.byte().cpu(), target_dir_out / name)


print("\n Patch Extraction Summary")
print(f" Total images processed     : {len(input_files)}")
print(f" Images skipped (mismatch)  : {skipped_images}")
print(f" Total patches saved        : {total_patches}")
print(f" Patches skipped (variance): {total_skipped_patches}")
print(f" Avg patches per image      : {np.mean(patches_per_image):.2f}")
print(f" Min patches from one image : {np.min(patches_per_image) if patches_per_image else 0}")
print(f" Max patches from one image : {np.max(patches_per_image) if patches_per_image else 0}")

print(" CUDA-based patching complete.")
