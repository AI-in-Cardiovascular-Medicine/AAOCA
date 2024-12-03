import argparse
import os
from os.path import join, split

import nibabel as nib
import numpy as np
import torchio as tio
from joblib import Parallel, delayed


def normalize_and_resample(input_img_path: str,
                           output_img_name: str = "img_cropped_resampled.nii.gz",
                           target_pixel: tuple[int, int, int] = (215, 215, 85)):
    """
    Normalize and resample the input nifty image!
    :param input_img_path: Path to folder containing images
    :param output_img_name: Name of the output image
    :param target_pixel: Target pixel shape of the output image. Default is (215, 215, 85) which is the median of
    retrospective dataset.
    :return:
    """
    out_path = join(os.path.dirname(input_img_path), output_img_name)
    if os.path.exists(out_path):
        print(f"[INFO] The output: {out_path} is already created. Exiting the function!")
        return
    nib_img = nib.load(input_img_path)  # load image
    img = nib_img.get_fdata().astype(np.float32)
    original_pixel = nib_img.header["dim"][1:4]
    # Resample image
    target_ratio = [(original_pixel[i] / target_pixel[i]) for i in range(len(original_pixel))]
    transform = tio.Resample(target=target_ratio, image_interpolation="bspline")
    img_resampled = transform(np.expand_dims(img, axis=0))[0]
    img_resampled = img_resampled[:int(target_pixel[0]), :int(target_pixel[1]), :int(target_pixel[2])]
    # Normalize image
    img_resampled_normalized = np.clip(img_resampled, a_min=-1024, a_max=1024)
    img_resampled_normalized = np.array((img_resampled_normalized + 1024) / 2048 * 255).astype(int) / 255
    # Save resampled image as nifty and npy
    nib_img.header["pixdim"][1:4] = nib_img.header["pixdim"][1:4] * target_ratio
    nib_img.header["dim"][1:4] = target_pixel
    new_nib_img = nib.Nifti1Image(img_resampled_normalized, nib_img.affine, nib_img.header)
    new_nib_img.to_filename(out_path)


def get_z(image_path):
    """
    Get z spacing from image given image path.
    :param image_path:
    :return:
    """
    # get z from nib image
    nib_img = nib.load(image_path)
    z = nib_img.header["pixdim"][3]
    return z


def main():
    """
    For each dataset, resample the images to the required resolution and normalize them. Preprocessed images are then
    saved as .nii.gz files, and a .npz file is created with images and labels for training and testing steps.
    Required dataset structure: dataset_path/outcome/pat_id/image/name_input_image.nii.gz
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to a dataset directory containing cropped images.")
    parser.add_argument("--output_path", type=str, required=True, help="Output path where npz datasets are saved.")
    parser.add_argument("--name_input_image", type=str, default="img_cropped", help="Name of the .nii.gz image files.")
    parser.add_argument("--name_output_image", type=str, default="img_cropped_resampled",
                        help="Name of the .nii.gz image files after resampling.")
    parser.add_argument("--resampling_resolution", nargs="+", type=int, default=(215, 215, 85),
                        help="Images are resampled to this target resolution (number of x,y,z pixels).")
    parser.add_argument("--z_threshold", type=float, default=0.8,
                        help="only images with z spacing lower than threshold (in mm) are kept.")

    args = parser.parse_args()
    dataset_path = args.dataset_path
    output_path = args.output_path
    resampling_shape = args.resampling_resolution
    name_resampled_imgs = args.name_output_image + ".nii.gz"
    outcomes = ["0", "1"]
    # 0 -> healthy, 1 -> narco patients
    # 0 -> right origin, 1 -> left origin
    # 0 -> low risk, 1 -> high risk

    print(f"--- Processing folder {dataset_path} ---")
    os.makedirs(split(output_path)[0], exist_ok=True)  # Create output directory if it doesn't exist.
    print("\t+ Normalizing images +")
    for outcome in outcomes:
        folder = join(dataset_path, outcome)
        path_imgs = [join(dp, f) for dp, dn, fn in os.walk(folder) for f in fn if args.name_input_image in f]
        print(folder, "  ,  ", len(path_imgs), " images")
        Parallel(n_jobs=-1)(delayed(normalize_and_resample)(path_img, name_resampled_imgs, resampling_shape)
                            for path_img in path_imgs)

    print("\t+ Creating npz arrays +")
    path_imgs_nib = []
    labels = []
    pat_ids = []
    img_names = []
    for i, outcome in enumerate(outcomes):
        folder = join(dataset_path, outcome)
        paths_outcome = [join(dp, f) for dp, dn, fn in os.walk(folder) for f in fn if
                         f == name_resampled_imgs]
        path_imgs_nib += paths_outcome
        # update label and pat_id lists
        labels += [i] * len(paths_outcome)
        pat_ids += [split(split(split(path_img)[0])[0])[-1] for path_img in paths_outcome]
        img_names += [split(split(path_img)[0])[-1] for path_img in paths_outcome]

    # remove images with low z resolution and with LF (wrong Kernel)
    z = np.array(Parallel(n_jobs=-1)(delayed(get_z)(path_img) for path_img in path_imgs_nib))
    mask_z = z < args.z_threshold
    mask_lf = np.array(["_LF_" not in path_img for path_img in path_imgs_nib])  # specific to our data
    ids = np.nonzero(mask_z & mask_lf)[0]
    path_imgs_nib = np.array(path_imgs_nib)[ids]
    labels = np.array(labels)[ids]
    pat_ids = np.array(pat_ids)[ids]
    img_names = np.array(img_names)[ids]

    print("\t+ Writing numpy array... +")
    # save arrays in npz format
    n = len(path_imgs_nib)
    npy_imgs = np.zeros(shape=(n, *resampling_shape), dtype=np.float16)
    for i, path_img in enumerate(path_imgs_nib):
        npy_imgs[i] = nib.load(path_img).get_fdata().astype(np.float16)
    np.savez_compressed(output_path,
                        imgs=npy_imgs,
                        labels=labels,
                        pat_ids=pat_ids,
                        img_names=img_names
                        )


if __name__ == '__main__':
    main()
