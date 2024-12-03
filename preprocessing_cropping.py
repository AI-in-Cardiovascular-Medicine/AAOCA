import os
from argparse import ArgumentParser
from os.path import join

import itk
import nibabel as nib
import numpy as np
from deep_utils import PyUtils
from scipy.ndimage import binary_dilation, distance_transform_edt
from tqdm import tqdm


def find_center_of_interest(nii_path: str, label1_value: int, label2_value: int):
    """
    Process a NIfTI file to determine the center of interest between two labeled regions.
    The center is calculated as follows:
        1. If the two labels overlap, the center of their intersection is computed as the mean coordinates of the
           intersecting region.
        2. If no overlap exists, the closest point in `label2` to the edge of `label1` is returned.
    :param nii_path: Path to the NIfTI file containing the 3D segmentation.
    :param label1_value: Integer value of the first label.
    :param label2_value: Integer value of the second label.
    :return: Tuple[int, int, int] with the 3D coordinates of the calculated center of interest.
    """
    nii_data = nib.load(nii_path).get_fdata()
    label1_mask = nii_data == label1_value
    label2_mask = nii_data == label2_value
    # Perform binary dilation on both masks to expand their boundaries.
    dilated_label1_mask = binary_dilation(label1_mask)
    dilated_label2_mask = binary_dilation(label2_mask)
    # Find the intersection of the two dilated regions.
    intersection_mask = dilated_label1_mask & dilated_label2_mask
    if intersection_mask.any():
        # If the two regions overlap, compute the mean coordinates of the intersection.
        intersection_indices = np.array(np.where(intersection_mask))
        center_of_intersection = np.mean(intersection_indices, axis=1)
        center_of_interest = np.round(center_of_intersection).astype(int)
    else:
        # If no overlap exists, find the closest point in label2 to label1 using distance transform.
        distance = distance_transform_edt(~dilated_label2_mask, return_distances=True, return_indices=False)
        distance[~dilated_label1_mask] = np.inf
        closest_point_label1 = np.unravel_index(np.argmin(distance), distance.shape)
        center_of_interest = closest_point_label1
    return center_of_interest


def crop_and_save_itk_with_cm_and_offset_adjusted(image_path: str, seg_path: str, center: list[int],
                                                  crop_size_cm: tuple[int, int, int], offset_cm: tuple[int, int, int],
                                                  cropped_image_path: str, cropped_seg_path: str):
    """
    Crop a region of interest from a 3D image and segmentation, adjust for an offset, and save the cropped outputs.
    :param image_path: Path to the input image file.
    :param seg_path: Path to the input segmentation file corresponding to the image.
    :param center: The center point (in pixels) around which to crop the image and segmentation.
    :param crop_size_cm: The desired size of the cropped region in centimeters for each dimension (x, y, z).
    :param offset_cm: Offset in centimeters to adjust the cropping center for each dimension (x, y, z).
    :param cropped_image_path: Path to save the cropped image file.
    :param cropped_seg_path: Path to save the cropped segmentation file.
    :return:
    """
    # Define the pixel type and image dimension
    pixel_type = itk.F
    dimension = 3
    image_type = itk.Image[pixel_type, dimension]
    # Load the input image and segmentation
    image = itk.imread(image_path, pixel_type)
    seg = itk.imread(seg_path, pixel_type)
    # Get the image spacing (in mm) and size
    spacing = image.GetSpacing()
    image_size = image.GetLargestPossibleRegion().GetSize()
    offset_pixels = [int(offset_cm[i] * 10 / spacing[i]) for i in range(3)]  # Convert offset from cm to pixels
    new_center = [center[i] + offset_pixels[i] for i in range(3)]  # Adjust the center by applying the offset
    # Convert crop size from cm to pixels and ensure it fits within the image size
    crop_size_pixels = [min(int(crop_size_cm[i] * 10 / spacing[i]), image_size[i]) for i in range(3)]
    # Compute the starting index for cropping
    start = [max(0, min(int(new_center[i] - crop_size_pixels[i] / 2), image_size[i] - crop_size_pixels[i])) for i in
             range(3)]
    # Define the cropping region
    region = itk.ImageRegion[dimension]()
    region.SetIndex(start)
    region.SetSize(crop_size_pixels)
    # Create and apply the cropping filters for the image and segmentation
    extract_filter_image = itk.ExtractImageFilter[image_type, image_type].New(Input=image, ExtractionRegion=region)
    extract_filter_seg = itk.ExtractImageFilter[image_type, image_type].New(Input=seg, ExtractionRegion=region)
    extract_filter_image.Update()
    extract_filter_seg.Update()
    # Save the cropped image and segmentation
    itk.imwrite(extract_filter_image.GetOutput(), cropped_image_path)
    itk.imwrite(extract_filter_seg.GetOutput(), cropped_seg_path)


def process_folders(main_folder: str, left_ventricle_label_value: int, aorta_label_value: int,
                    crop_size_cm: tuple[int, int, int],
                    offset_cm: tuple[int, int, int]):
    """
    Processes all sub-folders in a given main directory to find and crop 3D images and their segmentations.
    :param main_folder: Path to the root directory containing the images and segmentation files.
    :param left_ventricle_label_value: Label value representing the left ventricle in the segmentation files.
    :param aorta_label_value: Label value representing the aorta in the segmentation files.
    :param crop_size_cm: Dimensions of the cropped region in centimeters for each axis (x, y, z).
    :param offset_cm: Offset values.
    :return: None (the function saves cropped images and segmentations to the same directory)
    """
    for root, dirs, filenames in tqdm(os.walk(main_folder), total=len(list(os.walk(main_folder)))):
        for filename in filenames:
            if filename == "seg.nii.gz":
                seg_path = join(root, filename)
                image_path = join(root, "img.nii.gz")
                cropped_image_path = join(root, f"img_cropped.nii.gz")
                cropped_seg_path = join(root, f'seg_cropped.nii.gz')
                # Skip if cropped files already exist
                if os.path.exists(cropped_seg_path) and os.path.exists(cropped_image_path):
                    continue
                print(30 * "#")
                print(f"Current image being processed: {image_path}")

                if os.path.isfile(image_path):
                    try:
                        process_sample(
                            seg_path,
                            cropped_image_path,
                            cropped_seg_path,
                            image_path,
                            crop_size_cm,
                            left_ventricle_label_value,
                            aorta_label_value,
                            offset_cm=offset_cm
                        )
                        print(f"Processed and saved cropped files for {root}.")
                        print(30 * "#")
                    except Exception as e:
                        PyUtils.print(f"Failed to process {image_path}: {e}")
                else:
                    PyUtils.print(f"Image does not exists: {image_path}", color="Red")


def process_sample(seg_path: str,
                   cropped_image_path: str,
                   cropped_seg_path: str,
                   image_path: str,
                   crop_size_cm: tuple[int, int, int],
                   left_ventricle_label_value: int,
                   aorta_label_value: int,
                   offset_cm: tuple[int, int, int]):
    """
    Code cropping only one sample
    :param seg_path:
    :param cropped_image_path:
    :param cropped_seg_path:
    :param image_path:
    :param crop_size_cm:
    :param left_ventricle_label_value:
    :param aorta_label_value:
    :param offset_cm:
    :return:
    """
    # Find the center of interest based on the segmentation labels
    center = find_center_of_interest(seg_path, left_ventricle_label_value, aorta_label_value)
    # Crop and save the image and segmentation with offset adjustments
    crop_and_save_itk_with_cm_and_offset_adjusted(image_path, seg_path, center, crop_size_cm,
                                                  offset_cm, cropped_image_path, cropped_seg_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_dir", default="data/anomaly_detection/train")
    parser.add_argument("--crop_size", default=(8, 8, 6), type=int, nargs="+")
    args = parser.parse_args()
    left_ventricle_label_value = 2
    aorta_label_value = 3
    offset_cm = (-1, 1, 1)
    process_folders(args.input_dir, left_ventricle_label_value, aorta_label_value, args.crop_size, offset_cm=offset_cm)
