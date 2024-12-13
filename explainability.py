import sys

import numpy as np
import torchio as tio

import argparse
import os

import nibabel as nib
import torch
from deep_utils import NIBUtils
from monai.visualize.class_activation_maps import GradCAMpp
from torch import nn

from utils import normalize_and_resample, load_model


def get_grad_cam(out_grad_path: str, layer_name: str, net: nn.Module, img_path: str):
    """
    Generates a Grad-CAM++ heatmap for a trained model and input image and saves the result as a NIfTI image.
    :param out_grad_path: The output file path for saving the Grad-CAM++ heatmap. If the file already exists, the
                          function exits without any computation.
    :param layer_name:The name of the target layer in the neural network for which Grad-CAM++ is computed.
    :param net: The neural network model.
    :param img_path: The file path to the input image.
    :return: None. The function saves the Grad-CAM++ heatmap to the specified file path.
    """
    org_arr, org_img = NIBUtils.get_array_img(img_path)

    # Normalize and resample image
    img = normalize_and_resample(img_path)
    img = torch.Tensor(img).to(torch.float32)
    # Rearrange image axes for model input
    last_id = len(img.shape) - 1
    img = img.swapaxes(last_id - 1, last_id).swapaxes(last_id - 2, last_id - 1)
    img = img.unsqueeze(0).unsqueeze(0)
    # Compute Grad-CAM++
    cam = GradCAMpp(nn_module=net, target_layers=[layer_name])
    result = cam(x=img.to(device))
    # Save results maintaining the same affine matrix
    affine = NIBUtils.get_img(img_path).affine
    result = result.swapaxes(2, 3).swapaxes(3, 4)
    result = 1 - result[0, 0, :].cpu().numpy()

    original_pixel = result.shape
    # Resample image
    target_pixel = org_arr.shape
    target_ratio = [(original_pixel[i] / target_pixel[i]) for i in range(len(original_pixel))]
    transform = tio.Resample(target=target_ratio, image_interpolation="bspline")
    img_resampled = transform(np.expand_dims(result, axis=0))[0]

    nib_img = nib.Nifti1Image(img_resampled, affine=affine)
    nib.save(nib_img, out_grad_path)


if __name__ == '__main__':
    # Initialize argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the .pt model.")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the .nii.gz image for which GradCAM++ map has to be created.")
    parser.add_argument("--output_path", type=str, required=True, help="Path of the output Nifty file.")
    parser.add_argument("--layer", type=str, default="layer2", help="Layer name used to generate GradCAM++ map.")
    parser.add_argument("--device", type=str, default="cuda", help="Device where to perform computations.")
    parser.add_argument("--overwrite", action="store_true", help="If set to True, overwrites the output")
    args = parser.parse_args()
    # Extract arguments into variables
    model_path = args.model_path
    output_path = args.output_path
    image_path = args.image_path

    if os.path.exists(output_path) and not args.overwrite:
        print("[INFO] GradCam already exists. You can set overwrite to True with `--overwrite`")
        sys.exit(0)

    layer = args.layer
    device = torch.device(args.device)
    # Initialize the model and load its state dict from the specified path
    net = load_model(args.model_path, device)
    net = net.eval()  # Set the model to evaluation mode
    # Create the output directory if it doesn't exist
    directory_path = os.path.dirname(output_path)
    if directory_path:
        os.makedirs(directory_path, exist_ok=True)
    # Generate the Grad-CAM++ heatmap and save it to the specified output path
    get_grad_cam(output_path, layer, net, image_path)
