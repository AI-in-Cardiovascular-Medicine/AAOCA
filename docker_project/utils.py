import time
import os
from functools import partial

import numpy as np
import torch
import torchio as tio
from deep_utils import PyUtils, NIBUtils
from deep_utils import JsonUtils
from os.path import join, split
from monai.networks.nets import SENet154
import zipfile
from deep_utils import DirUtils
from torch import nn
from monai.visualize.class_activation_maps import GradCAMpp
from preprocessing_cropping import process_sample
import nibabel as nib


def dicom2nifti(nifti_dir: str, dicom_zip: str):
    dicom_dir = join(nifti_dir, split(DirUtils.split_extension(dicom_zip, suffix="_dicom").replace(".zip", ""))[-1])
    with zipfile.ZipFile(dicom_zip, 'r') as zip_ref:
        zip_ref.extractall(dicom_dir)
    os.makedirs(nifti_dir, exist_ok=True)
    command = (f'dcm2niix -z y -b y -ba n -f %f_%t_%p_%s_%d_%i_%e_%q_%z_%m_%a_%g -o'
               f' "{nifti_dir}" "{dicom_dir}"')
    output = os.system(command)
    if output != 0:
        raise ValueError(f"Input Dicom directory is not valid! nifti_dir: {nifti_dir}, dicom_dir: {dicom_dir}")
    img_path = DirUtils.list_dir_full_path(nifti_dir, interest_extensions=".gz")[0]
    return img_path


def remove_file(path: str) -> None:
    os.unlink(path)
    print(f"Removed the input: {path}")


def get_grad_cam(out_grad_path: str, net: nn.Module, img_path: str, device: str, layer_name: str = "layer2"):
    img = normalize_and_resample(img_path)
    img = torch.Tensor(img).to(torch.float32)
    last_id = len(img.shape) - 1
    img = img.swapaxes(last_id - 1, last_id).swapaxes(last_id - 2, last_id - 1)
    img = img.unsqueeze(0).unsqueeze(0)
    cam = GradCAMpp(nn_module=net, target_layers=[layer_name])
    result = cam(x=img.to(device))

    affine = NIBUtils.get_img(img_path).affine
    # input_np = img.swapaxes(2, 3).swapaxes(3, 4)
    # input_np = input_np[0, 0, :].cpu().numpy()
    result = result.swapaxes(2, 3).swapaxes(3, 4)
    result = 1 - result[0, 0, :].cpu().numpy()
    os.makedirs(os.path.dirname(out_grad_path), exist_ok=True)
    # nib_img = nib.Nifti1Image(input_np, affine=affine)
    # nib.save(nib_img, DirUtils.split_extension(out_grad_path, suffix="_main", current_extension=".nii.gz"))
    nib_img = nib.Nifti1Image(result, affine=affine)
    nib.save(nib_img, out_grad_path)


def load_model(model_path: str, device: torch.device | str, config: dict = None) -> SENet154:
    """
    Loads a pre-trained model from the specified path and prepares it for evaluation.
    :param model_path: Path to the model's weights file (.pth). The same folder must contain the relative configuration
    file 'config.json'.
    :param device: Device to load the model onto (e.g., "cuda" or "cpu").
    :param config: config file
    :return: Model loaded with the specified weights and moved to the given device.
    """
    if config is None:
        config_path = join(split(model_path)[0], "config.json")
        if not os.path.exists(config_path):
            config = dict(dropout=0)
        else:
            config = JsonUtils.load(config_path)
    net = SENet154(spatial_dims=3,  # number of spatial dimensions
                   in_channels=1,
                   num_classes=1,  # number of output nodes
                   dropout_prob=config["dropout"]).to(device).eval()
    try:
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
    except:
        # Due to change in the library during our develop
        weights = torch.load(model_path, map_location="cpu")
        weights['last_linear.weight'] = weights.pop('loss_func_layer.fc.weight')
        weights['last_linear.bias'] = weights.pop('loss_func_layer.fc.bias')
        net.load_state_dict(weights)
        torch.save(weights, model_path)
    net.to(device)
    return net


def torch_io_resize(img: np.ndarray, target_size: tuple[int, ...]) -> np.ndarray:
    """
    Resample/Resize input image using b-spline/bicubic interpolation method.
    :param img: Image array to be processed.
    :param target_size: Tuple with target size (x, y, z), in pixels.
    :return: Resampled image
    """
    original_pixel = img.shape
    target_ratio = [(original_pixel[i] / target_size[i]) for i in range(len(original_pixel))]
    transform = tio.Resample(target=target_ratio, image_interpolation="bspline")
    img_resampled = transform(np.expand_dims(img, axis=0))[0]
    return img_resampled


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Code adapted from https://github.com/Bjarten/early-stopping-pytorch.

    Attributes:
        patience (int): Number of epochs to wait for an improvement before stopping.
        verbose (bool): If True, prints messages when validation loss improves.
        delta (float): Minimum change in validation loss to qualify as an improvement.
        path (str): Filepath to save the model when validation loss improves.
        trace_func (callable): Function for printing trace messages (default uses a custom print function).

    Methods:
        __call__(val_loss, model):
            Checks if validation loss improves; saves the model if it does.
            Stops training if no improvement is observed for 'patience' epochs.

        save_checkpoint(val_loss, model):
            Saves the model when validation loss improves.
    """

    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt',
                 trace_func=partial(PyUtils.print, mode="bold")):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}', color="yellow")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...',
                color="red", )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def make_output_folder(config: dict, overwrite: bool = True):
    """
    Creates an output folder for storing results based on the directory path defined in the configuration.
    If the specified folder already exists, a unique folder is created by appending an incremental number
    (e.g., "output_dir_1", "output_dir_2", etc.).
    :param config: Dictionary containing configuration parameters. It must include the key "output_dir".
    :param overwrite: Whether to overwrite the folder in case it already exists (default True)
    :return: A string representing the path to the newly created or unique output directory.
    """
    base_dir = os.path.normpath(config["output_dir"])
    path_res = base_dir
    if os.path.exists(path_res) and (not overwrite):
        # If dir already exists, create new dir with an incremented number (e.g., output_dir_1, output_dir_2, ...)
        i = 1
        path_res = f"{base_dir}_{i}"
        while os.path.exists(path_res):
            i = i + 1
            path_res = f"{base_dir}_{i}"
    os.makedirs(path_res, exist_ok=True)  # Create the new directory with the unique name
    return path_res


def normalize_and_resample(img_path: str, target_pixel: tuple[int, int, int] = (215, 215, 85)):
    """
    Normalizes and resamples a `.nii.gz` image to the target shape.
    :param img_path: Path to the `.nii.gz` image file.
    :param target_pixel: Target shape for resampling.
    :return: Normalized and resampled 3D image np.ndarray.
    """
    # load image
    arr, img = NIBUtils.get_array_img(img_path)
    arr = arr.astype(np.float32)
    original_pixel = img.header["dim"][1:4]
    # Resample image
    target_ratio = [(original_pixel[i] / target_pixel[i]) for i in range(len(original_pixel))]
    transform = tio.Resample(target=target_ratio, image_interpolation="bspline")
    img_resampled = transform(np.expand_dims(arr, axis=0))[0]
    img_resampled = img_resampled[:int(target_pixel[0]), :int(target_pixel[1]), :int(target_pixel[2])]
    img_resampled_normalized = np.clip(img_resampled, a_min=-1024, a_max=1024)
    img_resampled_normalized = np.array((img_resampled_normalized + 1024) / 2048 * 255).astype(int) / 255
    return img_resampled_normalized


def process_cropped_sample(img_path: str | np.ndarray, threshold: float, device, anomaly_model, origin_model,
                           risk_model):
    tic = time.time()
    img = normalize_and_resample(img_path)
    img = torch.Tensor(img).to(torch.float32)
    last_id = len(img.shape) - 1
    img = img.swapaxes(last_id - 1, last_id).swapaxes(last_id - 2, last_id - 1)
    img = img.unsqueeze(0).to(device)
    if len(img.shape) == 4:
        img = img.unsqueeze(0)
    anomaly_percentage = torch.sigmoid(anomaly_model(img)[0]).item()
    anomaly = 1 if anomaly_percentage > threshold else 0
    if anomaly == 1:
        origin = torch.sigmoid(origin_model(img)[0]).item()
        risk = torch.sigmoid(risk_model(img)[0]).item()
    else:
        origin = 0
        risk = 0
    print(f"[INFO] {time.time() - tic} to get the output")
    return dict(anomaly=anomaly, risk=risk, origin=origin)


def main_inference(input_path: str,
                   output_dir: str,
                   is_cropped: bool,
                   is_nifti: bool,
                   threshold: float,
                   anomaly_model: nn.Module,
                   origin_model: nn.Module,
                   risk_model: nn.Module,
                   segmentation_model,
                   device: str):
    os.makedirs(output_dir, exist_ok=True)
    if not is_nifti:
        input_path = dicom2nifti(output_dir, input_path)

    with torch.no_grad():
        if is_cropped:
            output = process_cropped_sample(input_path, threshold, device, anomaly_model, origin_model, risk_model)
        else:
            seg_path = DirUtils.split_extension(input_path, suffix="_seg", current_extension=".nii.gz")
            cropped_seg_path = DirUtils.split_extension(seg_path, suffix="_cropped", current_extension=".nii.gz")
            cropped_img_path = DirUtils.split_extension(input_path, suffix="_cropped", current_extension=".nii.gz")
            print(f"{input_path} and {seg_path}")
            segmentation_model.predict_from_files(
                [[input_path]],
                [seg_path],
                save_probabilities=False,
                overwrite=False,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
                folder_with_segs_from_prev_stage=None,
                num_parts=1,
                part_id=0)
            # let's crop it :)
            process_sample(seg_path, cropped_img_path, cropped_seg_path, input_path, (8, 8, 6),
                           2, 3, (-1, 1, 1))
            output = process_cropped_sample(cropped_img_path, threshold, device, anomaly_model, origin_model,
                                            risk_model)
    return output
