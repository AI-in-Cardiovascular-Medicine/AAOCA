import os
from functools import partial
from os.path import join, split

import numpy as np
import torch
import torchio as tio
from deep_utils import JsonUtils
from deep_utils import PyUtils, NIBUtils
from monai.networks.nets import SENet154


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
