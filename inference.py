import torch
import numpy as np

import os
from argparse import ArgumentParser
from os.path import join

import pandas as pd
from deep_utils import DirUtils
from collections import defaultdict
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from preprocessing_cropping import process_sample
from utils import load_model, normalize_and_resample


def get_resampled_torch_img(img_path: str, resampling_shape: tuple[int, int, int] = (215, 215, 85)) -> torch.Tensor:
    """
    Get resampled torch tensor from cropped image!
    :param img_path:
    :param resampling_shape:
    :return:
    """
    img = normalize_and_resample(img_path, resampling_shape)
    img = torch.Tensor(img).to(torch.float32)
    last_id = len(img.shape) - 1
    img = img.swapaxes(last_id - 1, last_id).swapaxes(last_id - 2, last_id - 1)
    img = img.unsqueeze(0)
    return img


def remove_nnunet_artifact(img_path: str):
    """
    Remove nnunet artifacts
    :param img_path:
    :return:
    """
    img_dir = os.path.dirname(img_path)
    if os.path.exists(join(img_dir, "dataset.json")):
        os.remove(join(img_dir, "dataset.json"))
    if os.path.exists(join(img_dir, "plans.json")):
        os.remove(join(img_dir, "plans.json"))
    if os.path.exists(join(img_dir, "predict_from_raw_data_args.json")):
        os.remove(join(img_dir, "predict_from_raw_data_args.json"))


def get_predictions(sample_path: str,
                    model_paths: list[str],
                    threshold: float,
                    device: torch.device | str,
                    is_cropped: bool,
                    seg_predictor,
                    keep_crops: bool = False):
    """
    Generates predictions from an ensemble of models and computes metrics based on input threshold.
    :param sample_path: Path to the input sample (single image or directory).
    :param model_paths: List of paths to the trained models to be evaluated.
    :param device: Computation device (e.g. 'cuda' or 'cpu')
    :param threshold: threshold to determine the class
    :param is_cropped:
    :param seg_predictor: nnunet model
    :param keep_crops: keep the cropped samples
    :return: tuple:
            - names (list[str]): Names of the samples processed.
            - model_predictions (dict): Predictions from each model in the ensemble.
            - mean_predictions (list[float]): Mean probability predictions from the ensemble.
    """
    full_paths = get_file_paths(sample_path)
    models = [load_model(model_path, device).eval() for model_path in model_paths]
    model_predictions = defaultdict(list)
    mean_predictions = dict()
    with torch.no_grad():
        for path in full_paths:
            print(f"[INFO] Inferencing on {path}")
            img_name = path
            if not is_cropped:
                path = crop_sample(path, seg_predictor, keep_crops)
                img = get_resampled_torch_img(path)
                if not keep_crops:
                    os.remove(path)
                remove_nnunet_artifact(path)
            else:
                img = get_resampled_torch_img(path)
            ensemble_predictions = []
            for index, net in enumerate(models):
                logits = net(img[None, ...].to(device))
                pred = torch.sigmoid(logits[0, 0]).cpu().numpy()
                model_predictions[index].append([img_name, pred, 1 if pred > threshold else 0])
                ensemble_predictions.append(pred)
            mean_prediction = np.mean(ensemble_predictions, axis=0)
            mean_predictions[img_name] = (mean_prediction, 1 if mean_prediction > threshold else 0)

    return mean_predictions, model_predictions


def get_file_paths(dataset_root_path: str) -> list[str]:
    """
    Get file paths.
    :param dataset_root_path:
    :return:
    """
    full_paths = []
    if os.path.isdir(dataset_root_path):
        for dirpath, dirnames, filenames in os.walk(dataset_root_path):
            for name in filenames:
                if name.endswith(".nii.gz"):
                    full_paths.append(join(dirpath, name))
    else:
        full_paths.append(dataset_root_path)
    return full_paths


def crop_sample(input_path, seg_predictor, keep_crops: bool) -> str:
    """
    First segmentation is done, then based on that samples are cropped!
    :param input_path:
    :param seg_predictor:
    :param keep_crops:
    :return:
    """
    seg_path = DirUtils.split_extension(input_path, suffix="_seg", current_extension=".nii.gz")
    cropped_seg_path = DirUtils.split_extension(seg_path, suffix="_cropped", current_extension=".nii.gz")
    cropped_img_path = DirUtils.split_extension(input_path, suffix="_cropped", current_extension=".nii.gz")
    print(f"{input_path} and {seg_path}")
    seg_predictor.predict_from_files(
        [[input_path]],
        [seg_path],
        save_probabilities=False,
        overwrite=False,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0)
    process_sample(seg_path, cropped_img_path, cropped_seg_path, input_path, (8, 8, 6),
                   2, 3, (-1, 1, 1))
    if not keep_crops:
        os.remove(cropped_seg_path)
        os.remove(seg_path)

    return cropped_img_path


if __name__ == "__main__":
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("--model_path", help="path to model or models for ensemble", nargs="+", type=str)
    parser.add_argument("--sample_path", help="path to sample or samples directory.")
    parser.add_argument("--output_path", help="Directory to put the outputs", default="output")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="classification threshold")
    parser.add_argument("--seg_model_path", type=str, default="nnunet_segmentation/cardiac_segmentation",
                        help="Path to cardiac segmentation models")
    parser.add_argument("--seg_folds", type=int, nargs="+", default=(0, 1, 2, 3, 4),
                        help="folds to use in segmentation, default is (0, 1, 2, 3, 4)")
    parser.add_argument("--device", default="cuda",
                        help="device to run the model on")
    parser.add_argument("--is_cropped", action="store_true",
                        help="Whether the input data is cropped! Default is set to not cropped!")
    parser.add_argument("--seg_checkpoint_name", default="checkpoint_final.pth",
                        help="checkpoint_final.pth or checkpoint_best.pth")
    parser.add_argument("--keep_crops", action="store_true", help="If set to True, keeps the crops and segmentations")
    args = parser.parse_args()
    args.device = "cpu" if not torch.cuda.is_available() else args.device
    # Create the output directory if it does not exist
    os.makedirs(args.output_path, exist_ok=True)

    # Clear GPU memory if using CUDA
    if "cuda" in args.device:
        torch.cuda.empty_cache()
    if not args.is_cropped:
        seg_predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device(args.device),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        # initializes the network architecture, loads the checkpoint
        seg_predictor.initialize_from_trained_model_folder(
            args.seg_model_path,
            use_folds=args.seg_folds,
            checkpoint_name=args.seg_checkpoint_name,
        )
    else:
        seg_predictor = None

    # Generate predictions for the given samples using the specified models and thresholds
    mean_predictions, model_predictions = get_predictions(args.sample_path, model_paths=args.model_path,
                                                          threshold=args.threshold, device=args.device,
                                                          is_cropped=args.is_cropped,
                                                          seg_predictor=seg_predictor,
                                                          keep_crops=args.keep_crops)

    # Save individual model predictions to separate directories
    os.makedirs(args.output_path, exist_ok=True)
    df = pd.DataFrame([[img_name, pred, cls] for img_name, (pred, cls) in mean_predictions.items()],
                      columns=["name", "pred", "class"])
    df.to_csv(join(args.output_path, f"mean_prediction.csv"), index=False)

    for model_index, predictions in model_predictions.items():
        df = pd.DataFrame(predictions, columns=["name", "pred", "class"])
        df.to_csv(join(args.output_path, f"model_{model_index}_prediction.csv"), index=False)
