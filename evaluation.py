import os
import pprint
from argparse import ArgumentParser
from os.path import join

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from deep_utils import JsonUtils

from dataset import load_test
from train import evaluate_model
from utils import load_model
from evaluation_core import compute_metrics


def evaluate_single_model(test_loader: DataLoader, net: nn.Module, device: torch.device | str,
                          output_path: str) -> pd.DataFrame:
    """
    Evaluates a model on a given dataset and saves the results.
    :param test_loader: Path to the dataset for evaluation.
    :param net: Network to be evaluated.
    :param device: Device to run the evaluation on (e.g., "cuda" or "cpu").
    :param output_path: Directory to save evaluation results, including metrics and predictions.
    :return: pd.DataFrame with predictions. The function also outputs the evaluation results to files.
    """
    metrics, y_true, y_pred, _ = evaluate_model(net, test_loader, device=device)
    os.makedirs(output_path, exist_ok=True)
    JsonUtils.dump(join(output_path, "metrics.json"), metrics)
    predictions_df = pd.DataFrame({"filename": test_loader.dataset.img_names,
                                   "label": y_true,
                                   "prediction": y_pred})
    predictions_df.to_csv(join(output_path, "predictions.csv"), index=False)
    pprint.pprint(metrics)
    return predictions_df


def evaluate_ensemble(predictions: pd.DataFrame, output_path: str, threshold: float = 0.5):
    """
    Evaluates an ensemble of models given a dataframe of single predictions.
    :param predictions: pd.DataFrame with predictions of all models
    :param output_path: Directory to save evaluation results, including metrics and predictions.
    :return: None. Ensemble metrics are saved in the output directory.
    """
    # Compute ensemble prediction (mean)
    pred_cols = [col for col in predictions.columns if col not in ["filename", "label"]]
    predictions["ensemble"] = predictions[pred_cols].mean(axis=1)
    # Compute ensemble metrics
    metrics_ens = compute_metrics(y_true=predictions["label"], y_pred=predictions["ensemble"], thresh=threshold)
    for key in metrics_ens.keys():
        metrics_ens[key] = float(metrics_ens[key])
    # Save results
    os.makedirs(output_path, exist_ok=True)
    predictions.to_csv(join(output_path, "predictions.csv"), index=False)
    JsonUtils.dump(join(output_path, "metrics.json"), metrics_ens)


def evaluate_all_models(dataset_path: str, model_paths: list[str], device: torch.device | str, output_path: str,
                        batch_size: int = 2):
    """
    Evaluates a model on a given dataset and saves the results.
    :param dataset_path: Path to the .npz dataset for evaluation.
    :param model_paths: Path to the pre-trained models to be evaluated.
    :param device: Device to run the evaluation on (e.g., "cuda" or "cpu").
    :param output_path: Directory to save evaluation results, including metrics and predictions.
    :param batch_size: Batch size for loading the dataset (default 2).
    :return: pd.DataFrame with predictions. The function also outputs the evaluation results to files.
    """
    test_loader = load_test(dataset_path, batch_size=batch_size)  # load test set from .npz file
    if len(model_paths) == 1:
        # Evaluate single model
        net = load_model(device=device, model_path=model_paths[0])
        evaluate_single_model(test_loader, net, device, output_path)
    else:
        # Evaluate single models and ensemble
        predictions_all = []
        for i, model_path in enumerate(model_paths):
            output_path_i = join(args.output_path, f"model_{i}")
            net = load_model(device=device, model_path=model_path)
            predictions = evaluate_single_model(test_loader, net, device, output_path_i)
            predictions.rename(mapper={"prediction": f"prediction_{i}"}, inplace=True, axis="columns")
            predictions_all.append(predictions)
        predictions_df = predictions_all[0]
        for df in predictions_all[1:]:
            predictions_df = predictions_df.merge(df, on=["filename", "label"])
        # Compute ensemble metrics and save results
        output_path = join(args.output_path, f"ensemble")
        evaluate_ensemble(predictions_df, output_path)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("--model_path", help="Path to model or models for ensemble", nargs="+", type=str)
    parser.add_argument("--dataset_path", help="Path to numpy preprocessed dataset")
    parser.add_argument("--output_path", default="evaluation_output", help="Directory to put the outputs")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5], help="Classification thresholds")
    parser.add_argument("--device", default="cuda", help="Device to run the model on")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch Size")
    args = parser.parse_args()

    # Clear GPU memory if using CUDA
    if "cuda" in args.device:
        torch.cuda.empty_cache()

    # Evaluate the model using provided arguments
    evaluate_all_models(dataset_path=args.dataset_path, model_paths=args.model_path, device=args.device,
                        output_path=args.output_path)
