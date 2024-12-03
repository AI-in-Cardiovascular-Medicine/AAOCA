import os
from argparse import ArgumentParser
from os.path import join, split

import numpy as np
import torch
from deep_utils import DirUtils

import dataset
from utils import load_model


def get_features(model_path: str, dataset_path: str, device: str, threshold: float = 0.5):
    """

    :param model_path: path to the model
    :param dataset_path: path to numpy preprocessed dataset
    :param device: device on which to run the model
    :param threshold: threshold for the classification
    :return:
    """
    device = torch.device(device)
    train_dataset = dataset.CTDataset3D(dataset_path, augm_transform=None, get_img_names=True)

    net = load_model(model_path=model_path, device=device).eval()
    features_list = []
    features_names = []
    labels = []
    predicted_lbl = []
    with torch.no_grad():
        for img, lbl, img_name in train_dataset:
            features = net.features(img[None, ...].to(device))
            logits = net(img[None, ...].to(device))
            features_list.append(features[0].cpu().numpy().reshape(-1))
            features_names.append(img_name)
            labels.append(lbl)
            predicted_lbl.append(1 if logits[0, 0].item() > threshold else 0)

    return features_names, np.array(features_list), labels, predicted_lbl


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", help="Path to model", type=str)
    parser.add_argument("--suffix", default="_features", help="suffix to save the extracted features", type=str)
    parser.add_argument("--dataset_path", nargs="+", help="Path to numpy preprocessed dataset")
    parser.add_argument("--output_path", default="latent_features",
                        help="Directory to put the outputs",
                        )

    parser.add_argument("--device", default="cuda", help="Device to run the model on")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    for path in args.dataset_path:
        if "cuda" in args.device:
            torch.cuda.empty_cache()
        features_names, features_list, labels, predicted_lbl = get_features(args.model_path, path, args.device)
        npz_path = join(args.output_path, DirUtils.split_extension(split(path)[-1], suffix=args.suffix))
        # save the final extracted features!
        np.savez_compressed(npz_path,
                            names=features_names,
                            features=features_list,
                            labels=labels,
                            predicted_lbl=predicted_lbl)
