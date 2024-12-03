from os.path import join
import argparse
import os
import json
import pprint
import time
import traceback

import numpy as np
import pandas as pd
import torch
from deep_utils import JsonUtils
from torch import nn
from torch.utils.data import DataLoader
from monai.networks.nets import SENet154
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_recall_fscore_support

from dataset import load_train, load_test
from utils import EarlyStopping, make_output_folder

matplotlib.use('Agg')
plt.ioff()


def train_epoch(model: nn.Module, train_loader: DataLoader, loss_fn: nn.modules.loss._Loss,
                optimizer: torch.optim.Optimizer, device: torch.device | str, max_norm: float):
    """
    Train the model for one epoch using the provided training data loader.
    :param model: pyTorch model to be trained.
    :param train_loader: DataLoader for train dataset.
    :param loss_fn: Loss function.
    :param optimizer: Optimizer.
    :param device: Device where to perform computations.
    :param max_norm: Maximum norm for gradient clipping.
    :return: Average loss across all batches in the epoch.
    """
    model.train()  # Training mode
    train_losses_epoch = []  # list to track the training loss of each batch in the epoch
    for imgs, labels in train_loader:  # loop over all batches in training
        imgs = imgs.to(device)  # Move data and labels to the desired device
        labels = labels.to(device)
        optimizer.zero_grad()  # reset the gradients
        phi = model(imgs)
        loss = loss_fn(phi, labels[:, None].float())  # calculate loss on current batch
        loss.backward()  # backward pass
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()  # update the value of the params
        train_losses_epoch.append(loss.item())
    return np.average(train_losses_epoch)  # return the average loss of the epoch


def validation_epoch(model: nn.Module, val_loader: DataLoader, loss_fn: nn.modules.loss._Loss,
                     device: torch.device | str = "cuda"):
    """
    Evaluates the model for one epoch using the provided validation data loader.
    :param model: pyTorch model.
    :param val_loader: DataLoader for validation.
    :param loss_fn: Loss function.
    :param device: Device where to perform computations (default 'cuda').
    :return: Average loss across all batches in the epoch.
    """
    model.eval()  # Model in validation mode
    val_losses_epoch = []  # list to track validation loss
    with torch.no_grad():
        for imgs, labels in val_loader:  # loop over all batches in dataloader
            imgs = imgs.to(device)  # Move data and labels to the desired device
            labels = labels.to(device)
            # get the predictions and calculate loss for the current batch
            phi = model(imgs)
            loss = loss_fn(phi, labels[:, None].float())
            val_losses_epoch.append(loss.item())
    return np.average(val_losses_epoch)  # return the average loss of the epoch


def evaluate_model(model: nn.Module, loader: DataLoader, threshold: float = 0.5,
                   device: torch.device | str = "cuda"):
    """
    Evaluates the performance of a trained model on a given dataset and computes various metrics.
    :param model: Trained neural network to evaluate.
    :param loader: DataLoader providing batches of data for evaluation.
    :param threshold: Decision threshold for classifying predictions (default 0.5).
    :param device: Device on which to perform evaluation (default: "cuda").
    :return:
        - metrics (dict): A dictionary containing the computed evaluation metrics, with the following keys:
            - "roc_auc": ROC-AUC score (threshold-independent).
            - "threshold": A nested dictionary with threshold-dependent metrics (e.g., sensitivity, specificity).
        - y_true (numpy.ndarray): Ground truth labels across all batches.
        - y_pred (numpy.ndarray): Model predictions (probabilities after sigmoid activation).
        - threshold (float): The decision threshold used for threshold-dependent metrics.
    """
    model = model.to(device)
    model.eval()
    y_pred = []
    y_true = []
    # Get model logit predictions and ground-truth labels on all batches
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            phi = model(imgs)
            y_pred.append(phi)  # update the loss meter
            y_true.append(labels.cpu().numpy())
    # Get predicted probabilities (apply sigmoid)
    y_pred = torch.sigmoid(torch.cat(y_pred).squeeze(1)).cpu().numpy()
    y_true = np.concatenate(y_true, axis=0)

    # compute AUC
    metrics = {"roc_auc": None, "threshold": {}}
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred)
    except Exception as e:
        print(f"AUC not computed, error: {str(e)}")

    # Threshold-dependent metrics
    try:
        precision, recall, f1score, _ = precision_recall_fscore_support(y_true, y_pred > threshold, average='binary')
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred > threshold).ravel()
        specificity = tn / (tn + fp)
        accuracy = accuracy_score(y_true, y_pred > threshold)
    except Exception as e:
        print(f"Threshold-dependent metrics not computed, error: {str(e)}")
        recall, specificity, f1score, precision, accuracy = [None] * 5
    # Store metrics
    metrics["threshold"][str(threshold)] = {
        "sensitivity": recall,
        "specificity": specificity,
        "f1-score": f1score,
        "ppv": precision,
        "accuracy": accuracy
    }
    return metrics, y_true, y_pred, threshold


def evaluate_model_on_all_sets(net: nn.Module, train_loader: DataLoader,
                               val_loader: DataLoader, config: dict, model_choice: str,
                               device: torch.device | str = "cuda"):
    """
    Evaluates the model on training, validation, and test datasets, saving predictions and metrics.
    :param net: Trained neural network to evaluate.
    :param train_loader: DataLoader for the training dataset.
    :param val_loader: DataLoader for the validation dataset.
    :param config: Configuration dictionary containing model, data, and evaluation settings.
    :param model_choice: Identifier for the current model being evaluated (e.g. "best" or "last").
    :param device: Device to use for computation (default 'cuda').
    :return: A dictionary of performance metrics for all datasets.
    """
    # validation
    metrics_val, y_true, y_pred, optimal_threshold = evaluate_model(net, val_loader, device=device)
    predictions_df = pd.DataFrame({"filename": val_loader.dataset.img_names,
                                   "label": y_true,
                                   "prediction": y_pred})
    predictions_df.to_csv(join(config["output_dir"], "val_predictions_" + model_choice + ".csv"), index=False)
    # train
    metrics_train, _, _, _ = evaluate_model(net, train_loader, device=device, threshold=optimal_threshold)
    metrics = {
        "train": metrics_train,
        "val": metrics_val
    }
    # tests
    for path in config["roots_test"]:
        name = os.path.basename(path).split(".")[0]
        test_loader = load_test(path)
        metrics[name], y_true, y_pred, _ = evaluate_model(net, test_loader, device=device, threshold=optimal_threshold)
        predictions_df = pd.DataFrame({"filename": test_loader.dataset.img_names,
                                       "label": y_true,
                                       "prediction": y_pred})
        predictions_df.to_csv(join(config["output_dir"], name + "_predictions_" + model_choice + ".csv"), index=False)
    return metrics


def train_net(config):
    """
    Train a model with a given configuration.
    :param config:  A dictionary containing all configurations.
    :return: None
    """
    device = torch.device(config["device"])
    train_loader, val_loader = load_train(config["root_train"],
                                          batch_size=config["batch_size"],
                                          augm_list=config["augmentation"],
                                          augm_params=config["augm_params"],
                                          num_workers=config['num_workers'],
                                          val_fraction=config['val_fraction']
                                          )
    net = SENet154(spatial_dims=3,  # number of spatial dimensions
                   in_channels=train_loader.dataset[0][0].shape[0],
                   num_classes=1,  # number of output nodes
                   dropout_prob=config["dropout"]).to(device)
    if config["pretrained_path"] is not None:  # Load pre-trained model if required
        print(f"Loading pretrained network from {config['pretrained_path']}")
        net.load_state_dict(torch.load(config["pretrained_path"], map_location=device))
    optimizer = torch.optim.AdamW(net.parameters(), config["lr"], weight_decay=config["weight_decay"])
    loss_function = torch.nn.BCEWithLogitsLoss()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["t_max"], eta_min=1e-6)

    path_res = config["output_dir"]
    path_checkpoint = os.path.join(path_res, "best_val_model.pt")
    early_stopping = EarlyStopping(patience=config["patience"], verbose=True, path=path_checkpoint)

    # -- Model training --
    train_loss_epochs = []
    val_loss_epochs = []
    start_time = time.time()
    for epoch in range(1, config["n_epochs"] + 1):
        tic = time.time()
        train_loss = train_epoch(net, train_loader, loss_function, optimizer, device, max_norm=config["max_norm"])
        train_time = time.time() - tic
        tic = time.time()
        val_loss = validation_epoch(net, val_loader, loss_function, device)
        val_time = time.time() - tic
        print(f"Epoch {epoch} completed. Train loss: {round(train_loss, 4)}, Val loss: {round(val_loss, 4)},"
              f" lr: {round(optimizer.param_groups[0]['lr'], 6)},"
              f" train_time: {round(train_time, 1)}s, val_time: {round(val_time)}s"
              f"\n{config['output_dir']}")
        if str(train_loss) == "nan":
            raise ValueError("Faced nan in train")
        if str(val_loss) == "nan":
            raise ValueError("Faced nan in val")
        train_loss_epochs.append(train_loss)
        val_loss_epochs.append(val_loss)
        # Learning rate decay
        if lr_scheduler is not None:
            lr_scheduler.step()
        # early_stopping:
        early_stopping(val_loss, net)  # if validation loss has decreased, make a checkpoint of the current model
        if early_stopping.early_stop:  # stop if validation loss doesn't improve after a given patience
            print("Early stopping...")
            break
        # save loss graph in each epoch
        fig = plt.figure()
        plt.plot(train_loss_epochs)
        plt.plot(val_loss_epochs)
        fig.savefig(join(path_res, "loss_epochs.png"))
        plt.close()
        torch.save(net.state_dict(), join(path_res, "last_model.pt"))
    print("\nElapsed time: ", time.time() - start_time)
    # save train/val losses
    np.save(join(path_res, "train_losses"), train_loss_epochs)
    np.save(join(path_res, "val_losses"), val_loss_epochs)

    # -- Model Evaluation --
    print("Finished Training and Evaluating")
    train_loader.dataset.augment = False
    val_loader.dataset.augment = False
    # last model
    metrics = evaluate_model_on_all_sets(net, train_loader, val_loader, config, device=device, model_choice="last")
    JsonUtils.dump(join(path_res, "metrics_last.json"), metrics)
    print("LAST model")
    pprint.pprint(metrics)
    # best model on validation
    net.load_state_dict(torch.load(path_checkpoint))
    metrics = evaluate_model_on_all_sets(net, train_loader, val_loader, config, device=device, model_choice="best")
    JsonUtils.dump(join(path_res, "metrics_best.json"), metrics)
    print("BEST model on validation")
    pprint.pprint(metrics)
    # save the optimization metrics
    torch.save(optimizer.state_dict(), join(path_res, "last_optimizer.pt"))
    torch.save(lr_scheduler.state_dict(), join(path_res, "last_scheduler.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for mode train evaluation")
    parser.add_argument("--bs", type=int, default=8, help="Batch size of training data")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=100, help="Early-stopping patience")
    parser.add_argument("--epochs", type=int, default=300, help="Max number of epochs")
    parser.add_argument("--augm_prob", type=float, default=0.15, help="Augmentation probability")
    parser.add_argument("--train_on", type=str, required=True,
                        help="Path to the .npz file containing the dataset for training and validation")
    parser.add_argument("--test_on", nargs="+", type=str, help="Path to the .npz files with the datasets for testing")
    parser.add_argument("--output_path", type=str, help="Path of the output directory")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Pretrained model path.")
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Validation set fraction")
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction,
                        help="Whether to overwrite results or to create a new directory")
    args = parser.parse_args()

    architecture = "senet154_3d"
    device = args.device
    pretrained_path = args.pretrained_path

    dropout = 0.3
    epochs = args.epochs
    patience = args.patience
    augmentations = ["Noise", "Gamma", "Blur"]
    augm_params = {
        "p": args.augm_prob,
        "log_gamma": 0.3,
        "noise_mean": 0.,
        "noise_std": 0.02,
        "blur_std": (2., 2., 2.),
        "biasfield": 0.05,
    }
    root_train = args.train_on
    test_on = args.test_on
    print(f"Traning on: {root_train}")

    augm_str = ("".join(augmentations) or "noAugm") + "_monai"
    print(f"\n+++ Starting a new training +++")
    try:
        config = {
            "root_train": root_train,
            "roots_test": test_on,
            "batch_size": args.bs,
            "t_max": int(epochs * (3 / 4)),  # for 1/3 part of the flow the minimum lr will be applied (1e-6)
            "architecture": architecture,
            "dropout": dropout,
            "lr": args.lr,
            "max_norm": 1,
            "num_workers": 20,
            "weight_decay": 0.01,
            "augmentation": augmentations,
            "val_fraction": args.val_fraction,
            "patience": patience,
            "n_epochs": epochs,
            "augm_params": augm_params,
            "device": device,
            "pretrained_path": pretrained_path,
            "output_dir": args.output_path
        }
        config["output_dir"] = make_output_folder(config, args.overwrite)  # Make output folder without overwriting
        print("Output folder name: ", config["output_dir"], "\n")
        with open(os.path.join(config["output_dir"], "config.json"), "w") as fp:  # save config file
            json.dump(config, fp)
        # Train
        torch.cuda.empty_cache()
        train_net(config)
    except Exception as e:
        print(
            f"An error occurred. Error message: {str(e)}, traceback: {traceback.format_exc()}")
    print(
        f"\n+++ Finished Training +++")
