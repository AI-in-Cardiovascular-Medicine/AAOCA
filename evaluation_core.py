import os
from typing import Dict, List
from os.path import join

import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from deep_utils import JsonUtils, DirUtils
from sklearn.metrics import (roc_auc_score, precision_recall_fscore_support, confusion_matrix, accuracy_score,
                             average_precision_score, ConfusionMatrixDisplay)
from roc_utils import plot_roc, plot_roc_bootstrap, compute_roc, plot_mean_roc
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(y_true: ArrayLike, y_pred: ArrayLike, thresh: float = 0.5):
    """
    Compute various performance metrics for binary classification.
    :param y_true: Ground truth (true binary labels), where each value is 0 or 1.
    :param y_pred: Predicted probabilities or scores output by the model. These values are thresholded to classify predictions.
    :param thresh: Threshold to binarize `y_pred`. Values greater than `thresh` are considered as positive predictions (1).
    :return: A dictionary containing classification metrics.
    """
    # Compute metrics
    precision, recall, f1score, _ = precision_recall_fscore_support(y_true, y_pred > thresh, average='binary')
    aupr = average_precision_score(y_true, y_pred > thresh)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred > thresh).ravel()
    specificity = tn / (tn + fp)
    # Store metrics into the output dictionary
    metrics = {
        "roc_auc": roc_auc_score(y_true, y_pred),
        "sensitivity": recall,
        "specificity": specificity,
        "f1-score": f1score,
        "ppv": precision,
        "aupr": aupr,
        "accuracy": accuracy_score(y_true, y_pred > thresh),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }
    return metrics


def ensemble_metrics(folders_path: list[str], test_dataset: str, thresh: float,
                     gender: str,
                     gender_dict: dict[str, str] | None):
    """
    Compute ensemble performance metrics and predictions for multiple models.
    :param folders_path: List of paths to the folders containing model prediction files.
    :param test_dataset: The name of the test dataset (used to find prediction files).
    :param thresh: The threshold for binarizing the model predictions.
    :param gender: If specified, filters the dataset based on the given gender.
    :param gender_dict:
    :return: A dictionary mapping filenames to gender labels. Used if gender filtering is applied.
    """
    # Initialize lists to store labels, predictions, and model metrics
    labels = []
    predictions = []
    filenames = []
    metrics_folders = []
    # Loop through each model folder to read predictions and compute metrics
    for folder_path in folders_path:
        # Load model predictions from CSV file
        df = pd.read_csv(join(folder_path, test_dataset + "_predictions_best.csv"))
        if gender:  # If gender filtering is required, apply it
            df = df[[gender_dict.get(item) == gender for item in df['filename']]]
        # Store labels and predictions from the current model
        labels.append(df["label"].values)
        predictions.append(df["prediction"].values)
        filenames.append(df["filename"].values)
        # Compute and store the model's metrics
        metrics = compute_metrics(df["label"].values, df["prediction"].values, thresh)
        metrics_folders.append(metrics)

    labels = labels[0]  # Labels are the same across all models (same test set)
    filenames = filenames[0]
    predictions = np.array(predictions).transpose()

    # Compute ensemble prediction (mean)
    mean = np.mean(predictions, axis=1)
    df_ensemble = pd.DataFrame({"filename": filenames, "label": labels, "ensemble": mean})
    df_ensemble = pd.concat(
        [df_ensemble, pd.DataFrame(predictions, columns=[f"model_{i}" for i in range(predictions.shape[1])])], axis=1)
    # Compute ensemble metrics
    metrics_ens = compute_metrics(labels, mean, thresh)
    metrics = {"roc_auc": [metrics_i["roc_auc"] for metrics_i in metrics_folders]}
    for metric in ["accuracy", "sensitivity", "specificity", "f1-score", "ppv", "aupr", "tp", "tn", "fp", "fn"]:
        metrics[metric] = [metrics_i[metric] for metrics_i in metrics_folders]
    metrics = pd.DataFrame(metrics)

    return pd.DataFrame([metrics_ens, metrics.mean().to_dict()], index=['ensemble', 'mean']).transpose(), df_ensemble


def metrics_single_models(folders_path: list[str], test_datasets: list, thresh: float,
                          genders: List[str],
                          gender_dicts: List[Dict[str, str] | None]):
    """
    Compute metrics for multiple models across different test datasets.
    :param folders_path: List of paths to the folders containing model prediction files.
    :param test_datasets: List of test dataset names (used to locate prediction files).
    :param thresh: The threshold for binarizing the model predictions.
    :param genders: List of genders to filter the datasets by, or empty string for no filtering. Inputs: f, m, ''
    :param gender_dicts: List of dictionaries mapping filenames to gender labels for filtering, or None if no filtering is required.
    :return:
    """
    # Validate that the number of gender_dicts and genders match the number of datasets
    if (gender_dicts is not None) and (len(test_datasets) != len(gender_dicts)) and (len(test_datasets) != len(genders)):
        raise ValueError()
    metrics_folders = []
    for folder_path in folders_path:  # Iterate through model folders
        metrics = {}
        for test_dataset, gender_dict, gender in zip(test_datasets, gender_dicts, genders):  # iterate through datasets
            # Load prediction and compute metrics
            df = pd.read_csv(join(folder_path, test_dataset + "_predictions_best.csv"))
            if gender:
                df = df[[gender_dict.get(item) == gender for item in df['filename']]]
            metrics[test_dataset] = compute_metrics(df["label"].values, df["prediction"].values, thresh)
        metrics_folders.append(metrics)

    # Compute average metrics and ensemble metrics
    metrics = {}
    names = [dataset for dataset in test_datasets]
    multi_index = pd.MultiIndex.from_product([names,
                                              ["roc_auc", "sensitivity", "specificity", "f1-score", "ppv", "accuracy",
                                               "aupr", "tp", "tn", "fp", "fn"]])
    for test_dataset in test_datasets:
        metrics["roc_auc" + "_" + test_dataset] = [metrics_i[test_dataset]["roc_auc"] for metrics_i in metrics_folders]
        for metric in ["sensitivity", "specificity", "f1-score", "ppv", "accuracy", "aupr", "tp", "tn", "fp", "fn"]:
            metrics[metric + "_" + test_dataset] = [metrics_i[test_dataset][metric] for metrics_i in metrics_folders]
    metrics = pd.DataFrame(metrics)
    metrics_multi = pd.DataFrame(metrics.values, columns=multi_index)
    return metrics_multi


def ensemble_metrics_for_different_thresholds(thresholds: ArrayLike, test_dataset: str, folders: list[str],
                                              gender: str, gender_dict: Dict[str, str] | None):
    """
    Compute ensemble metrics for multiple thresholds on a test dataset.
    :param thresholds: A list of thresholds to evaluate the ensemble metrics.
    :param test_dataset: Name of the test dataset (used to find prediction files).
    :param folders: List of folders containing model prediction files.
    :param gender: Gender filter to apply to the dataset, or empty string for no filtering.
    :param gender_dict:  A dictionary mapping filenames to gender labels, used for gender-based filtering.
    :return: pd.DataFrame containing ensemble metrics for each threshold, with metrics as rows and thresholds as columns.
    """
    metrics_all = []
    for thresh in thresholds:
        metrics, _ = ensemble_metrics(folders_path=folders, test_dataset=test_dataset, thresh=thresh, gender=gender,
                                      gender_dict=gender_dict)
        df = metrics.loc[
            ["sensitivity", "specificity", "f1-score", "ppv", "accuracy", "aupr", "tp", "tn", "fp", "fn"], "ensemble"]
        df.name = str(thresh)
        metrics_all.append(df)
    out_df = pd.concat(metrics_all, axis=1)
    return out_df


def add_gender(filename: str, gender: str, name: bool = False):
    """
    Append gender to a filename as a suffix or descriptive label.
    :param filename: The original filename.
    :param gender: Gender to append ('m' for male, 'f' for female, or empty for no change).
    :param name:  If True, adds a descriptive label; otherwise, adds a suffix.
    :return: The updated filename with gender appended.
    """
    if gender:
        if name:
            mapping = {"f": "Female", "m": "Male"}
            filename = f"{filename} - {mapping[gender]}"
        else:
            filename = DirUtils.split_extension(filename, suffix=f"_{gender}")
    return filename


def plot_confusion_matrix(predictions: ArrayLike, labels: ArrayLike, output_path: str | None, title: str,
                          title_size: float = 16, size: float = 11, dpi: int = 300, show: bool = True):
    """
    Plot and save a confusion matrix with customizable appearance.
    :param predictions: Predicted labels.
    :param labels: True labels.
    :param output_path: Path to save the confusion matrix plot. If None, the image is not saved.
    :param title: Title of the plot.
    :param title_size: Font size for the title and axis labels (default is 16).
    :param size: Font size for confusion matrix text (default is 11).
    :param dpi: Output figure's dpi (default is 300).
    :param show: If True, display the plot; otherwise, save and close it (default is True).
    :return:
    """
    cm = confusion_matrix(y_pred=predictions, y_true=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.PuBuGn, colorbar=False, text_kw={"fontsize": size})
    plt.title(title, fontsize=title_size)
    plt.grid(False)
    plt.tight_layout()
    plt.yticks(fontsize=title_size)
    plt.xticks(fontsize=title_size)
    disp.ax_.xaxis.label.set_fontsize(title_size)
    disp.ax_.yaxis.label.set_fontsize(title_size)
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()


def main():
    gender_info = {
        "test_internal": JsonUtils.load("internal_gender_img_names.json"),
        "test_external": JsonUtils.load("external_gender_img_names.json")
    }
    sns.set_theme(style="white")
    color_palette = matplotlib.colormaps.get_cmap('tab10').colors

    name_to_title = {
        "train": f"Train",
        "test_internal": f"Internal Test",
        "test_external": f"External Test"
    }

    n_boot = 10000
    threshold_title = "Cut-Off"
    dpi = 300

    threshold = 0.5  # Threshold used to evaluate the models
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]  # Other thresholds explored

    folders = [
        'model_1',
        'model_2',
        'model_3',
        'model_4',
        'model_5',
    ]

    # Run through task, gender and training strategy, and plot results
    for task in ["anomaly_detection", "origin_classification", "risk_classification"]:
        print(f"----- {task} ------")
        training_strategies = ["train"] if task != "anomaly_detection" else ["train", "strategy2"]
        for gender in ["", "f", "m"]:
            print(f"\t+ Gender selection: {gender if gender != '' else 'No selection'}")
            for training_strategy in training_strategies:
                print(f"\t\tTraining strategy: {training_strategy}")
                test_datasets = ["test_internal", "test_external"] if training_strategy == "train" else ["test_external"]
                path = f"results/{task}/{training_strategy}/"
                results_output_folder = f"results/{task}/{training_strategy}"
                images_output_folder = f"images/all/{task}/{training_strategy}"
                os.makedirs(results_output_folder, exist_ok=True)
                os.makedirs(images_output_folder, exist_ok=True)
                folders_path = [join(path, folder) for folder in folders]  # build folder paths

                # Compute metrics for single models and ensemble
                genders = [gender] * len(test_datasets)
                gender_dicts = [gender_info[test_dataset] for test_dataset in test_datasets]
                metrics_all_models = metrics_single_models(folders_path, test_datasets=test_datasets,
                                                           thresh=threshold, genders=genders, gender_dicts=gender_dicts)
                metrics_all_models.to_csv(join(results_output_folder, add_gender("single_models.csv", gender)), index=False)

                metrics_datasets = {}
                df_datasets = {}
                for test_dataset in test_datasets:
                    metrics, df = ensemble_metrics(folders_path=folders_path, test_dataset=test_dataset, thresh=threshold,
                                                   gender=gender, gender_dict=gender_info[test_dataset])
                    metrics_datasets[test_dataset] = metrics
                    df_datasets[test_dataset] = df
                    filename = add_gender(f"predictions_{task}_{test_dataset}_{training_strategy}.xlsx", gender)
                    df.drop(columns=["filename"]).to_excel(join("raw_data_plots_tables", filename))

                multi_index = pd.MultiIndex.from_product([test_datasets, ["ensemble", "means"]])
                test_internal_external_metrics = pd.DataFrame(
                    pd.concat(list(metrics_datasets.values()), axis=1).values,
                    columns=multi_index,
                    index=metrics_datasets[test_datasets[0]].index)
                test_internal_external_metrics.to_csv(
                    join(results_output_folder, add_gender("metrics_ensemble.csv", gender)),
                    index=False
                )

                # Metrics for different thresholds
                metrics_thresholds_datasets = {}
                for test_dataset in test_datasets:
                    metrics_thresholds = ensemble_metrics_for_different_thresholds(thresholds=thresholds,
                                                                                   test_dataset=test_dataset,
                                                                                   folders=folders_path,
                                                                                   gender=gender,
                                                                                   gender_dict=gender_info[test_dataset])
                    metrics_thresholds_datasets[test_dataset] = metrics_thresholds

                multi_index = pd.MultiIndex.from_product([test_datasets, thresholds])
                test_internal_external_thresholds = pd.DataFrame(
                    pd.concat(list(metrics_thresholds_datasets.values()), axis=1).values, columns=multi_index,
                    index=metrics_thresholds_datasets[test_datasets[0]].index)
                test_internal_external_thresholds.to_csv(
                    join(results_output_folder, add_gender("metrics_ensemble_thresholds.csv", gender)),
                    index=False
                )

                for df, name, color in zip(df_datasets.values(),
                                           [name_to_title[key] for key in df_datasets.keys()],
                                           ["red", "blue"]):
                    # Generate and save confusion matrices
                    for tsh in thresholds:
                        path = join(images_output_folder, add_gender(f"confusion_matrix_{name}_{tsh}.png", gender))
                        title = add_gender(f"{name.replace('_', ' ')} - {threshold_title}: " + str(tsh), gender, True)
                        plot_confusion_matrix(predictions=df["ensemble"] > tsh, labels=df["label"].astype(int), output_path=path,
                                              title=title, show=False)

                    # Plot ensemble ROC curve
                    plot_roc_bootstrap(X=df["ensemble"], y=df["label"], pos_label=1,
                                       label=f"{name}",
                                       color=color,
                                       n_bootstrap=n_boot, show_boots=False)
                    plt.title(add_gender(f"{name} - Ensemble", gender, True))
                    plt.suptitle("")  # Clear the subtitle
                    path = join(images_output_folder, add_gender(f"{name}.png", gender))
                    plt.savefig(path, bbox_inches='tight', dpi=dpi)
                    plt.close()

                    # Plot ROC curve for single models
                    roc_curves = []
                    for i in [0, 1, 2, 3, 4]:
                        roc = compute_roc(X=df[f"model_{i}"], y=df["label"], pos_label=1)
                        roc_curves.append(roc)
                        plot_roc(roc, label=f"Model {i + 1}", color=color_palette[i])
                    plt.title(add_gender(f"{name} - 5-Folds", gender, True))
                    plt.savefig(join(images_output_folder, add_gender(f"{name}_single_models.png", gender)),
                                bbox_inches='tight', dpi=dpi)
                    plt.close()

                    # Plot mean ROC curve across single models
                    plot_mean_roc(roc_curves, show_ci=False, show_ti=False, show_all=True, color=color)
                    plt.title(add_gender(f"{name} - Mean ROC", gender, True))
                    plt.savefig(join(images_output_folder, add_gender(f"{name}_single_models_average.png", gender)),
                                bbox_inches='tight', dpi=dpi)
                    plt.close()


if __name__ == '__main__':
    main()
