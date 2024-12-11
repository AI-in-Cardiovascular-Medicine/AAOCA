"""
This module is to create tsne projections, to do be able to do for various genders the following extra files are required:
1. internal_gender_img_names.json: A json file which contains a dictionary. The keys are patient ids and the values are genders
2. external_gender_img_names.json: A json file which contains a dictionary. The keys are patient ids and the values are genders
"""
import numpy as np
import os
from os.path import join, split
import argparse
from enum import StrEnum
from deep_utils import JsonUtils

import pandas as pd
from sklearn.manifold import TSNE


class TitleName(StrEnum):
    test_external = "External testing"
    test_internal = "Internal testing"
    train = "Training & validation"


def get_genders(sample_names: list, gender: str, internal_gender_data: dict, external_gender_data: dict):
    """
    Extract gender names
    :param sample_names: sample names
    :param gender: gender
    :param internal_gender_data: dictionary with internal dataset genders
    :param external_gender_data: dictionary with external dataset genders
    :return: gender of the input samples
    """
    output = []
    if gender:
        for name in sample_names:
            if name in internal_gender_data:
                output.append(internal_gender_data.get(name).strip().lower())
                continue
            else:
                output.append(external_gender_data.get(name).strip().lower())
    else:
        output = [gender for _ in sample_names]
    return output


def get_names(data_path: str, name: str, label_increase: dict):
    """
    Get the names
    :param data_path:
    :param name:
    :param label_increase:
    :return:
    """
    data = {"labels": [], "img_names": []}
    for dirpath, dirnames, filenames in os.walk(data_path):
        for filename in filenames:
            if filename == "img_cropped_resampled.nii.gz":
                lbl_str = dirpath.split("/")[-3]
                lbl = int(lbl_str)
                data['img_names'].append(split(dirpath)[-1])
                data['labels'].append(lbl + label_increase[name])
    return data


def main():
    """Compute TSNE mapping for latent deep learning features"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_features_path", type=str, default="latent_features",
                        help="Path to the directory where DL latent features are stored.")
    parser.add_argument("--original_data_dir", type=str, required=True,
                        help="Path to the directory where DL latent features are stored.")
    parser.add_argument("--output_dir", default="raw_data_plots_tables",
                        help="output directory to save excel files.")
    args = parser.parse_args()

    data_path = args.latent_features_path
    original_data_dir = args.original_data_dir
    test_external_array = np.load(join(data_path, "test_external_features.npz"), allow_pickle=True)
    train_array = np.load(join(data_path, "train_features.npz"), allow_pickle=True)
    test_internal_array = np.load(join(data_path, "test_internal_features.npz"), allow_pickle=True)
    array_dict = dict(test_external=test_external_array, train=train_array, test_internal=test_internal_array)

    gender_info = {
        "internal": JsonUtils.load("internal_gender_img_names.json"),
        "external": JsonUtils.load("external_gender_img_names.json")
    }

    label_map = {
        "anomaly": {
            0: f"{TitleName.test_external} (normal)",
            1: f"{TitleName.test_external} (AAOCA)",
            4: f"{TitleName.train} (normal)",
            5: f"{TitleName.train} (AAOCA)",
            6: f"{TitleName.test_internal} (normal)",
            7: f"{TitleName.test_internal} (AAOCA)"
        },
        "origin": {
            0: f"{TitleName.test_external} (R-AAOCA)",
            1: f"{TitleName.test_external} (L-AAOCA)",
            2: f"{TitleName.train} (R-AAOCA)",
            3: f"{TitleName.train} (L-AAOCA)",
            4: f"{TitleName.test_internal} (R-AAOCA)",
            5: f"{TitleName.test_internal} (L-AAOCA)"
        },
        "risk": {
            0: f"{TitleName.test_external} (Low Risk)",
            1: f"{TitleName.test_external} (High Risk)",
            2: f"{TitleName.train} (Low Risk)",
            3: f"{TitleName.train} (High Risk)",
            4: f"{TitleName.test_internal} (Low Risk)",
            5: f"{TitleName.test_internal} (High Risk)",
        }
    }

    label_increase_all = {
        "anomaly": dict(test_external=0, train=4, test_internal=6),
        "origin": dict(test_external=0, train=2, test_internal=4),
        "risk": dict(test_external=0, train=2, test_internal=4)
    }

    graph_combinations = [('test_external', 'train', 'test_internal')]

    for combination in graph_combinations:
        y, sample_names, x, preds = [], [], [], []
        print("--- Anomaly task ---")
        for name in combination:
            array = array_dict[name]
            y.extend((array['labels'].astype(np.float32) + label_increase_all["anomaly"][name]).tolist())
            sample_names.extend(array['names'].tolist())
            x.extend(array['features'].astype(np.float32).tolist())
            preds.extend(array['predicted_lbl'].tolist())
            print(f"{name}: {len(array['labels'])} images , {sum(array['labels'])} cases")
        y_names = [label_map["anomaly"][i] for i in y]

        manifold = TSNE(n_components=2)
        x_reduced = manifold.fit_transform(np.array(x).astype(np.float32))

        df = pd.DataFrame({
            "tsne_1": x_reduced[:, 0],
            "tsne_2": x_reduced[:, 1],
            "labels_anomaly": y_names,
            "pred": preds,
            "img_names": sample_names
        })

        # Get labels for origin and risk classification tasks
        for task in ["origin", "risk"]:
            print(f"--- {task.capitalize()} task ---")
            y, sample_names = [], []
            for name in combination:
                data = get_names(join(original_data_dir, f"{task}_classification", name), name=f"{name}",
                                 label_increase=label_increase_all[task])
                sample_names.extend(data["img_names"])
                y.extend([label_map[task][lbl] for lbl in data["labels"]])

            df_task = pd.DataFrame({"img_names": sample_names, f"labels_{task}": y})
            df = df.merge(df_task, on="img_names", how="left")

        path = join(args.output_dir, "tsne_" + "-".join(combination))
        df.drop(columns=["img_names"]).to_excel(path + ".xlsx", index=False)

        print(f"Creating gender datasets\n\ttot: {len(df)}")
        for gender in ["f", "m"]:
            gender_filter = [g == gender for g in get_genders(df["img_names"], gender, gender_info["internal"],
                                                              gender_info["external"])]
            print("\t", gender, ": ", sum(gender_filter))
            df.loc[gender_filter].drop(columns=["img_names"]).to_excel(path + f"_{gender}.xlsx", index=False)

        print(f"TSNE embeddings are done for {combination} in {path}")


if __name__ == '__main__':
    main()
