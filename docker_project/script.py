import os
import shutil
from argparse import ArgumentParser

import torch.cuda
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from utils import *


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(h)
        free = info.free / 1024 / 1024 / 1024  # GB
        if free < 5:
            device = "cpu"
    return device


ANOMALY_DETECTION_MODEL_PATH = [
    "web_service_models/classification_models/anomaly_detection/model_1/best_val_model.pt",
    "web_service_models/classification_models/anomaly_detection/model_2/best_val_model.pt",
    "web_service_models/classification_models/anomaly_detection/model_3/best_val_model.pt",
    "web_service_models/classification_models/anomaly_detection/model_4/best_val_model.pt",
    "web_service_models/classification_models/anomaly_detection/model_5/best_val_model.pt",
]
RISK_CLASSIFICATION_MODEL_PATH = [
    "web_service_models/classification_models/risk_classification/model_1/best_val_model.pt",
    "web_service_models/classification_models/risk_classification/model_2/best_val_model.pt",
    "web_service_models/classification_models/risk_classification/model_3/best_val_model.pt",
    "web_service_models/classification_models/risk_classification/model_4/best_val_model.pt",
    "web_service_models/classification_models/risk_classification/model_5/best_val_model.pt"
]
ORIGIN_CLASSIFICATION_MODEL_PATH = [
    "web_service_models/classification_models/origin_classification/model_1/best_val_model.pt",
    "web_service_models/classification_models/origin_classification/model_2/best_val_model.pt",
    "web_service_models/classification_models/origin_classification/model_3/best_val_model.pt",
    "web_service_models/classification_models/origin_classification/model_4/best_val_model.pt",
    "web_service_models/classification_models/origin_classification/model_5/best_val_model.pt",

]
SEGMENTATION_MODEL_PATH = "web_service_models/cardiac_segmentation"
INPUT_DIR = "input"
OUTPUT_DIR = "output"


def get_modes_performance(models_path: list[str], fast: bool, device, img: torch.tensor):
    percentages = []
    for model_path in models_path:
        model = load_model(device=device, model_path=model_path).eval()
        percentage = torch.sigmoid(model(img)[0]).item()
        del model
        if fast:
            return percentage
        percentages.append(percentage)
    return np.mean(percentages)


@torch.no_grad()
def process_cropped_sample(img_path: str | np.ndarray, threshold: float, device, fast: bool):
    tic = time.time()
    img = normalize_and_resample(img_path)
    img = torch.Tensor(img).to(torch.float32)
    last_id = len(img.shape) - 1
    img = img.swapaxes(last_id - 1, last_id).swapaxes(last_id - 2, last_id - 1)
    img = img.unsqueeze(0).to(device)
    if len(img.shape) == 4:
        img = img.unsqueeze(0)
    if threshold != 1:
        anomaly_percentage = get_modes_performance(ANOMALY_DETECTION_MODEL_PATH, fast, device, img)
        anomaly = 1 if anomaly_percentage > threshold else 0
    else:
        anomaly_percentage = 1
        anomaly = 1

    if anomaly == 1:
        origin = get_modes_performance(ORIGIN_CLASSIFICATION_MODEL_PATH, fast, device, img)
        risk = get_modes_performance(RISK_CLASSIFICATION_MODEL_PATH, fast, device, img)
    else:
        origin = 0
        risk = 0
    output = dict(anomaly=round(anomaly_percentage, 4), risk=round(risk, 4), origin=round(origin, 4))
    print(f"[INFO] {time.time() - tic} to get the output: {output}")
    return output


def create_report(output: dict, threshold):
    anomaly, l_origin, risk = output['anomaly'] > threshold, output['origin'] > threshold, output['risk'] > threshold
    if not anomaly:
        txt = "No coronary anomalies (AAOCA) have been detected."
    else:
        if l_origin and risk:
            txt = "L-AAOCA has been detected with anatomical high-risk features."
        elif not l_origin and risk:
            txt = "R-AAOCA has been detected with anatomical high-risk features."
        elif l_origin and not risk:
            txt = "L-AAOCA has been detected with anatomical low-risk features."
        elif not l_origin and not risk:
            txt = "R-AAOCA has been detected with anatomical low-risk features."
        else:
            raise ValueError()
    return txt


def main_inference(input_path: str,
                   output_dir: str,
                   is_cropped: bool,
                   is_nifti: bool,
                   threshold: float,
                   device: str,
                   fast: bool):
    os.makedirs(output_dir, exist_ok=True)
    if not is_nifti:
        input_path = dicom2nifti(output_dir, input_path)

    with torch.no_grad():
        if is_cropped:
            output = process_cropped_sample(input_path, threshold, device, fast)
        else:
            seg_path = DirUtils.split_extension(input_path, suffix="_seg", current_extension=".nii.gz")
            cropped_seg_path = DirUtils.split_extension(seg_path, suffix="_cropped", current_extension=".nii.gz")
            cropped_img_path = DirUtils.split_extension(input_path, suffix="_cropped", current_extension=".nii.gz")
            print(f"{input_path} and {seg_path}")
            predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=torch.device(device),
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=True
            )

            # initializes the network architecture, loads the checkpoint
            predictor.initialize_from_trained_model_folder(
                SEGMENTATION_MODEL_PATH,
                use_folds=(0,) if fast else (0, 1, 2, 3, 4),
                checkpoint_name="checkpoint_final.pth",
            )
            predictor.predict_from_files(
                [[input_path]],
                [seg_path],
                save_probabilities=False,
                overwrite=False,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
                folder_with_segs_from_prev_stage=None,
                num_parts=1,
                part_id=0)
            del predictor
            # let's crop it :)
            process_sample(seg_path, cropped_img_path, cropped_seg_path, input_path, (8, 8, 6),
                           2, 3, (-1, 1, 1))
            output = process_cropped_sample(cropped_img_path, threshold, device, fast)
    return output


def create_output_file(report: dict, thresh: float, txt_path: str):
    thresh *= 100
    anomaly_val = report.get("anomaly") * 100
    if anomaly_val < thresh:
        anomaly_label = "Normal"
        origin_val = origin_label = "-"
        risk_val = risk_label = "-"
    else:
        anomaly_label = "AAOCA"
        origin_val = report.get("origin") * 100
        risk_val = report.get("risk") * 100
        if origin_val < thresh:
            origin_label = "R-AAOCA"
        else:
            origin_label = "L-AAOCA"
        if risk_val < thresh:
            risk_label = "Low Risk"
        else:
            risk_label = "High Risk"
    distance = 30
    with open(txt_path, mode="w") as f:
        f.write(
            "############################################################################################################################\n")
        f.write(
            f"##########################################     Model Outputs for threshold: {round(thresh)}    ##########################################\n")
        f.write(
            "                              |   Anomaly Detection          |     Origin Classification    |Anatomical Risk Classification\n")
        line = "Probability".center(distance) + "|" + str(anomaly_val).center(distance) + "|" + str(origin_val).center(
            distance) + "|" + str(risk_val).center(distance) + "\n"
        f.write(line)

        line = "Final Results".center(distance) + "|" + str(anomaly_label).center(distance) + "|" + str(
            origin_label).center(
            distance) + "|" + str(risk_label).center(distance) + "\n"
        f.write(line)
        f.write(
            "############################################################################################################################\n")

        f.write(
            "################################################## Final Diagnosis Report ##################################################\n")
        f.write(report.get("report") + "\n")
        f.write(
            "############################################################################################################################")


def main(args):
    tic = time.time()
    device = get_device()
    print(f"[INFO] Running on device: {device}")

    if args.is_nifti and args.input_path.endswith(".nii"):
        shutil.copy(join(INPUT_DIR, args.input_path), join(INPUT_DIR, args.input_path + ".gz"))
        args.input_path = args.input_path + ".gz"

    img_name = os.path.basename(args.input_path).replace(".nii.gz", "").replace(".zip", "").replace(".nii", "")
    output_path = join(OUTPUT_DIR, img_name)
    input_path = join(INPUT_DIR, args.input_path)
    output = main_inference(input_path, output_path, args.is_cropped, args.is_nifti, args.threshold, device, args.fast)
    output['report'] = create_report(output, args.threshold)
    JsonUtils.dump(join(OUTPUT_DIR, "output.json"),
                   {k: (v * 100) if isinstance(v, float) else v for k, v in output.items()})
    create_output_file(output, args.threshold, join(OUTPUT_DIR, "output.txt"))
    print(f"[INFO] Finished in {time.time() - tic} with threshold: {args.threshold}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_path", help="Path to the input image")
    parser.add_argument("--is_cropped", action="store_true", help="Whether the input data is cropped!")
    parser.add_argument("--fast", action="store_true", help="Whether to use only one model!")
    parser.add_argument("--is_nifti", action="store_true",
                        help="Whether the input data is nifti! If set to False, means dicom!")
    parser.add_argument("--threshold", default=0.5, type=float, help="anomaly detection threshold")

    args = parser.parse_args()
    main(args)
