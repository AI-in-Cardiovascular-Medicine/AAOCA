import os
from argparse import ArgumentParser
from os.path import join, split

import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default=f"cardiac_segmentation")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--input_dir", default="../data/anomaly_detection/train")
parser.add_argument("--checkpoint_name", default='checkpoint_final.pth')

args = parser.parse_args()


def get_files(input_dir: str, processed_files=None, key="img.nii.gz"):
    files = []
    for dirpath, dirnames, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename == key:
                img_path = join(dirpath, filename)
                if processed_files and split(img_path)[0] in processed_files:
                    continue
                files.append(img_path)
    return files


if __name__ == '__main__':
    predictor = nnUNetPredictor(
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
    predictor.initialize_from_trained_model_folder(
        args.model_path,
        use_folds=(0, 1, 2, 3, 4),
        # use_folds=(1, 2),
        checkpoint_name=args.checkpoint_name,
    )

    # variant 2, use list of files as inputs. Note how we use nested lists!!!
    processed_files = [split(item)[0] for item in get_files(args.input_dir, key="seg.nii.gz")]
    remaining_files = get_files(args.input_dir, processed_files)
    output_files = [join(split(file_path)[0], "seg.nii.gz") for file_path in remaining_files]
    if len(remaining_files) and len(output_files):
        predictor.predict_from_files(
            [[item] for item in remaining_files],
            output_files,
            save_probabilities=False,
            overwrite=False,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0)
    print("Segmentation finished!")
