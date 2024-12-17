# Docker
Here we present the full flow of creating docker image that supports the following features:
1. Full preprocessing for dicom and nifti files including:
   1. dicom to nifti
   2. segmentation
   3. cropping
2. anomaly detection
3. risk classification
4. origin classification

# Provide weights
Create a folder called `web_service_models` it should be like the following:
```commandline
├── web_service_models
│   ├── anomaly_detection
│   │   ├── model_1
│   │   │   ├── best_val_model.pt
│   │   │   ├── config.json
│   │   ├── model_2
│   │   │   ├── ...
│   ├── risk_classification
│   │   ├── model_1
│   │   │   ├── best_val_model.pt
│   │   │   ├── config.json
│   │   ├── model_2
│   │   │   ├── ...
│   ├── origin_classification
│   │   ├── model_1
│   │   │   ├── best_val_model.pt
│   │   │   ├── config.json
│   │   ├── model_2
│   │   │   ├── ...
│   ├── cardiac_segmentation
│   │   ├── plans.json
│   │   ├── dataset.json
│   │   ├── fold_0
│   │   │   ├── checkpoint_final.pth
│   │   ├── ...
```

# Docker Image
Build the image using the following command:
```commandline
docker build . -t narco_script:latest
```

# Run Docker
Create two folders namely `input` and `output`. Then, put input image in the input directory and then run one of the following commands.

## Running cropped nifti file
```commandline
docker run --rm  -v ./input:/app/input -v ./output:/app/output --gpus all narco_script:latest --input_path img-path.nii.gz --is_cropped --is_nifti
```
## Running cropped nifti file with threshold
```commandline
docker run --rm  -v ./input:/app/input -v ./output:/app/output --gpus all narco_script:latest --input_path img-path.nii.gz --is_cropped --is_nifti --threshold 0.5
```
## Running non cropped nifti file
```commandline
docker run --rm  -v ./input:/app/input -v ./output:/app/output --gpus all narco_script:latest --input_path img-path.nii.gz --is_nifti
```
## Running with dicom zip file
```commandline
docker run --rm  -v ./input:/app/input -v ./output:/app/output --gpus all narco_script:latest --input_path dicom-path.zip
```
## Output
The output reports are in `output/output.txt` and `output/output.json`.

## Expected Output for the samples:
Download the samples from the `Inference on samples data` and put them in the `input` directory, then run the following:
```commandline
docker run --rm  -v ./input:/app/input -v ./output:/app/output --gpus all narco_script:latest --input_path samples/normal/10064059/img.nii.gz --is_nifti
docker run --rm  -v ./input:/app/input -v ./output:/app/output --gpus all narco_script:latest --input_path samples/narco/11943667/img.nii.gz --is_nifti
docker run --rm  -v ./input:/app/input -v ./output:/app/output --gpus all narco_script:latest --input_path samples/narco/12017913/img.nii.gz --is_nifti
docker run --rm  -v ./input:/app/input -v ./output:/app/output --gpus all narco_script:latest --input_path samples/narco/12065293/img.nii.gz --is_nifti
docker run --rm  -v ./input:/app/input -v ./output:/app/output --gpus all narco_script:latest --input_path samples/narco/12076929/img.nii.gz --is_nifti
```
Expected Outputs:

**Note:** The range is from 0 to 100.
```json
10064059: {"anomaly": 0.0, "risk": 0, "origin": 0, "report": "No coronary anomalies(AAOCA) have been detected."}
11943667: {"anomaly": 59.39, "risk": 0.0, "origin": 0.83, "report": "R-AAOCA has been detected with anatomical low-risk features."}
12017913: {"anomaly": 100.0, "risk": 100.0, "origin": 0.49, "report": "R-AAOCA has been detected with anatomical high-risk features."}
12065293: {"anomaly": 100.0, "risk": 99.99, "origin": 0.07, "report": "R-AAOCA has been detected with anatomical high-risk features."}
12076929: {"anomaly": 81.39, "risk": 0.0, "origin": 100.0, "report": "L-AAOCA has been detected with anatomical low-risk features."}
```

The report.txt is like the following:
```commandline
############################################################################################################################
##########################################     Model Outputs for threshold: 70    ##########################################
                              |   Anomaly Detection          |     Origin Classification    |Anatomical Risk Classification
         Probability          |            81.39             |            100.0             |             0.0              
        Final Results         |            AAOCA             |           L-AAOCA            |           Low Risk           
############################################################################################################################
################################################## Final Diagnosis Report ##################################################
R-AAOCA has been detected with anatomical low-risk features.
############################################################################################################################
```


## Save docker image:
```commandline
docker save narco_script:latest -o narco_script.tar
```

## Load docker image:
```commandline
docker load -i narco_script.tar
```

## Large files:

For large files increase the shared memory size of the docker container if you face any sudden docker failure or Core Dump Error:
```commandline
docker run --rm --shm-size 128MB -v ./input:/app/input -v ./output:/app/output --gpus all narco_script:latest --input_path samples/normal/10064059/img.nii.gz --is_nifti
```