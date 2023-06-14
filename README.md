# M-FLAG-MICCAI2023

[M-FLAG: Medical Vision-Language Pre-training with Frozen Language Models and Latent Space Geometry Optimization](link), MICCAI 2023.

###  Installation
To clone this repository:
```
git clone https://github.com/cheliu-computation/M-FLAG-MICCAI2023.git
```
To install Python dependencies:
```
pip install -r requirements.txt
```
All experiments are implemented on A100 GPU.

### Pre-train Dataset downloading
Datasets we used are as follows:
- **MIMIC-CXR**: We downloaded the [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) dataset as the radiographs. Paired medical reports can be downloaded in [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/mimic-cxr-reports.zip).

### Preprocessing
- First we follow [MGCA](https://github.com/HKU-MedAI/MGCA) preprocessing to extract a master csv includes all CXR scans associated with report. You can find in [Preprocessing](https://github.com/HKU-MedAI/MGCA/blob/main/mgca/preprocess/mimic_cxr.py). 
- Then, run 'ext_data.py' to extract all scans and save as a npy file. It will accelerate the pre-training stage.

### Pre-training
We pre-trained MGCA on MIMIC-CXR using this command:
```

cd M-FLAG-MICCAI2023/pretrain
torchrun --nnodes=1 --nproc_per_node=2 main.py
```

### Finetune on downstream tasks
We evlauate the performance of M-FLAG on three downstream tasks: classification, object detection and semantic segmentation. 

For classification task, we follow [CheXclusion](https://github.com/LalehSeyyed/CheXclusion), please follow their offical code to extract data and implement classification tasks.

For semantic segmentation and object detection, we follow [MGCA](https://github.com/HKU-MedAI/MGCA) offical configuration and code. The dataset can be found in MGCA repository.
