# SSR-Net: Spectral-Spatial Rectification Network

> Spectral-Spatial Rectification Network with Dual-Domain Synergy for Medical Image Segmentation.

---

## 📖 Overview
This repository provides the official implementation, pre-trained weights, and usage instructions for SSR-Net. 

---

## ⚙️ Prerequisites
Ensure your system meets the following specifications before proceeding:

* **OS:** Ubuntu 16.04 or higher
* **Compute:** Single GPU with 12GB+ VRAM (RTX 3090 recommended)
* **CUDA:** Version 11.1 or higher
* **Python:** Version 3.7 or higher
* **Framework:** PyTorch 1.7 or higher

---

## 🗄️ Model Weights
Pre-trained weights are required for initialization.

| Dataset | Model | Download Link |
| :--- | :--- | :--- |
| **ImageNet** | MaxViT Small 224 | [Download weights](https://drive.google.com/file/d/1MaWFYadsYFEROLNvYG8hZAYnlGCkPLaN/view?usp=sharing) *(third-party)* |
| **Synapse** | SSR-Net | [Download weights](https://drive.google.com/file/d/19-rCliU6VM1eMtN084U01ZDZayF8iMVZ/view?usp=drive_link) |

---

## 🚀 Getting Started

### 1. Data Preparation
Download the Synapse dataset from this [third-party resource](https://drive.google.com/uc?export=download&id=18I9JHH_i0uuEDg-N6d7bfMdf7Ut6bhBi). Extract and place it in your designated data directory.

### 2. Weight Initialization
Download the MaxViT Small 224×224 pre-trained weights from the table above. Place the downloaded file into the following directory:
`networks/merit_lib/networks.py`

### 3. Installation
Install the required Python dependencies:
```bash
pip install -r requirements.txt
```

---

## 💻 Training and Evaluation

### Training SSR-Net
To train the model on the Synapse dataset, execute the following command:

```bash
python train.py \
    --root_path ./data/Synapse/train_npz \
    --test_path ./data/Synapse/test_vol_h5 \
    --batch_size 20 \
    --eval_interval 20 \
    --max_epochs 700
```

**Training Arguments:**
* `--root_path`: Path to the training dataset.
* `--test_path`: Path to the testing dataset.
* `--eval_interval`: Number of epochs between evaluations.

### Testing the Model

To evaluate the model on the test set, run:

```bash
python test.py \
    --volume_path ./data/Synapse/ \
    --output_dir ./model_out \
    --weights_fpath <path_to_your_trained_weights>
```

**Testing Arguments:**
* `--volume_path`: Root directory containing the test data.
* `--output_dir`: Target directory for output predictions.
* `--weights_fpath`: File path to your specific model weights.

---

## 📎 Acknowledgements
This project builds upon the excellent work of the following repositories:

* **D-LKA Net:** [xmindflow/deformableLKA](https://github.com/xmindflow/deformableLKA)
* **SKNet:** [osmr/imgclsmob](https://github.com/osmr/imgclsmob/tree/68335927ba27f2356093b985bada0bc3989836b1)
* **FAT:** [qhfan/FAT](https://github.com/qhfan/FAT)
* **MSA²Net:** [xmindflow/MSA-2Net](https://github.com/xmindflow/MSA-2Net)

---
*Built for the Medical Image Segmentation community.*
