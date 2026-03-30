# 🚀 SSR-Net: Spectral-Spatial Rectification Network with Dual-Domain Synergy for Medical Image Segmentation

## 📋 How to Use

### 🔧 Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Ubuntu 16.04 or higher |
| **CUDA** | 11.1 or higher |
| **Python** | v3.7 or higher |
| **PyTorch** | v1.7 or higher |
| **GPU** | Single GPU with 12GB+ VRAM _(we used RTX 3090)_ |

---

### 📦 Model Weights

Pre-trained and learned model weights are available for download:

| Dataset | Model | Download Link |
|---------|-------|---------------|
| **ImageNet** | MaxViT Small 224 | [⬇️ Download](https://drive.google.com/file/d/1MaWFYadsYFEROLNvYG8hZAYnlGCkPLaN/view?usp=sharing) *(third-party)* |
| **Synapse** | SSR-Net ||

---

### 🎯 Training and Testing

#### 1️⃣ Download Dataset

Download the Synapse dataset from [**here**](https://drive.google.com/uc?export=download&id=18I9JHH_i0uuEDg-N6d7bfMdf7Ut6bhBi) *(third-party resource)*.

#### 2️⃣ Download Pre-trained Weights

Download the MaxViT Small 224×224 pretrained weights from [**here**](https://drive.google.com/file/d/1MaWFYadsYFEROLNvYG8hZAYnlGCkPLaN/view?usp=sharing) and place it in `networks/merit_lib/networks.py` for initialization.

#### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4️⃣ Train the Model

Run the following command to train SSR-Net on the Synapse dataset:

```bash
python train.py --root_path ./data/Synapse/train_npz \
                --test_path ./data/Synapse/test_vol_h5 \
                --batch_size 20 \
                --eval_interval 20 \
                --max_epochs 700
```

**Parameter Descriptions:**
- `--root_path`: Path to training data
- `--test_path`: Path to test data
- `--eval_interval`: Evaluation interval (epochs)

#### 5️⃣ Test the Model

Run the following command to test HCA_Net on the Synapse dataset:

```bash
python test.py --volume_path ./data/Synapse/ \
               --output_dir ./model_out \
               --weights_fpath Your_model_weight_path
```

**Parameter Descriptions:**
- `--volume_path`: Root directory of test data
- `--output_dir`: Directory for output results
- `--weights_fpath`: Path to your trained model weights

---

## 📚 References

- **D-LKA Net**: [https://github.com/xmindflow/deformableLKA](https://github.com/xmindflow/deformableLKA)
- **SKNet**: [https://github.com/osmr/imgclsmob](https://github.com/osmr/imgclsmob/tree/68335927ba27f2356093b985bada0bc3989836b1)
- **FAT**: [https://github.com/qhfan/FAT](https://github.com/qhfan/FAT)
- **MSA<sup>2</sup>Net**: [https://github.com/xmindflow/MSA-2Net](https://github.com/xmindflow/MSA-2Net)

---

<div align="center">
  <sub>Built with ❤️ for Medical Image Segmentation</sub>
</div>
