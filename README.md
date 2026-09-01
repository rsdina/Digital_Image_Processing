# SAR Port Functional Zone Segmentation

## Deep Learning Benchmarking for SAR Image Segmentation

This repository contains a comprehensive benchmarking study of deep learning models for **Semantic Segmentation of SAR (Synthetic Aperture Radar) Port Functional Zone Imagery**. The project evaluates and compares multiple state-of-the-art segmentation architectures on a specialized dataset of SAR port images.

## Project Overview

### Motivation
SAR (Synthetic Aperture Radar) imagery is crucial for maritime surveillance, port monitoring, and defense applications. Semantic segmentation of port functional zones enables automated analysis of harbor activities, infrastructure monitoring, and change detection over time.

### Dataset
The dataset consists of **SAR port functional zone images** with pixel-level annotations across **5 classes**:
- 🟦 **Water Area**
- 🟥 **Cargo Area**
- 🟩 **Container Area**
- 🟧 **Tank Area**
- ⬜ **Others**

**Dataset Statistics**:
- Total samples: ~5,267 paired images and masks
- Training set: 3,686 samples (70%)
- Validation set: 790 samples (15%)
- Test set: 791 samples (15%)

### Methodology
The study follows a systematic benchmarking approach:
1. **NB0**: EDA & Data Preparation (Baseline)
2. **NB1**: U-Net (CNN-based) - First Benchmark
3. **NB2**: SegFormer-B0 (Transformer-based)
4. **NB3**: MobileNetV3 (Lightweight/Edge Device)

## Model Results Comparison

| Model | Test Loss | Test mIoU | Avg Precision | Avg Recall | Avg F1 | Parameters |
|-------|-----------|-----------|---------------|------------|--------|------------|
| **U-Net (NB1)** | 0.0648 | 0.9336 | 0.9407 | 0.9363 | 0.9380 | ~7.8M |
| **SegFormer-B0 (NB2)** | **0.0265** | **0.9732** | **0.9785** | **0.9814** | **0.9799** | ~3.7M |
| **MobileNetV3 (NB3)** | TBD | TBD | TBD | TBD | TBD | ~2.5M |

### Key Observations
- **SegFormer-B0** achieves the highest mIoU (0.9732) with only 3.7M parameters
- **U-Net** performs well (mIoU 0.9336) but requires more compute resources
- Transformer-based architecture (SegFormer) significantly outperforms CNN-based U-Net
- The lightweight nature of SegFormer-B0 makes it suitable for deployment scenarios

## Per-Class Performance

### SegFormer-B0 (NB2) - Best Performing Model

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Water Area | 0.9775 | 0.9796 | 0.9786 |
| Cargo Area | 0.9744 | 0.9883 | 0.9813 |
| Container Area | 0.9800 | 0.9841 | 0.9820 |
| Tank Area | 0.9872 | 0.9866 | 0.9869 |
| Others | 0.9734 | 0.9682 | 0.9708 |
| **Average** | **0.9785** | **0.9814** | **0.9799** |

### U-Net (NB1)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Water Area | 0.9602 | 0.9527 | 0.9564 |
| Cargo Area | 0.9332 | 0.9538 | 0.9434 |
| Container Area | 0.9712 | 0.9521 | 0.9615 |
| Tank Area | 0.9896 | 0.9680 | 0.9787 |
| Others | 0.8493 | 0.8547 | 0.8520 |
| **Average** | **0.9407** | **0.9363** | **0.9380** |

### Insights
- Both models struggle most with the "Others" class (lowest F1 scores)
- "Tank Area" is the most accurately predicted class
- SegFormer-B0 shows significantly better performance on all classes
- The performance gap is largest on "Others" (19% improvement in F1)

## Technical Details

### Architecture Specifications

#### U-Net (NB1)
- **Encoder**: Convolutional (downsampling path)
- **Decoder**: Upsampling with skip connections
- **Params**: ~7.8 million
- **Training**: 50 epochs, batch size 8, LR 1e-3
- **Loss**: Weighted CrossEntropyLoss

#### SegFormer-B0 (NB2)
- **Encoder**: MiT-B0 (Hierarchical Transformer) 
- **Channels**: 32 → 64 → 160 → 256
- **Decoder**: MLP Head (Lightweight)
- **Params**: ~3.7 million (47% fewer than U-Net)
- **Training**: 50 epochs, batch size 4, LR 5e-4
- **Loss**: Weighted CrossEntropyLoss

#### MobileNetV3 
- **Architecture**: Lightweight MobileNetV3 backbone with segmentation head
- **Params**: ~2.5 million
- **Training**: In progress...

### Data Augmentation
- Resize: 384×384 (NB2/NB3) / 512×512 (NB1)
- Horizontal Flip: 50% probability
- Vertical Flip: 30% probability
- Random Rotate90: 30% probability
- RandomBrightnessContrast: 20% variation

### Training Details
- **Mixed Precision**: Enabled for CUDA
- **Scheduler**: CosineAnnealingLR
- **Optimizer**: AdamW
- **Class Weights**: Inverse frequency weighting (computed in NB0)

## Notebook Structure

### NB0 - EDA & Data Preparation
- Dataset exploration and validation
- Class distribution analysis
- Generating class weights (inverse frequency)
- Creating train/val/test splits (70/15/15)
- Output: `class_weights.json`, `split_indices.pkl`

### NB1 - U-Net Implementation
- Complete U-Net architecture training
- Per-epoch loss and IoU tracking
- Test evaluation with detailed metrics
- Confusion matrix visualization
- Results appended to `results.json`

### NB2 - SegFormer-B0 Implementation 
- HuggingFace transformers integration
- Pretrained weights (ADE20K fine-tuning)
- Memory-optimized training
- Test evaluation with detailed metrics
- Results appended to `results.json`


## Key Features

- ✅ **Reproducible results** with fixed random seeds
- ✅ **Comprehensive metrics**: mIoU, Precision, Recall, F1 per class
- ✅ **Visualization**: Confusion matrices, loss curves, sample predictions
- ✅ **Weighted loss** for handling class imbalance
- ✅ **Data augmentation** for robust training
- ✅ **Mixed precision** for faster training
- ✅ **Shared data splits** for fair comparison across models

## Results Visualization

### Confusion Matrix (SegFormer-B0)
The confusion matrix shows excellent diagonal dominance, indicating high classification accuracy across all classes. Tank and Container areas show the highest accuracy, while "Others" is the most challenging class.

### Training Curves
Both models show:
- Rapid decrease in loss during first 10-15 epochs
- Validation loss stabilizing after ~30 epochs
- Smooth learning rate decay with CosineAnnealingLR

## Conclusion

This benchmarking study demonstrates that:
1. **SegFormer-B0** significantly outperforms U-Net on SAR port segmentation
2. Transformer architectures generalize better on this specialized dataset
3. Weighted loss and data augmentation effectively handle class imbalance
4. The 47% parameter reduction of SegFormer-B0 makes it more suitable for production deployment

### Deployment Recommendations
- **Mobile/Edge**: SegFormer-B0 (best accuracy/size trade-off)
- **High Accuracy Required**: SegFormer-B0 (best mIoU)
- **Legacy Systems**: U-Net (more stable and widely supported)

## Dependencies

```
Python 3.12+
PyTorch 2.10.0+
Transformers 5.0.0+
Timm 1.0.26+
Albumentations 2.0.8+
Scikit-learn
Matplotlib
Pandas
NumPy
```

## Usage

### Running the Notebooks
1. Clone this repository
2. Ensure dataset is available at the specified path
3. Run notebooks in order: NB0 → NB1 → NB2 
4. Results will be automatically aggregated in `results.json`

### Key Paths
- Dataset: `/kaggle/input/datasets/rebekadina/sar-port-functional-zone-segmentation`
- Output: `/kaggle/working/nb*_outputs/`

## Future Work

- Evaluate additional architectures (DeepLabV3, Vision Transformer)
- Explore ensemble methods
- Test on larger SAR datasets
- Quantize model for edge deployment
- Real-time inference optimization

## License

This project is for research purposes. Please refer to individual notebook licenses for specific library usage rights.

---

**Acknowledgments**: This project utilizes the `nvidia/segformer-b0-finetuned-ade-512-512` pretrained weights from HuggingFace and the SAR Port Functional Zone Segmentation dataset.
