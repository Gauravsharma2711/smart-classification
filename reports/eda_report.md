# Exploratory Data Analysis (EDA) Report

## 1. Dataset Overview
- **Directory**: `data/raw`
- **Total Scanned Images**: 2527
- **Valid Decodable Images**: 2527
- **Corrupt / Unreadable Images**: 0
- **Imbalance Ratio (Max/Min Class)**: 4.34

## 2. Class Counts and Loss Weights
| Class Name | Image Count | Percentage (%) | Balanced Class Weight |
|---|---|---|---|
| `cardboard` | 403 | 15.95% | 1.0451 |
| `glass` | 501 | 19.83% | 0.8407 |
| `metal` | 410 | 16.22% | 1.0272 |
| `paper` | 594 | 23.51% | 0.7090 |
| `plastic` | 482 | 19.07% | 0.8738 |
| `trash` | 137 | 5.42% | 3.0742 |

## 3. Image Dimensions and Resolution
- **Width (px)**: Min=512.0, Mean=512.0, Max=512.0 (Median=512.0)
- **Height (px)**: Min=384.0, Mean=384.0, Max=384.0 (Median=384.0)
- **Aspect Ratio**: Mean=1.33, Median=1.33

## 4. File Attributes
- **Color Modes**: {"RGB": 2527}
- **File Size (KB)**: Min=5.47 KB, Mean=16.73 KB, Max=56.51 KB

## 5. Visualizations
- Class Distribution: `reports/class_distribution.png`
- Dimensions & Size: `reports/image_resolution_distribution.png`
