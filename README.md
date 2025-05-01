# Eye Blink Detection System

A PyTorch-based system for detecting and analyzing eye blinks. This project provides real-time blink detection from webcam feeds or video files, along with statistical analysis.

![Blink Detection Demo](https://github.com/username/blink_detection/raw/main/docs/demo.gif)

## Overview

This Python-based system detects and analyzes eye blinks in real-time using computer vision and deep learning techniques. It combines facial recognition, eye detection, and blink detection functionalities to track eye movement patterns.

### Technology Stack

- Python, PyTorch, OpenCV, dlib (optional), NumPy
- Computer vision algorithms (face/eye detection, landmark tracking)
- Deep learning models (CNN, RNN)

### Detection Methods

1. **EAR (Eye Aspect Ratio) Method**:

   - Calculates eye aspect ratio using facial landmarks
   - Detects blinks based on threshold values
   - Uses dlib library (with OpenCV fallback if unavailable)
   - Simple yet effective method that works without training models

2. **CNN Model**:

   - Image-based classification model for eye state detection
   - Detects open/closed eye states in single frames

3. **RNN Model**:
   - Temporal data-based blink detection
   - Combines CNN feature extractor with LSTM architecture

## Key Features

- Real-time eye blink detection
- Blink frequency measurement and statistical analysis
- Multiple model support (CNN, RNN, EAR-based)
- Video processing and result visualization
- User-friendly interface with keyboard shortcuts
- Adjustable detection thresholds for different environments

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/username/blink_detection.git
cd blink_detection
```

### 2. Set Up a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install Required Packages

```bash
# Install necessary packages
pip install torch torchvision opencv-python numpy matplotlib pandas scikit-learn tqdm
```

### 4. Install dlib (Optional, Recommended for EAR-based Method)

```bash
pip install dlib
```

If you have difficulties installing dlib, the EAR-based method also supports an alternative mode using OpenCV cascade classifiers.

### 5. Download Facial Landmark Model (For EAR Method)

Download dlib's facial landmark model and save it to the project root directory:

```bash
# Download link: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Extract after downloading and save to the project root directory
```

## Usage

### Real-time Eye Blink Detection

```bash
# EAR-based method (no trained model required)
python run_blink_detection.py detect --model_type ear

# Run with specific EAR threshold
python run_blink_detection.py detect --model_type ear --ear_threshold 0.12

# Use CNN model (pre-trained model required)
python run_blink_detection.py detect --model_type cnn --model_path /path/to/model.pth

# Process video file and save results
python run_blink_detection.py detect --model_type ear --video_path /path/to/video.mp4 --output_path /path/to/output.mp4
```

### Program Controls

During execution, you can use the following keys to control the program:

- `+` or `=`: Increase EAR threshold (more sensitive)
- `-`: Decrease EAR threshold (less sensitive)
- `r`: Reset blink counter
- `q`: Quit program

### Model Training (Coming Soon)

```bash
# Train CNN model with dataset
python -m src.models.train --model_type cnn --data_path /path/to/dataset --batch_size 32 --epochs 50 --save_dir ./output
```

## Dataset Preparation

Prepare your dataset with the following directory structure:

- CNN Model:

  ```
  dataset_root/
  ├── train/
  │   ├── open/
  │   │   ├── image1.jpg
  │   │   └── ...
  │   └── closed/
  │       ├── image1.jpg
  │       └── ...
  ├── val/
  │   ├── open/
  │   │   └── ...
  │   └── closed/
  │       └── ...
  └── test/
      ├── open/
      │   └── ...
      └── closed/
          └── ...
  ```

- RNN Model:

  ```
  dataset_root/
  ├── frames/
  │   ├── video1/
  │   │   ├── frame_000001.jpg
  │   │   └── ...
  │   └── video2/
  │       └── ...
  ├── train_sequences.csv
  ├── val_sequences.csv
  └── test_sequences.csv
  ```

- EAR Model:
  ```
  ear_data.csv  # Format: timestamp, ear_value, blink_label
  ```

## Project Structure

```
blink_detection/
├── src/
│   ├── data/
│   │   └── datasets.py        # Dataset classes
│   ├── models/
│   │   ├── blink_detection_model.py  # Model definitions
│   │   └── train.py           # Model training code
│   ├── utils/
│   │   ├── face_utils.py      # Face and eye detection utilities
│   │   └── data_utils.py      # Data processing utilities
│   └── detect_blinks.py       # Eye blink detection logic
├── run_blink_detection.py     # Main execution script
└── README.md                  # This document
```

### Key Files and Functions

- `run_blink_detection.py`: CLI interface, command parsing
- `src/detect_blinks.py`: Real-time eye blink detection algorithms, multiple detection method implementations
- `src/models/blink_detection_model.py`: Model class definitions
- `src/utils/face_utils.py`: Face/eye detection and EAR calculation functions
- `src/utils/data_utils.py`: Data preprocessing, augmentation, sequence generation functions

## Performance and Limitations

- **Advantages**

  - EAR method works quickly and effectively without requiring trained models
  - Real-time threshold adjustment allows adaptation to various environments
  - Multiple detection methods support optimization for different use cases

- **Limitations**
  - Accuracy may vary depending on environmental conditions (lighting, glasses, etc.)
  - Slightly reduced accuracy when using OpenCV without dlib
  - Trade-off between processing speed and accuracy in real-time use

## Applications

- Eye fatigue measurement and monitoring
- Attention tracking systems
- Drowsy driving prevention
- Medical research and diagnostic assistance tools

## Known Issues and Future Improvements

- Algorithm improvements for accurate eye detection without dlib
- Enhanced accuracy in various environments (lighting, glasses, etc.)
- Mobile platform support
- Complete implementation of model training functionality

## How to Contribute

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License

## Contact

If you have questions or suggestions about the project, please open an issue or contact us via email.
