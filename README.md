Drone Denoise: Propeller Noise Removal from Mixed Audio Recordings
Overview

This project implements a neural network-based audio denoising system that removes drone propeller noise from audio recordings while preserving human speech and traffic sounds. It is built with PyTorch and torchaudio, and supports both training with real recordings and generating synthetic mixtures on the fly.

Directory Structure
drone_denoise/
├── data/
│   ├── drone/         - Contains drone-only WAV files
│   ├── human/         - Contains human/traffic-only WAV files
│   └── mixture/       - Contains real or synthetic mixtures for testing
├── checkpoints/       - Contains saved model weights (epoch1.pth, epoch100.pth, etc.)
├── src/
│   ├── dataset_on_the_fly.py - Dataset class that generates training pairs dynamically
│   ├── model.py              - Conv1D denoiser model definition
│   ├── train.py              - Training script
│   ├── infer.py              - Inference script to denoise audio
│   ├── post_filter.py        - Optional script for notch/high-pass filtering
│   └── eval_sisnri.py        - Evaluation script using SI-SNR improvement
├── README.md          - Project documentation (this file)
└── venv/              - Python virtual environment (not version controlled)

Requirements

Python 3.8 or later

PyTorch with MPS (Apple Silicon) or CUDA (Linux/Windows)

torchaudio

numpy

scipy

tqdm

matplotlib

soundfile

Installation and Setup

Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate


Install dependencies:

pip install --upgrade pip
pip install torch torchaudio numpy scipy tqdm matplotlib soundfile

Data Preparation

Place drone-only recordings in: data/drone/

Place human/traffic-only recordings in: data/human/

Optionally, place real mixtures in: data/mixture/

All audio should be WAV, mono, 16 kHz sample rate, at least 10 seconds long.

Synthetic Data Generation

Synthetic mixtures can be generated automatically during training. The dataset pairs random drone and human clips, applies random SNR, and normalizes output. No precomputed mixtures are required.

Training the Model

To train the model on synthetic mixtures:

python src/train.py \
  --data_dir data \
  --save_dir checkpoints \
  --epochs 100 \
  --batch_size 8 \
  --lr 1e-3


Checkpoints will be saved to the checkpoints/ directory after each epoch.

Denoising New Audio Files

To apply a trained model to a mixture:

python src/infer.py \
  --checkpoint checkpoints/epoch100.pth \
  --input data/mixture/your_file.wav \
  --output denoised.wav

Evaluation

To compute SI-SNR improvement on synthetic test pairs:

python src/eval_sisnri.py --checkpoint checkpoints/epoch100.pth --N 30

Post-Filtering (Optional)

After inference, optional notch and high-pass filtering can be applied to further reduce tonal drone components:

python src/post_filter.py --input denoised.wav --output denoised_filtered.wav

Improvement Directions

Switch to spectrogram-based U-Net architectures for better separation

Add validation and evaluation metrics beyond MSE

Explore real-time processing and lightweight models for edge devices

Increase dataset diversity with augmentation

Author

Developed by Vivaan Chadha. Intended for research, prototyping, and educational purposes.