Drone Denoise: Propeller Noise Removal from Mixed Audio Recordings
Overview

This project implements a neural network–based audio denoising system that removes drone propeller noise from audio recordings while preserving human and traffic background sounds. It uses PyTorch and torchaudio and supports training from unpaired data by constructing synthetic pairs on the fly.

Directory Structure
drone_denoise/
├── data/
│   ├── drone/                   - drone-only WAV audio files
│   ├── human/                   - human/traffic-only WAV audio files
│   └── mixture/                 - optional real mixtures for testing
├── checkpoints/                 - saved model weights (epoch1.pth, epoch100.pth, etc.)
├── src/
│   ├── dataset_on_the_fly.py    - dataset that generates paired mixes during training
│   ├── model.py                 - Conv1D denoiser model definition
│   ├── train.py                 - model training script
│   ├── infer.py                 - inference script to denoise new audio files
│   ├── post_filter.py           - optional notch/high-pass cleanup after inference
│   └── eval_sisnri.py           - SI-SNR improvement evaluation
├── README.md                    - project documentation (this file)
└── venv/                        - Python virtual environment (not version controlled)

Requirements

Python 3.8 or later
PyTorch
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

Place drone-only recordings in data/drone/.
Place human/traffic-only recordings in data/human/.
Optionally place real mixtures in data/mixture/.
All audio should be WAV, mono, 16 kHz sample rate, and at least 10 seconds long.

Training the Model

Train using on-the-fly synthetic mixtures (no precomputed pairs required):

python src/train.py \
  --data_dir data \
  --save_dir checkpoints \
  --epochs 100 \
  --batch_size 8 \
  --lr 1e-3


Checkpoints are written to checkpoints/ after each epoch.

Denoising New Audio Files

Apply a trained model to a mixture:

python src/infer.py \
  --checkpoint checkpoints/epoch100.pth \
  --input data/mixture/your_file.wav \
  --output denoised.wav

Evaluation

Compute SI-SNR improvement on synthetic test pairs:

python src/eval_sisnri.py --checkpoint checkpoints/epoch100.pth --N 30

Post-Filtering (Optional)

Apply notch and high-pass filtering after inference to further reduce tonal remnants:

python src/post_filter.py --input denoised.wav --output denoised_filtered.wav

Improvement Directions

Potential next steps include moving to spectrogram-based U-Net architectures, adding a validation split and objective metrics (SI-SNR, SDR), exploring real-time processing, and optimizing for edge deployment via model compression and quantization.

Author

Developed by Vivaan Chadha for research, prototyping, and educational purposes.