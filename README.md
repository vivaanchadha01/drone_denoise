Drone Denoise: Propeller Noise Removal from Mixed Audio Recordings
------------------------------------------------------------------

Overview
--------

This project implements a neural network-based audio denoising system that removes drone propeller noise from audio recordings while preserving human and traffic background sounds. It uses PyTorch and torchaudio libraries and supports training with real and synthetically generated audio data.

Directory Structure
-------------------

drone_denoise/
├── data/
│   ├── drone/         - Contains drone-only WAV audio files
│   ├── human/         - Contains human/traffic-only WAV audio files
│   └── mixture/       - Contains drone+human mixed audio files (either real or synthetic)
├── checkpoints/       - Contains saved model weights for each epoch (e.g., epoch1.pth, epoch100.pth)
├── src/
│   ├── dataset.py     - PyTorch dataset class for loading training data
│   ├── model.py       - Conv1D autoencoder model definition
│   ├── train.py       - Model training script
│   ├── infer.py       - Inference script to denoise new audio files
│   └── synthetic.py   - Script to generate synthetic training mixtures with augmentation
├── README.txt         - Project documentation (this file)
└── venv/              - Python virtual environment (not included in version control)

Requirements
------------

- Python 3.8 or later
- PyTorch
- torchaudio
- numpy
- scipy
- tqdm
- soundfile

Installation and Setup
----------------------

1. Create and activate a virtual environment:

   python3 -m venv venv
   source venv/bin/activate

2. Install dependencies:

   pip install --upgrade pip
   pip install torch torchaudio numpy scipy tqdm soundfile

Data Preparation
----------------

1. Place your drone-only recordings in the folder:
   data/drone/

2. Place human/traffic-only background sounds in:
   data/human/

All audio files should be in WAV format, mono-channel, and at least 10 seconds long. The expected sample rate is 16000 Hz.

Synthetic Data Generation
-------------------------

To create synthetic drone-human mixtures with randomized augmentation and signal-to-noise ratios (SNR), run:

   python src/synthetic.py

This will populate the data/mixture/ folder with new training files.

Training the Model
------------------

To train the Conv1D denoising autoencoder on the mixture dataset:

   python src/train.py \
     --data_dir data \
     --save_dir checkpoints \
     --epochs 100 \
     --batch_size 8 \
     --lr 1e-3

Model weights will be saved in the checkpoints/ folder after each epoch.

Denoising New Audio Files
-------------------------

To apply a trained model to a new mixture audio file:

   python src/infer.py \
     --checkpoint checkpoints/epoch100.pth \
     --input data/mixture/your_file.wav \
     --output denoised.wav

This will output the denoised version of the selected file.

Improvement Directions
----------------------

- Transition to spectrogram-based U-Net architectures
- Add validation set and compute SNR improvement metrics
- Incorporate real-time processing support
- Explore model compression for edge deployment

Author
------

Developed by Vivaan Chadha. This project is intended for research, prototyping, and educational purposes.
