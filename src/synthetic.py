import os
import glob
import random
import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.transforms as T

def augment(audio, sr):
    # Time stretch (simulate speed changes)
    if random.random() < 0.5:
        rate = random.uniform(0.9, 1.1)
        audio = T.Resample(orig_freq=sr, new_freq=int(sr * rate))(audio)
    
    # Pitch shift
    if random.random() < 0.5:
        shift = random.uniform(-2, 2)  # semitones
        audio = T.PitchShift(sr, n_steps=shift)(audio)
    
    # Volume scale
    scale = random.uniform(0.8, 1.2)
    audio = audio * scale

    # Random noise
    if random.random() < 0.5:
        noise = torch.randn_like(audio) * 0.005
        audio = audio + noise

    return audio

def generate_synthetic_mixtures(
    drone_dir='data/drone',
    human_dir='data/human',
    out_dir='data/mixture',
    sr=16000,
    segment=10,          # seconds
    mixes_per_drone=5,
    snr_range_db=(-5, 5)
):
    os.makedirs(out_dir, exist_ok=True)
    drone_files = glob.glob(os.path.join(drone_dir, '*.wav'))
    human_files = glob.glob(os.path.join(human_dir, '*.wav'))
    L = sr * segment

    for d_path in drone_files:
        drone_raw, _ = torchaudio.load(d_path)
        drone_raw = drone_raw.mean(0, keepdim=True)  # mono

        for i in range(mixes_per_drone):
            h_path = random.choice(human_files)
            human_raw, _ = torchaudio.load(h_path)
            human_raw = human_raw.mean(0, keepdim=True)

            # Augment both
            drone = augment(drone_raw.clone(), sr)
            human = augment(human_raw.clone(), sr)

            # Pad or crop to same length
            if drone.size(1) < L:
                drone = F.pad(drone, (0, L - drone.size(1)))
            else:
                drone = drone[:, :L]

            if human.size(1) < L:
                human = F.pad(human, (0, L - human.size(1)))
            else:
                human = human[:, :L]

            # Apply SNR
            snr_db = random.uniform(*snr_range_db)
            rms_h = human.pow(2).mean().sqrt()
            rms_d = drone.pow(2).mean().sqrt()
            factor = rms_h / (rms_d * 10**(snr_db / 20))
            mix = drone * factor + human
            mix = mix / mix.abs().max().clamp(min=1e-4)  # normalize

            # Save
            d_base = os.path.splitext(os.path.basename(d_path))[0]
            h_base = os.path.splitext(os.path.basename(h_path))[0]
            out_name = f"{d_base}__{h_base}__mix{i}__snr{int(snr_db)}.wav"
            out_path = os.path.join(out_dir, out_name)
            torchaudio.save(out_path, mix.detach(), sr)
            print(f"Saved {out_path}")

if __name__ == '__main__':
    generate_synthetic_mixtures()
