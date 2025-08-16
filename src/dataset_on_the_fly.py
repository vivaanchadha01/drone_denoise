import os, glob, random, torch
import torchaudio
import torch.nn.functional as F

class OnTheFlyMixDataset(torch.utils.data.Dataset):
    def __init__(self, drone_dir="data/drone", human_dir="data/human",
                 sr=16000, segment_sec=10, snr_range_db=(-5, 5)):
        self.drone_files = sorted(glob.glob(os.path.join(drone_dir, "*.wav")) +
                                  glob.glob(os.path.join(drone_dir, "*.WAV")))
        self.human_files = sorted(glob.glob(os.path.join(human_dir, "*.wav")) +
                                  glob.glob(os.path.join(human_dir, "*.WAV")))
        if not self.drone_files:
            raise RuntimeError(f"No drone files found in {drone_dir}")
        if not self.human_files:
            raise RuntimeError(f"No human files found in {human_dir}")
        self.sr = sr
        self.L = int(sr * segment_sec)
        self.snr_range_db = snr_range_db

    def __len__(self):
        # Large virtual length to provide variety each epoch
        return max(len(self.drone_files), len(self.human_files)) * 50

    def _load_mono(self, path):
        x, srx = torchaudio.load(path)
        if x.size(0) > 1:
            x = x.mean(0, keepdim=True)
        if srx != self.sr:
            x = torchaudio.transforms.Resample(srx, self.sr)(x)
        return x

    def _segment(self, x):
        # pad or randomly crop to exactly L samples
        if x.size(1) < self.L:
            x = F.pad(x, (0, self.L - x.size(1)))
        else:
            start = random.randint(0, max(0, x.size(1) - self.L))
            x = x[:, start:start+self.L]
        return x

    def __getitem__(self, _):
        d_path = random.choice(self.drone_files)
        h_path = random.choice(self.human_files)

        drone = self._segment(self._load_mono(d_path))
        human = self._segment(self._load_mono(h_path))

        # Mix at random SNR: mix = human + scaled(drone)
        snr_db = random.uniform(*self.snr_range_db)
        rms_h = human.pow(2).mean().sqrt().clamp(min=1e-6)
        rms_d = drone.pow(2).mean().sqrt().clamp(min=1e-6)
        factor = rms_h / (rms_d * (10 ** (snr_db / 20.0)))
        mix = human + drone * factor

        # Peak-normalize both so target/mix are on the same scale
        peak = mix.abs().max().clamp(min=1e-4)
        mix = mix / peak
        human = human / peak

        return mix, human
