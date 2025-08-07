import os, random, torch, torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset

class DenoiseDataset(Dataset):
    def __init__(self, mix_dir, human_dir, sr=16000, segment=10.0):
        self.mix_files   = [os.path.join(mix_dir, f)   for f in os.listdir(mix_dir)   if f.endswith('.wav')]
        self.human_files = [os.path.join(human_dir, f) for f in os.listdir(human_dir) if f.endswith('.wav')]
        self.sr = sr
        self.seg_len = int(sr * segment)

    def __len__(self):
        return len(self.mix_files)

    def __getitem__(self, idx):
        mix_path   = self.mix_files[idx]
        human_path = random.choice(self.human_files)

        # load + mono + resample
        mix, sr1   = torchaudio.load(mix_path)
        human, sr2 = torchaudio.load(human_path)

        if sr1 != self.sr:
            mix = torchaudio.transforms.Resample(sr1, self.sr)(mix)
        if sr2 != self.sr:
            human = torchaudio.transforms.Resample(sr2, self.sr)(human)

        mix   = mix.mean(0, keepdim=True)
        human = human.mean(0, keepdim=True)

        L = self.seg_len

        # pad shorter signals
        if mix.size(1) < L:
            mix = F.pad(mix, (0, L - mix.size(1)))
        if human.size(1) < L:
            human = F.pad(human, (0, L - human.size(1)))

        # truncate (or leave) to exactly L
        mix   = mix[:, :L]
        human = human[:, :L]

        return mix, human
