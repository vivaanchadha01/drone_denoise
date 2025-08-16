import os, glob
import torch
import torchaudio
import torch.nn.functional as F

def split_and_save(in_dir="data/human", sr=16000, chunk_sec=10):
    chunk_len = sr * chunk_sec

    files = glob.glob(os.path.join(in_dir, "*.wav")) + glob.glob(os.path.join(in_dir, "*.WAV"))
    if not files:
        print(f"No WAV files found in {in_dir}")
        return

    for path in files:
        wav, sr_in = torchaudio.load(path)
        wav = wav.mean(0, keepdim=True)  # convert to mono
        if sr_in != sr:
            wav = torchaudio.transforms.Resample(sr_in, sr)(wav)

        base = os.path.splitext(os.path.basename(path))[0]

        num_chunks = (wav.size(1) + chunk_len - 1) // chunk_len
        for i in range(num_chunks):
            start = i * chunk_len
            end = min(start + chunk_len, wav.size(1))
            chunk = wav[:, start:end]

            # pad last chunk if shorter
            if chunk.size(1) < chunk_len:
                chunk = F.pad(chunk, (0, chunk_len - chunk.size(1)))

            out_path = os.path.join(in_dir, f"{base}_part{i+1}.wav")
            torchaudio.save(out_path, chunk.detach(), sr)
            print(f"Saved {out_path}")

if __name__ == "__main__":
    split_and_save()
