import os, torch, torchaudio
from model import ConvDenoiser

def denoise(checkpoint_path, input_path, output_path, sr=16000):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load model
    model = ConvDenoiser().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    
    # 2. Load and preprocess input
    mix, sr1 = torchaudio.load(input_path)
    if sr1 != sr:
        mix = torchaudio.transforms.Resample(sr1, sr)(mix)
    mix = mix.mean(0, keepdim=True).unsqueeze(0).to(device)  # [1,1,T]
    
    # 3. Denoise
    with torch.no_grad():
        out = model(mix)
    out = out.squeeze(0).cpu()  # [1,T]
    
    # 4. Write file
    torchaudio.save(output_path, out, sr)
    print(f"Denoised file saved to {output_path}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pth model checkpoint")
    p.add_argument("--input",      required=True,
                   help="Path to noisy (mixture) WAV")
    p.add_argument("--output",     default="denoised.wav",
                   help="Where to save cleaned WAV")
    args = p.parse_args()
    denoise(args.checkpoint, args.input, args.output)
