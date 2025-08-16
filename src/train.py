import os, torch
from torch.utils.data import DataLoader
from dataset_on_the_fly import OnTheFlyMixDataset
from model import ConvDenoiser
from tqdm import tqdm

def train(data_dir, save_dir, epochs=100, bs=8, lr=1e-3):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    ds = OnTheFlyMixDataset(
        drone_dir=os.path.join(data_dir, 'drone'),
        human_dir=os.path.join(data_dir, 'human'),
        sr=16000,
        segment_sec=10,
        snr_range_db=(-5, 5)
    )
    # macOS/MPS: num_workers=0 to avoid exit hang; pin_memory not useful on MPS
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)

    model = ConvDenoiser().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for mix, human in tqdm(dl, desc=f"Epoch {epoch}/{epochs}", leave=False):
            mix, human = mix.to(device), human.to(device)
            pred = model(mix)
            loss = loss_fn(pred, human)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()

        avg = running / max(len(dl), 1)
        print(f"Epoch {epoch} | Loss: {avg:.6f}")
        torch.save(model.state_dict(), f"{save_dir}/epoch{epoch}.pth")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data')
    p.add_argument('--save_dir', default='checkpoints')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    args = p.parse_args()
    train(args.data_dir, args.save_dir, args.epochs, args.batch_size, args.lr)
