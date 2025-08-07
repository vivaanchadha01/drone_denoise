import os, torch
from torch.utils.data import DataLoader
from dataset import DenoiseDataset
from model   import ConvDenoiser
from tqdm    import tqdm

def train(data_dir, save_dir, epochs=50, bs=8, lr=1e-3):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ds = DenoiseDataset(
        mix_dir=os.path.join(data_dir, 'mixture'),
        human_dir=os.path.join(data_dir, 'human')
    )
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)

    model = ConvDenoiser().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for mix, human in tqdm(dl, desc=f"Epoch {epoch}/{epochs}"):
            mix, human = mix.to(device), human.to(device)
            pred = model(mix)
            loss = loss_fn(pred, human)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(dl)
        print(f"Epoch {epoch} | Loss: {avg:.6f}")
        torch.save(model.state_dict(), f"{save_dir}/epoch{epoch}.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   default='data')
    parser.add_argument('--save_dir',   default='checkpoints')
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=8)
    parser.add_argument('--lr',         type=float, default=1e-3)
    args = parser.parse_args()
    train(args.data_dir, args.save_dir, args.epochs, args.batch_size, args.lr)
