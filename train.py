import shutil
import os
import torch
from torch.utils.data import DataLoader

bin_path = "/content/drive/MyDrive/train1.bin"
config = {
    "context_len": 512,
    "batch_size": 192, # Increased batch size (greater than 128, less than 256)
    "lr": 3e-4,
    "epochs": 1,
    "save_every": 500
}

torch.manual_seed(42)

def save_state(model, optimizer, epoch, batch_idx, paths="/content/drive/MyDrive/tpt_checkpoint1.pt"):
    state = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "batch_size": config["batch_size"],
        "path": paths
    }
    torch.save(state, paths)

source_path = "/content/drive/MyDrive/train1.bin"
local_path = "/content/train.bin"

if not os.path.exists(local_path):
    print("Copying dataset to local runtime...")
    shutil.copy(source_path, local_path)

dataset = dset_byte_stream(local_path, seq_len=config["context_len"])

dataloader = DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    drop_last=True,
    pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_v1 = TPT_v1(tpt_config).to(device)
optimizer = torch.optim.AdamW(model_v1.parameters(), lr=config["lr"])
criterion = torch.nn.CrossEntropyLoss()


model_v1.train()

for epoch in range(config["epochs"]):
    for batch_idx, (xb, yb) in enumerate(dataloader):
        xb, yb = xb.to(device).long(), yb.to(device).long()
        optimizer.zero_grad(set_to_none=True)

        # bfloat16 
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model_v1(xb)
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_v1.parameters(), max_norm=1.0)
        optimizer.step()

        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

        if batch_idx > 0 and batch_idx % config["save_every"] == 0:
            save_state(model_v1, optimizer, epoch, batch_idx)

        if batch_idx == 23400:
            save_state(model_v1, optimizer, epoch, batch_idx, paths="/content/drive/MyDrive/tpt_checkpoint1.pt")
            print("Main threshold reached and saved.")