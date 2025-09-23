# imports
from torch.utils.data import DataLoader, random_split
from pytorch3d.datasets import ShapeNetCore
import torch
import torch.nn as nn 
from multiprocessing import freeze_support

# Some Metadata
dataset_path = "Dataset/ShapeNetCore"
train_ratio = 0.8 # 80% training 20% validation
max_points = 700 # max points to sample from each mesh
device = "cuda" if torch.cuda.is_available() else "cpu" # use GPU if available
NUM_WORKERS = 8  
PIN_MEMORY = (device == "cuda")  # pin_memory only makes sense when using GPU
print(device)

dataset = ShapeNetCore(dataset_path, version=2, load_textures= False)

# Get class info
present_sids = sorted(dataset.synset_ids)
sid2idx      = {sid: i for i, sid in enumerate(present_sids)}
NUM_CLASSES  = len(present_sids)

# Custom collate
def custom_collate_fn(batch): 
    verts_list = []
    labels = []

    for sample in batch: 
        verts = sample['verts']              # [V, 3] (CPU tensor)
        n = verts.shape[0]

        # --- random sampling to exactly max_points ---
        if n >= max_points:
            idx = torch.randperm(n)[:max_points]
        else:
            # sample WITH replacement to reach max_points
            idx = torch.randint(0, n, (max_points,))
        pts = verts[idx]                     # [max_points, 3]
        
        # --- normalize to zero mean and fit in unit sphere ---
        pts = pts - pts.mean(0, keepdim=True)
        scale = pts.norm(p=2, dim=1).max()
        pts = pts / (scale + 1e-6)

        verts_list.append(pts)
        # in collate:
        labels.append(sid2idx[sample['synset_id']])

    batched_verts = torch.stack(verts_list, dim=0)        
    labels = torch.tensor(labels, dtype=torch.long)    
    return {'verts': batched_verts, 'labels': labels}


# PointNet model for point cloud classification
class PointNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU())
        
        self.fc1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32,num_classes)
        )
    def forward(self, x):
        x = x.transpose(2,1)
        
        x = self.mlp1(x)
        
        x = torch.max(x,2)[0]
        
        x = self.fc1(x)
        
        return x

# Setup run_epoch()
def run_epoch(model, loader, optimizer=None, device="cpu", epoch_tag="train", sid2idx=None):
    is_train = optimizer is not None
    model.train(is_train)

    crit = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader: # iterate over batches
        x = batch["verts"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)  

        if is_train:
            optimizer.zero_grad()

        logits = model(x)                  # (B, C)
        loss = crit(logits, y)

        if is_train:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            bs = y.size(0)
            total_samples += bs
            total_loss += loss.item() * bs

    avg_loss = total_loss / max(1, total_samples)
    avg_acc  = total_correct / max(1, total_samples)
    print(f"[{epoch_tag}] loss={avg_loss:.4f} | acc={avg_acc:.4f} | n={total_samples}")
    return avg_loss, avg_acc

def main():
    # load dataset
    dataset = ShapeNetCore(dataset_path, version=2, load_textures= False)

# Get class info
    present_sids = sorted(dataset.synset_ids)
    sid2idx      = {sid: i for i, sid in enumerate(present_sids)}
    NUM_CLASSES  = len(present_sids)

    print("Present classes on disk:", present_sids)
# Split dataset
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Setup dataloaders
    train_loader =  DataLoader(train_dataset, batch_size = 4, shuffle = True,  collate_fn = custom_collate_fn, num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY)
    val_loader   =  DataLoader(val_dataset  , batch_size = 4, shuffle = False, collate_fn = custom_collate_fn, num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY)

# Setup model, optimizer, training loop
    EPOCHS = 15
    lr = 3e-4
    weight_decay = 1e-4

    model = PointNet(NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr = lr, weight_decay=weight_decay)

    best_val_acc = 0

    for epoch in range(1,EPOCHS+1):
        tr_loss, tr_acc   = run_epoch(model, train_loader, optimizer, device, epoch_tag=f"train/ep{epoch}", sid2idx=sid2idx)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer=None, device=device, epoch_tag=f"val/ep{epoch}",sid2idx=sid2idx)

        
        if val_acc > best_val_acc: # save best model
            best_val_acc = val_acc
            torch.save(model.state_dict(), "Week5/best_pointnet.pt")
            print(f"saved new best (val_acc={best_val_acc:.4f})")

if __name__ == "__main__":
    # Required on Windows whenever DataLoader uses worker processes
    freeze_support()
    # If you still get a spawn error, set NUM_WORKERS = 0 at the top and try again.
    main()