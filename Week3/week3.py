# imports
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from pytorch3d.datasets import ShapeNetCore
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.ops import GraphConv
from multiprocessing import freeze_support

# global metadata
ROOT = "Dataset/ShapeNetCore"      
TRAIN_RATIO = 0.8
BATCH_SIZE = 8
NUM_WORKERS = 4
SEED = 42
dataset = ShapeNetCore(ROOT, version=2, load_textures=False)                  
CLASS_NAMES = sorted(dataset.synset_inv.keys())          # e.g. ['airplane','chair',...]
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(CLASS_NAMES)}
SYNSET_TO_IDX = {dataset.synset_inv[lbl]: i for i, lbl in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)  

def collate_meshes_no_textures(batch): # collate function for DataLoader
    mesh_list, labels = [], []
    for item in batch:  # ShapeNetCore returns dicts per your class
        verts, faces = item["verts"], item["faces"]
        mesh_list.append(Meshes(verts=[verts], faces=[faces]))
        li = LABEL_TO_IDX.get(item.get("label",""),
             SYNSET_TO_IDX.get(item.get("synset_id",""), -1))
        labels.append(li)
    y = torch.tensor(labels, dtype=torch.long)
    # Hard assert: all labels mapped
    if (y < 0).any() or (y >= NUM_CLASSES).any():
        bad = y[(y < 0) | (y >= NUM_CLASSES)]
        raise RuntimeError(f"Collate produced out-of-range labels: {bad.tolist()}")
    return join_meshes_as_batch(mesh_list), y

# GCN model for mesh classification
class MeshGCN(nn.Module): 
    def __init__(self, num_classes: int):
        super().__init__()
        self.g1 = GraphConv(3, 64)
        self.g2 = GraphConv(64, 128)
        self.g3 = GraphConv(128, 128)
        self.head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    def forward(self, meshes: Meshes): # forward pass
        # uses minibaches with pytorch3d packed format
        x = meshes.verts_packed() 
        edges = meshes.edges_packed() 
        m_idx = meshes.verts_packed_to_mesh_idx()

        # three graph conv layers with ReLU
        x = torch.relu(self.g1(x, edges))
        x = torch.relu(self.g2(x, edges))
        x = torch.relu(self.g3(x, edges))

        # global mean pooling
        B, D = len(meshes), x.size(1)
        device = x.device
        sums   = torch.zeros((B, D), device=device)
        counts = torch.zeros(B, device=device)
        sums.index_add_(0, m_idx, x)
        counts.index_add_(0, m_idx, torch.ones_like(m_idx, dtype=x.dtype))
        global_feat = sums / counts.clamp_min(1e-6).unsqueeze(-1)
        
        return self.head(global_feat)

# training and evaluation loop
def run_epoch(model, loader, optimizer=None, device="cpu", epoch_tag="train"): 
    is_train = optimizer is not None # training if optimizer is given else validation
    model.train(is_train)
    crit = nn.CrossEntropyLoss()

    total_loss = total_acc = 0.0
    total_samples = 0
    printed = 0

    for step, (meshes, y) in enumerate(loader): #

        meshes = meshes.to(device) 
        y = y.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(meshes)
            loss = crit(logits, y)

        if is_train: # backpropagation and optimization step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        bs = y.size(0)
        total_samples += bs
        total_loss += loss.item() * bs
        total_acc  += (logits.argmax(1) == y).float().sum().item()

    if total_samples == 0: # avoid division by zero 
        print(f"[{epoch_tag}] WARNING: no valid samples accumulated -> returning NaN")
        return float("nan"), float("nan")
    return total_loss / total_samples, total_acc / total_samples

# main function
def main():
    
    # load dataset
    dataset = ShapeNetCore(ROOT, version=2, load_textures=False)       
    # set up dataset info           
    CLASS_NAMES = sorted(dataset.synset_inv.keys())
    LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(CLASS_NAMES)}
    SYNSET_TO_IDX = {dataset.synset_inv[lbl]: i for i, lbl in enumerate(CLASS_NAMES)}
    NUM_CLASSES = len(CLASS_NAMES)  
    
    # create DataLoaders
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_meshes_no_textures)

    # test dataset loader
    meshes,y = next(iter(loader))
    print(meshes)

    # split dataset into train and val sets
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size

    train_set, val_set = random_split(dataset, [train_size,val_size], generator=torch.Generator().manual_seed(SEED))

    # create DataLoaders for train and val sets
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle = True,
                            num_workers=NUM_WORKERS, collate_fn=collate_meshes_no_textures)

    val_loader = DataLoader(val_set,batch_size=BATCH_SIZE, shuffle = False,
                            num_workers=NUM_WORKERS, collate_fn=collate_meshes_no_textures)

    # print dataset info
    print(f"Detected {NUM_CLASSES} classes:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i:2d}: {name} ({dataset.synset_inv[name]})")

    # create model
    device = "cuda" if torch.cuda.is_available() else "cpu" # use GPU if available
    model = MeshGCN(NUM_CLASSES).to(device) # create model and move to device
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,weight_decay=1e-4)
    EPOCHS = 15

    best_val_acc = 0

    # training loop
    for epoch in range(1,EPOCHS + 1):
        
        # one epoch of training and validation
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer=optimizer, device=device, epoch_tag="train")
        va_loss, va_acc = run_epoch(model, val_loader, optimizer=None, device=device, epoch_tag="val")
        dt = time.time() - t0
        
        # print epoch stats
        print(f"[{epoch:02d}/{EPOCHS}] "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f} | {dt:.1f}s")
        
        if va_acc > best_val_acc: # save best model
            best_val_acc = va_acc
            torch.save({"model":model.state_dict(),
                    "val_acc":va_acc}, "Week3/meshgcn_best.pt")
            
        print(f"Best val acc: {best_val_acc:.3f}") # print best val acc

if __name__ == "__main__": # main entry
    freeze_support()  # for Windows support
     # If you still get a spawn error, set NUM_WORKERS = 0 at the top and try again.
    main()