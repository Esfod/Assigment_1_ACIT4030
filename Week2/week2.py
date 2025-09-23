# imports
from pathlib import Path
import numpy as np
from read_binvox import read_binvox as rb
import random
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import torch.nn as nn

from multiprocessing import freeze_support

def Setup_Voxel_Classifier_Dataset(rootPath, filetype = "*surface.binvox", pool = 2): # or "*surface.binvox" "*solid.binvox"
    #goes trough each class folder and each model folder to find binvox files
    #returns list of (voxel_tensor, label) pairs
    dataset = []
    i = 0
    t = 0
    for class_dir in rootPath.iterdir():
        if class_dir.is_dir():
            label = class_dir.name
            i = 0
            for model_dir in class_dir.iterdir():
                if model_dir.is_dir():
                    models_subdir = model_dir / "models"
                    if models_subdir.exists():
                        for f in models_subdir.glob(filetype):
                            # print("found binvox",f)
                            vol = rb(f)

                            x = torch.tensor(np.asarray(vol), dtype=torch.float32).squeeze(0)

                            #print(x.ndim)
                            if x.ndim == 3:                      # (D,H,W) -> add channel
                                x = x.unsqueeze(0)
                            elif x.ndim == 5:                    # (1,1,D,H,W) -> drop batch
                                x = x.squeeze(0)
                            #print(x.ndim)
                            
                            
                            dataset.append((x,label))
                            i = i + 1
                            #print("model nr:",i)
            print("class :", label)
            if i > 0:
                t += i + 1
                print("Number of model in class :", i,"\n")
    print("Number of Files :", t)
    return dataset


# Prints out voxel in with mathplot
def visualize_voxel_grid(vol):
    import numpy as np
    # Accept torch or numpy; accept (1,D,H,W) or (D,H,W)
    if torch.is_tensor(vol):
        vol = vol.detach().to('cpu').numpy()
    vol = np.asarray(vol)
    if vol.ndim == 4:  # (C,D,H,W) or (1,D,H,W)
        vol = vol.squeeze(0)
    assert vol.ndim == 3, f"Expected (D,H,W), got {vol.shape}"

    D,H,W = vol.shape
    # Use ax.voxels for small grids; scatter for large
    if D*H*W <= 40**3:
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(vol > 0, edgecolor='k', linewidth=0.1)
        ax.set_title(f'Voxels ({D}×{H}×{W})')
        plt.show()
    else:
        coords = np.argwhere(vol > 0)  # (Z,Y,X) = (D,H,W)
        if len(coords) > 50_000:
            idx = np.random.choice(len(coords), 50_000, replace=False)
            coords = coords[idx]
        z, y, x = coords[:,0], coords[:,1], coords[:,2]
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, s=1, alpha=0.6)
        try: ax.set_box_aspect((W, H, D))
        except: pass
        ax.set_title(f'Occupied voxels ({D}×{H}×{W})')
        plt.show()



# Simple 3D CNN classifier for voxel grids
class Voxel_CNN_Classifer(nn.Module): 
    def __init__(self, num_classes : int):
        super().__init__()
        self.stem = nn.Sequential(
        # (B,1,128,128,128) -> (B,16,64,64,64)
        nn.Conv3d(1,16, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(16),
        nn.ReLU(inplace=True),
        
        # (B,16,64,64,64) -> (B,16,32,32,32)
        nn.MaxPool3d(2),

        # (B,16,32,32,32) -> (B,32,32,32,32)
        nn.Conv3d(16,32,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm3d(32),
        nn.ReLU(inplace=True),
        
        # (B,32,32,32,32) -> (B,32,16,16,16)
        nn.MaxPool3d(2),
        
        # (B,32,16,16,16) -> (B,64,16,16,16)
        nn.Conv3d(32,64,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm3d(64),
        nn.ReLU(inplace=True),

        # (B,64,16,16,16) -> (B,64,8,8,8)
        nn.MaxPool3d(2),

        # (B,64,8,8,8) -> (B,128,8,8,8)
        nn.Conv3d(64,128,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm3d(128),
        nn.ReLU(inplace=True),
        # (B,128,8,8,8) -> (B,128,1,1,1)
        nn.AdaptiveAvgPool3d(1)
        )
        self.head = nn.Linear(128,num_classes)

    def forward(self,x):
        x = self.stem(x)
        x = x.flatten(1)
        return self.head(x)
    

#simple wrapper
class ListVoxelDataset(Dataset): #
    def __init__(self, pairs, class_to_idx):
        self.pairs = pairs
        self.class_to_idx = class_to_idx
        
    def __len__(self): return len(self.pairs)
    def __getitem__(self,i):
        x, lbl = self.pairs[i]
        y = self.class_to_idx[lbl]
        return x, torch.tensor(y, dtype=torch.long)


def main():
    EPOCHS = 15
    SEED = 42 # seed to split equaly for all models to work on same training and val data
    TRAIN_RATIO = 0.8
    NUM_CLASSES = 5
    NUM_WORKERS = 4
    lr = 1e-3
    weight_decay = 1e-4
    rootPath = Path("ShapeNetCore")

    #load dataset
    dataset = Setup_Voxel_Classifier_Dataset(rootPath)
    
    # 
    classes = sorted({lbl for _, lbl in dataset})
    class_to_idx = {c:i for i,c in enumerate(classes)}
    NUM_CLASSES = len(classes)


    # split dataset into train and val sets
    from torch.utils.data import random_split

    full_ds = ListVoxelDataset(dataset, class_to_idx)
    
    train_size = int(TRAIN_RATIO * len(full_ds))
    val_size   = len(full_ds) - train_size

    train_set, val_set = torch.utils.data.random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

    #crate torch Dataloaders
    train_loader = DataLoader(train_set, batch_size=2,shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"classes: {classes}")

    # test dataset
    vox_t, label = random.choice(dataset)   # vox_t: (1, D, H, W) torch.Tensor
    print("Picked class:", label, "\tTensor shape:", tuple(vox_t.shape))

    # visualize the voxel grid
    visualize_voxel_grid(vox_t)             # the function handles torch 4D

    # create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Voxel_CNN_Classifer(NUM_CLASSES).to(device)

    # setup training loop
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for multi-class classification
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) # AdamW optimizer
    
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS+1): # loop over the dataset epoch amount times
        model.train() 
        running = 0.0
        for xb, yb in val_loader:  
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(train_set)

        model.eval()
        
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        val_acc = correct / max(1, total)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "Week2/best_voxel_cnn.pt")
            
        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} | val acc {val_acc:.3f}")
    

if __name__ == "__main__":
    freeze_support()  # for Windows support
     # If you still get a spawn error, set NUM_WORKERS = 0 at the top and try again.
    main()