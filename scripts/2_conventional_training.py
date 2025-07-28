import os
import sys
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from progressbar import progressbar
from matplotlib.colors import ListedColormap

sys.path.append(".")
from src.data.datasetup import DataSetup
from src.deeplearning.loss_functions import DiceLoss
from src.deeplearning.model_architectures import AGT_UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_setup = DataSetup(
                       experiment_name="S_160_4.7",
                       transforms_name="minimal", 
                       tip_size=2) 
dataloader_train, dataloader_val = data_setup.dataloaders["training"], data_setup.dataloaders["validation"]

# set up model and optimizer
model = AGT_UNet(in_ch=2, out_ch=1, dropout_prob=0.3).to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-4, amsgrad=True)
criterion = DiceLoss()

# set up storage
train_loss, val_loss =[], []
min_val_loss, min_val_loss_epoch = 1, 0
storage_folder = "results/tests/conventional_training"
os.makedirs(f"{storage_folder}/models", exist_ok=True)
os.makedirs(f"{storage_folder}/plots", exist_ok=True)
os.makedirs(f"{storage_folder}/data", exist_ok=True)

# clear models folder if there are any models from previous runs
for file in os.listdir(f"{storage_folder}/models"):
    os.remove(os.path.join(f"{storage_folder}/models", file))

# train the model
epochs = 100
for epoch in range(epochs):
    train_loss_epoch, val_loss_epoch = [], []
    
    # train
    model.train()
    for data, target, expl in dataloader_train:
        optimizer.zero_grad()
        
        input_data = data.to(device).float()
        target_seg = target.to(device)
        
        pred = model(input_data)
        
        loss = criterion(torch.sigmoid(pred), target_seg)
        loss.backward()
        optimizer.step()

        train_loss_epoch.append(loss.item())    
    train_loss.append(np.mean(train_loss_epoch))

    # validate
    model.eval()
    with torch.no_grad():
        for data, target, expl in dataloader_val:
            input_data = data.to(device).float()
            target_seg = target.to(device)
            
            pred = model(input_data)
            
            loss = criterion(torch.sigmoid(pred), target_seg)
            
            val_loss_epoch.append(loss.item())
    val_loss.append(np.mean(val_loss_epoch))

    # store lowest validation loss for "early stopping"
    if val_loss[-1] < min_val_loss:
        min_val_loss = val_loss[-1]
        min_val_loss_epoch = epoch
        torch.save(model.state_dict(), f"{storage_folder}/models/model_epoch_{epoch}.pt")

    # print loss
    padded_epoch_str = str(epoch).zfill(len(str(epochs)))
    print(f"Epoch {padded_epoch_str} - Training loss: {train_loss[-1]:.4f}, Validation loss: {val_loss[-1]:.4f}", flush=True)

# save final model
torch.save(model.state_dict(), f"{storage_folder}/models/model_final.pt")

# save losses
np.save(f"{storage_folder}/data/train_loss.npy", train_loss)
np.save(f"{storage_folder}/data/val_loss.npy", val_loss)

# visualize performance
random_index = np.random.randint(0, dataloader_val.batch_size)

plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 20
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["savefig.dpi"] = 300

input, target, _ = next(iter(dataloader_val))
prediction = model(input.to(device).float())
prediction = torch.sigmoid(prediction).cpu().detach()
segmentation = prediction[random_index][0].squeeze()
segmentation = torch.where(segmentation > 0.5, 1, 0).numpy()

segmentation_cmap = ListedColormap(["none", "lime"])
segmentation_alpha = np.where(segmentation > 0, 0.6, 0)

target_cmap = ListedColormap(["none", "black"])
target_alpha = np.where(target[random_index][0] > 0, 1., 0)

fig, ax = plt.subplots(1,3, figsize=(15,5))
fig.tight_layout(pad=2.0)

ax[0].plot(train_loss, label="Training Loss (prediction)", marker="^", color="tab:blue", linewidth=.4, markersize=1.)
ax[0].plot(val_loss, label="Validation Loss (prediction)", marker="2", color="tab:orange", linewidth=.4, markersize=1.)
ax[0].set_xlabel("Epoch [a.u.]")
ax[0].set_ylabel("Loss [a.u.]")
ax[0].axvline(min_val_loss_epoch, color="red", label=f"Lowest loss epoch: {min_val_loss_epoch} - {min_val_loss:.2f}")
ax[0].legend(loc="lower left", fontsize=12)
ax[0].set_title("Training and Validation Loss")

ax[1].imshow(input[random_index][0].squeeze(), cmap="coolwarm", vmin=-3, vmax=3)
ax[1].set_title("Displacements $u_x$")
ax[1].axis("off")

ax[2].imshow(np.ones_like(segmentation), cmap="gray", vmin=0, vmax=1)  # white background
ax[2].imshow(segmentation, cmap=segmentation_cmap, alpha=segmentation_alpha)
ax[2].imshow(target[random_index][0].squeeze(), cmap=target_cmap, alpha=target_alpha)
ax[2].set_title("Crack Tip Segmentation")
handles = [plt.Line2D([0], [0], color="black", lw=4, label="Target"),
           plt.Line2D([0], [0], color="lime", lw=4, label="Segmentation")]
ax[2].legend(handles=handles, loc="upper right", fontsize=12)
ax[2].set_xticks([])
ax[2].set_yticks([])

ax[0].set_aspect("auto")
bbox0 = ax[0].get_position()
bbox1 = ax[1].get_position()
ax[0].set_position([bbox0.x0, bbox0.y0, bbox1.width, bbox1.height])

plt.savefig(f"{storage_folder}/plots/training_summary.png")
print(f"Training completed. Plots and models saved in {storage_folder}.")