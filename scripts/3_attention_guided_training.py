import os, sys, json
import torch
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
from progressbar import progressbar
from matplotlib.colors import ListedColormap

sys.path.append(".")
from src.data.datasetup import DataSetup
from src.deeplearning.model_architectures import AGT_UNet
from src.explainability.methods.GradCAMpp import GradCAMpp
from src.deeplearning.loss_functions import DiceLoss, CSILoss

# configuration parameters we expose for each run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_params = {
    "experiment_name": "attention_guided_training",
    "device": str(device),

    # Data setup
    "data_augmentation": "full",
    "tip_size": 2,
    "explanation_type": "gradual williams",
    "explanation_lower_bound": 75,
    "explanation_upper_bound": 200,

    # Model
    "model": "AGT_UNet",
    "in_channels": 2,
    "out_channels": 1,
    "init_features": 64,
    "dropout_prob": 0.3,

    # Training
    "optimizer": "Adam",
    "learning_rate": 5e-4,
    "amsgrad": True,
    "prediction_loss": "DiceLoss",
    "explanation_loss": "CSILoss",
    "explanation_weight": 2.0,
    "pretrain_epochs": 30,
    "agt_epochs": 70,

    # Explanation setup
    "explanation_method": "GradCAM++",
    "explanation_layers": ["down1", "down2", "down3", "down4", "base"], # this indicates that all encoder layers are used
    "tip_selection": False,
    "heatmap_normalization": True,
}

# logging 
storage_folder = os.path.join("results", "tests", run_params["experiment_name"])
os.makedirs(os.path.join(storage_folder, "models"), exist_ok=True)
os.makedirs(os.path.join(storage_folder, "plots"), exist_ok=True)
os.makedirs(os.path.join(storage_folder, "data"), exist_ok=True)
json.dump(run_params, open(os.path.join(storage_folder, "metadata.json"), "w"), indent=2)

# load data
data_setup = DataSetup(transforms_name=run_params["data_augmentation"],
                       tip_size=run_params["tip_size"], 
                       explanation_type=run_params["explanation_type"], 
                       explanation_lower_bound=run_params["explanation_lower_bound"], 
                       explanation_upper_bound=run_params["explanation_upper_bound"])
dataloader_train, dataloader_val = data_setup.dataloaders["training"], data_setup.dataloaders["validation"]

# initialize model and criteria
model = AGT_UNet(
                 in_ch=run_params["in_channels"],
                 out_ch=run_params["out_channels"],
                 init_features=run_params["init_features"],
                 dropout_prob=run_params["dropout_prob"]
                ).to(device)
optimizer = optim.Adam(
                       model.parameters(), 
                       lr=run_params["learning_rate"],
                       amsgrad=run_params["amsgrad"]
                      )
criterion = DiceLoss()
criterion_expl = CSILoss()
explainer = GradCAMpp(
                      model, 
                      run_params["explanation_layers"], 
                      tip_selection=run_params["tip_selection"], 
                      scaling=run_params["heatmap_normalization"]
                     )


# training
epochs = [] # convenience for plotting
train_loss_predictions = []
train_loss_explanations = []
valid_loss_predictions = [] 
valid_loss_explanations = []
        
n_epochs_pretrain = run_params["pretrain_epochs"]
n_epochs_agt = run_params["agt_epochs"]
weight_factor = run_params["explanation_weight"]


for epoch in range(n_epochs_pretrain):
    loss_epoch, val_loss_epoch = [], []
    
    model.train()
    for input_data, target_data, target_explanations in dataloader_train:
        optimizer.zero_grad()
        input_data, target_data = input_data.to(device), target_data.to(device)

        predictions = model(input_data)
        
        loss = criterion(torch.sigmoid(predictions), target_data)
        loss_epoch.append(loss.item())
        
        loss.backward()
        optimizer.step()

    model.eval()
    for input_data, target_data, target_explanations in dataloader_val:
        input_data, target_data = input_data.to(device), target_data.to(device)

        predictions = model(input_data)

        val_loss = criterion(torch.sigmoid(predictions), target_data)
        
        val_loss_epoch.append(val_loss.item())

    epochs.append(epoch)
    train_loss_predictions.append(np.mean(loss_epoch))
    valid_loss_predictions.append(np.mean(val_loss_epoch))
    train_loss_explanations.append(0) # just for convenience when plotting. We do not use any explanation loss during pretraining
    valid_loss_explanations.append(0)

    padded_epoch = str(epoch).zfill(len(str(n_epochs_pretrain + n_epochs_agt)))
    print(f"Epoch {padded_epoch} done! Loss: {np.mean(loss_epoch)} - Val Loss: {np.mean(val_loss_epoch)}")
print("Pretraining done!")

# save model after pretraining
torch.save(model.state_dict(), f"{storage_folder}/models/model_pretrained.pt")

for epoch in range(n_epochs_agt):
    loss_epoch, val_loss_epoch = [], []
    loss_epoch_expl, val_loss_epoch_expl = [], []
   
    model.train()
    for input_data, target_data, target_explanations in dataloader_train:
        optimizer.zero_grad()
        
        input_data, target_data, target_explanations = input_data.to(device), target_data.to(device), target_explanations.to(device)

        heatmaps, predictions = explainer(input_data=input_data, agt_data=input_data)
        
        loss_prediction = criterion(torch.sigmoid(predictions), target_data)
        loss_explanation = criterion_expl(heatmaps, target_explanations)
        loss = loss_prediction + weight_factor * loss_explanation
        
        loss_epoch.append(loss_prediction.item())
        loss_epoch_expl.append(loss_explanation.item())

        loss.backward()
        optimizer.step()

    model.eval()
    for input_data, target_data, target_explanations in dataloader_val:
        input_data, target_data, target_explanations = input_data.to(device), target_data.to(device), target_explanations.to(device)
        
        heatmaps, predictions = explainer(input_data=input_data, agt_data=input_data)
        val_loss_prediciton = criterion(torch.sigmoid(predictions), target_data)
        val_loss_explanation = criterion_expl(heatmaps, target_explanations)

        val_loss_epoch.append(val_loss_prediciton.item())
        val_loss_epoch_expl.append(val_loss_explanation.item())

    epochs.append(epoch + n_epochs_pretrain)
    train_loss_predictions.append(np.mean(loss_epoch))
    train_loss_explanations.append(np.mean(loss_epoch_expl))
    valid_loss_predictions.append(np.mean(val_loss_epoch))
    valid_loss_explanations.append(np.mean(val_loss_epoch_expl))
    
    # save model depening on the best validation loss
    if epoch % 5 == 0 or np.mean(val_loss_epoch) <= min(train_loss_predictions[n_epochs_pretrain-1:]): # exclude pretraining epochs
        torch.save(model.state_dict(), f"{storage_folder}/models/model_agt_{epoch+n_epochs_pretrain}.pt")
    
    padded_epoch = str(epoch).zfill(len(str(n_epochs_pretrain + n_epochs_agt)))
    print(f"Epoch {padded_epoch} done! Training prediction loss: {np.mean(loss_epoch):.4f}, training explanation loss: {np.mean(loss_epoch_expl):.4f} - Validation prediction loss: {np.mean(val_loss_epoch):.4f}, validation explanation Loss: {np.mean(val_loss_epoch_expl):.4f}")


# save final model
torch.save(model.state_dict(), f"{storage_folder}/models/model_final.pt")

# save training history
np.save(f"{storage_folder}/data/training_epochs.npy", epochs)
np.save(f"{storage_folder}/data/training_loss_learner.npy", train_loss_predictions)
np.save(f"{storage_folder}/data/training_loss_critic.npy", train_loss_explanations)
np.save(f"{storage_folder}/data/validation_loss_learner.npy", valid_loss_predictions)
np.save(f"{storage_folder}/data/validation_loss_critic.npy", valid_loss_explanations)

# visualize the training
random_index = np.random.randint(0, len(dataloader_val.dataset))

pretrained_model = torch.load(f"{storage_folder}/models/model_pretrained.pt", map_location=device)
model.load_state_dict(pretrained_model)

validation_input, validation_target, validation_explanation = dataloader_val.dataset[random_index]
validation_input = validation_input.to(device).unsqueeze(0)
validation_target = validation_target.squeeze()
validation_explanation = validation_explanation.squeeze()

pretrained_heatmaps, pretrained_predictions = explainer(input_data=validation_input, agt_data=validation_input)
pretrained_segmentation = torch.sigmoid(pretrained_predictions).detach().cpu()
pretrained_segmentation = torch.where(pretrained_segmentation > 0.5, 1, 0).squeeze()
pretrained_heatmaps = pretrained_heatmaps.detach().cpu().squeeze()

agt_model = torch.load(f"{storage_folder}/models/model_final.pt", map_location=device)
model.load_state_dict(agt_model)

agt_heatmaps, agt_predictions = explainer(input_data=validation_input, agt_data=validation_input)
agt_segmentation = torch.sigmoid(agt_predictions).detach().cpu()
agt_segmentation = torch.where(agt_segmentation > 0.5, 1, 0).squeeze()
agt_heatmaps = agt_heatmaps.detach().cpu().squeeze()


plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 20
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["savefig.dpi"] = 300

segmentation_cmap = ListedColormap(["none", "lime"])
target_cmap = ListedColormap(["none", "black"])

fig = plt.figure(figsize=(20, 10))
grid = fig.add_gridspec(2, 4)

ax1 = fig.add_subplot(grid[0, 0])
ax1.imshow(np.zeros_like(pretrained_segmentation), cmap="gray", vmin=0, vmax=1)
ax1.imshow(pretrained_segmentation, cmap=segmentation_cmap, alpha=0.6)
ax1.imshow(validation_target, cmap=target_cmap, alpha=0.5)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_ylabel("Pretrained Model")
ax1.set_title("Segmentation")

handles = [plt.Line2D([0], [0], color="black", lw=4, label="Target"),
           plt.Line2D([0], [0], color="lime", lw=4, label="Segmentation")]
ax1.legend(handles=handles, loc="upper right", fontsize=12)


ax2 = fig.add_subplot(grid[0, 1])
ax2.imshow(pretrained_heatmaps, cmap="coolwarm", vmin=0, vmax=1)
ax2.axis("off")
ax2.set_title("Attention heatmap")


ax3 = fig.add_subplot(grid[1, 0])
ax3.imshow(np.zeros_like(agt_segmentation), cmap="gray", vmin=0, vmax=1)
ax3.imshow(agt_segmentation, cmap=segmentation_cmap, alpha=0.6)
ax3.imshow(validation_target, cmap=target_cmap, alpha=0.5)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_ylabel("AGT Model")

ax4 = fig.add_subplot(grid[1, 1])
ax4.imshow(agt_heatmaps, cmap="coolwarm", vmin=0, vmax=1)
ax4.axis("off")

ax5 = fig.add_subplot(grid[:, 2:])
ax5.plot(epochs, train_loss_predictions, label="Training Loss (prediction)", marker="^", color="tab:blue")
ax5.plot(epochs, valid_loss_predictions, label="Validation Loss (prediction)", marker="2", color="tab:orange")
ax5.plot(epochs, train_loss_explanations, label="Training Loss (explanation)", marker="o", color="tab:purple")
ax5.plot(epochs, valid_loss_explanations, label="Validation Loss (explanation)", marker="s", color="tab:red")

ax5.fill_betweenx([0, 1], 0, n_epochs_pretrain, color="tab:blue", alpha=0.1, label="Pretraining Phase")
ax5.fill_betweenx([0, 1], n_epochs_pretrain, n_epochs_pretrain + n_epochs_agt, color="tab:orange", alpha=0.1, label="AGT Phase")

ax5.set_xlim(0, n_epochs_pretrain + n_epochs_agt)
ax5.set_ylim(0, 1)
ax5.set_xlabel("Epoch")
ax5.set_ylabel("Loss")
ax5.legend(loc="lower left", fontsize=12)
ax5.set_title("Training and Validation Loss")

plt.tight_layout()
plt.savefig(os.path.join(storage_folder, "plots", "training_summary.png"))