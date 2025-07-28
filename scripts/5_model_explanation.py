import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(".")
from src.data.datasetup import DataSetup
from src.deeplearning.model_architectures import AGT_UNet
from src.explainability.methods.GradCAMpp import GradCAMpp

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load to be explained model
model = AGT_UNet(in_ch=2, out_ch=1, dropout_prob=0.)

# preset for example AGT run
base_path = os.path.join("results", "tests", "attention_guided_training")
model_path = os.path.join(base_path, "models", "model_final.pt")

# preset for example conventional run
# base_path = os.path.join("results", "tests", "conventional_training")
# model_path = os.path.join(base_path, "models", "model_final.pt")

state = torch.load(model_path)
model.load_state_dict(state)
model.to(device)
model.eval()


# load data
ds = DataSetup(experiment_name="S_160_4.7",
                        transforms_name="validation")

dataloader_train, dataloader_val = ds.dataloaders["training"], ds.dataloaders["validation"]

# get an example image closer to the center of the validation set for a more beautiful explanation. Samples close to the edge of an image are often difficult to predict and have worse signal-to-noise ratio.
input_data = dataloader_val.dataset[len(dataloader_val.dataset) // 2]
input_data = input_data[0].unsqueeze(0).to(device)  if ds.experiment_hasTargets else input_data.unsqueeze(0).to(device)

explainer = GradCAMpp(model=model, layers=["down1", "down2", "down3", "down4", "base"])
explanation = explainer(input_data).detach().cpu().numpy()

# visualize
plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 20
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["savefig.dpi"] = 300

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

hm = ax.imshow(explanation.squeeze(), cmap="coolwarm")
cbar = fig.colorbar(hm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Attention [a.u.]", rotation=90, labelpad=20)

ax.set_title("Explanation")
ax.axis("off")

fig.tight_layout()
plt.savefig(f"{base_path}/plots/example_explanation.png")