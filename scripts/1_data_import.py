# general imports
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import torch

# src imports
sys.path.append(".") # assuming the script is run from the root directory via "python scripts/....py"
from src.data.datasetup import DataSetup


# initialize data setup class that orchestrates the data loading, preprocessing selection and adds target explanations
ds = DataSetup(
            data_folder="data/", # path to the data folder
            experiment_name="S_160_2.0", # name of the experiment, used to load the data. Valid options: "S_160_4.7", "S_160_2.0", "S_950_1.6"
            
            # Data augmentation and preprocessing parameters
            transforms_name="minimal", # valid options: "full", "minimal", "validation", and "no tips" (for datasets without targets)
            
            # Explanation parameters for AGT datasets
            explanation_type="gradual williams", # valid options: "binary", "gradient", "fake", "multiple fake"
            explanation_lower_bound=75, # lower bound for the explanation, used to clamp and scale the explanation values
            explanation_upper_bound=200, # upper bound for the explanation, used to clamp and scale the explanation values
            tip_size=2) # size of the radius around the single crack tip pixel to be considered a valid segmentation
    
# get the dataloaders for training and validation
dataloader_train, dataloader_val = ds.dataloaders["training"], ds.dataloaders["validation"]

# the data consists of input data, target data and target explanations
if ds.experiment_hasTargets:
    input_data, target_data, target_explanation = next(iter(dataloader_train))

    input_data_val, target_data_val, target_explanation_val = next(iter(dataloader_val))
else:
    input_data = next(iter(dataloader_train))
    target_data, target_explanation = torch.zeros_like(input_data)[:,0], torch.zeros_like(input_data)[:,0] # no targets or explanations available, so we create empty tensors

    input_data_val = next(iter(dataloader_val))
    target_data_val, target_explanation_val = torch.zeros_like(input_data_val)[:,0], torch.zeros_like(input_data_val)[:,0] 


# describe the data
total_data = len(dataloader_train.dataset)
print(f"Dataset: {ds.experiment_name}")
print(f"Total number of samples: {total_data}, batch size: {dataloader_train.batch_size}")
print(f"Input data shape: {input_data.shape} - min: {input_data.min():.4f} - max:{input_data.max():.4f} - mean: {input_data.mean():.4f} - std: {input_data.std():.4f}")
print(f"Target data shape: {target_data.shape} - min: {target_data.min():.4f} - max:{target_data.max():.4f} - mean: {target_data.mean():.4f} - std: {target_data.std():.4f}")
print(f"Target explanation shape: {target_explanation.shape} - min: {target_explanation.min():.4f} - max:{target_explanation.max():.4f} - mean: {target_explanation.mean():.4f} - std: {target_explanation.std():.4f}")

# full dataset information
inputs_full = torch.cat([x for x, _, _ in dataloader_train], dim=0) if ds.experiment_hasTargets else torch.cat([x for x in dataloader_train], dim=0)
print(f"Full input data shape: {inputs_full.shape} - min: {inputs_full.min():.4f} - max:{inputs_full.max():.4f} - mean: {inputs_full.mean():.4f} - std: {inputs_full.std():.4f}")


# visualize the data
random_index = np.random.randint(0, dataloader_val.batch_size)

plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 20
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["savefig.dpi"] = 300

fig, ax = plt.subplots(2, 4, figsize=(20, 10))

ax[0,0].imshow(input_data[random_index][0].squeeze(), cmap="coolwarm", vmin=-3, vmax=3)
ax[0,0].set_title("Input data - displacements $u_x$")
ax[0, 0].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
ax[0, 0].tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
for spine in ax[0, 0].spines.values():
    spine.set_visible(False)
ax[0,0].set_ylabel("Training side")

ax[0,1].imshow(input_data[random_index][1].squeeze(), cmap="coolwarm", vmin=-3, vmax=3)
ax[0,1].set_title("Input data - displacements $u_y$")
ax[0,1].axis("off")

ax[0,2].imshow(target_data[random_index].squeeze(), cmap="gray")
ax[0,2].set_title("Target data - crack tip" if ds.experiment_hasTargets else "No available target data")
ax[0,2].axis("off")

ax[0,3].imshow(target_explanation[random_index].squeeze(), cmap="coolwarm") 
ax[0,3].set_title("Target explanation - AGT explanation" if ds.experiment_hasTargets else "No available target explanation")
ax[0,3].axis("off")


ax[1,0].imshow(input_data_val[random_index][0].squeeze(), cmap="coolwarm", vmin=-3, vmax=3)
ax[1,0].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
ax[1,0].tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
for spine in ax[1, 0].spines.values():
    spine.set_visible(False)
ax[1,0].set_ylabel("Validation side")

ax[1,1].imshow(input_data_val[random_index][1].squeeze(), cmap="coolwarm", vmin=-3, vmax=3)
ax[1,1].axis("off")

ax[1,2].imshow(target_data_val[random_index].squeeze(), cmap="gray")
ax[1,2].axis("off")

ax[1,3].imshow(target_explanation_val[random_index].squeeze(), cmap="coolwarm")
ax[1,3].axis("off")


fig.tight_layout(pad=2.0)
os.makedirs("results/tests", exist_ok=True)
plt.savefig(f"results/tests/data_{ds.experiment_name}.png")
print(f"Data setup test completed. Plot saved to 'results/tests/data_{ds.experiment_name}.png'")