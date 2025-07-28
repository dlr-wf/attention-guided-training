import os
import sys
import json
import torch
import numpy as np

sys.path.append(".")
from src.data.datasetup import DataSetup
from src.deeplearning.loss_functions import DiceLoss
from src.deeplearning.model_architectures import AGT_UNet
from src.evaluation.validation_metrics import get_no_label_reliability

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load to be benchmarked model
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

# set settings on which to benchmark the model (criterion + datasets)
criterion = DiceLoss() # -> used to determine predictive performance on datasets that have a defined target
dataset_names = [
    "S_160_4.7",
    "S_160_2.0",
    "S_950_1.6",
]
results = {}

for dataset_name in dataset_names:
    data_setup = DataSetup(
                           experiment_name=dataset_name,
                           transforms_name="validation"
                          )
    dataloader_train, dataloader_val = data_setup.dataloaders["training"], data_setup.dataloaders["validation"]

    loss_epoch = []
    predictions = []
    if data_setup.experiment_hasTargets:
        for input, target, explanation in dataloader_val:
            input_data = input.to(device)
            target_mask = target.to(device)
            
            pred = torch.sigmoid(model(input_data)).detach()
            loss = criterion(pred, target_mask)

            predictions.append(torch.where(pred > 0.5, 1, 0).cpu())
            loss_epoch.append(loss.item())
        loss_mean = np.mean(loss_epoch)
    else:
        for data in dataloader_val:
            input_data = data.to(device)
            
            pred = torch.sigmoid(model(input_data)).detach()
            
            predictions.append(torch.where(pred > 0.5, 1, 0).cpu())
        loss_mean = None # cannot be determined without targets
    
    # calculate reliability
    predictions = torch.cat(predictions, dim=0)
    rel_score, n_invalids, list_invalids, list_nonsegmented, list_multiple_segmented = get_no_label_reliability(predictions)

    # save results
    results[dataset_name] = {
        "loss": loss_mean,
        "reliability": rel_score,
    }

json.dump(results, open(os.path.join(base_path, "benchmark.json"), "w"), indent=2)
print(f"Saved benchmark results to {base_path}/benchmark.json")