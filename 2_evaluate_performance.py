import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.data.datasetup import DataSetup
from src.deeplearning.model_architectures import AGT_UNet
from src.evaluation.validation_metrics import get_no_label_reliability
from src.deeplearning.loss_functions import DiceLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# benchmark datasets
dataset_names = [
    "S_160_4.7",
    "S_160_2.0",
    "S_950_1.6",
]

# model run folders
run_base = os.path.join("results", "runs")
runs = os.listdir(run_base)

for run in runs:
    print(f"Benchmarking {run}...")
    base_path = os.path.join(run_base, run)
    model_folder = os.path.join(base_path, "models")
    plot_folder = os.path.join(base_path, "plots")
    os.makedirs(plot_folder, exist_ok=True)

    metadata_path = os.path.join(base_path, "metadata.json")
    metadata = json.load(open(metadata_path))

    val_loss = np.load(os.path.join(base_path, "data", "validation_loss_prediction.npy"))
    min_val_loss_epoch = int(np.argmin(val_loss))


    model_files = [f for f in os.listdir(model_folder) if f.endswith(".pt")]
    selected_models = []
    for f in model_files:
        if "pretrained" in f or "final" in f or f"agt_{min_val_loss_epoch}" in f:
            selected_models.append(f) # only evaluate the important 3 model instances for brevity, but could evaluate the evolution of the model over all epochs if desired

    for model_file in selected_models:
        print(f"> Evaluating: {model_file}")
        model_path = os.path.join(model_folder, model_file)
        model = AGT_UNet(
            in_ch=metadata["in_channels"],
            out_ch=metadata["out_channels"],
            dropout_prob=metadata["dropout_prob"],
            init_features=metadata["init_features"]
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        criterion = DiceLoss()
        benchmark = {}

        for dataset_name in dataset_names:
            data_setup = DataSetup(experiment_name=dataset_name, transforms_name="validation")
            dataloader = data_setup.dataloaders["validation"]
            predictions = []
            loss_list = []

            if data_setup.experiment_hasTargets:
                for input_data, target_data, _ in dataloader:
                    input_data = input_data.to(device)
                    target_data = target_data.to(device)
                    with torch.no_grad():
                        pred = torch.sigmoid(model(input_data))
                    pred_bin = torch.where(pred > 0.5, 1, 0).cpu()
                    predictions.append(pred_bin)
                    loss = criterion(pred, target_data)
                    loss_list.append(loss.item())
                loss_mean = np.mean(loss_list)
            else:
                for input_data in dataloader:
                    input_data = input_data.to(device)
                    with torch.no_grad():
                        pred = torch.sigmoid(model(input_data))
                    pred_bin = torch.where(pred > 0.5, 1, 0).cpu()
                    predictions.append(pred_bin)
                loss_mean = None

            predictions = torch.cat(predictions, dim=0)
            rel_score, *_ = get_no_label_reliability(predictions)
            benchmark[dataset_name] = {
                "loss": loss_mean,
                "reliability": rel_score
            }

        if "final" in model_file:
            benchmark["epoch"] = metadata["agt_epochs"] + metadata["pretrain_epochs"]
        elif "pretrained" in model_file:
            benchmark["epoch"] = metadata["pretrain_epochs"]
        else:
            try:
                benchmark["epoch"] = int(model_file.split("_")[-1].split(".")[0])
            except ValueError:
                benchmark["epoch"] = None

        out_file = model_file.replace(".pt", "_benchmark.json")
        out_path = os.path.join(base_path, out_file)
        with open(out_path, "w") as f:
            json.dump(benchmark, f, indent=2)
        
        dataset_name = dataset_names[0]
        data_setup = DataSetup(experiment_name=dataset_name, transforms_name="validation")
        dataloader_val = data_setup.dataloaders["validation"]

        input_batch, target_batch, _ = next(iter(dataloader_val))
        random_index = np.random.randint(0, input_batch.shape[0])
        prediction = torch.sigmoid(model(input_batch.to(device).float())).cpu().detach()
        segmentation = torch.where(prediction[random_index][0] > 0.5, 1, 0).numpy()

        segmentation_cmap = ListedColormap(["none", "lime"])
        segmentation_alpha = np.where(segmentation > 0, 0.6, 0)

        target_cmap = ListedColormap(["none", "black"])
        target_alpha = np.where(target_batch[random_index][0] > 0, 1., 0)

        train_pred = np.load(os.path.join(base_path, "data", "training_loss_prediction.npy"))
        val_pred = np.load(os.path.join(base_path, "data", "validation_loss_prediction.npy"))
        train_exp = np.load(os.path.join(base_path, "data", "training_loss_explanation.npy"))
        val_exp = np.load(os.path.join(base_path, "data", "validation_loss_explanation.npy"))
        min_val_loss = np.min(val_pred)


        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.size"] = 20
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["savefig.dpi"] = 300

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        fig.tight_layout(pad=2.0)

        ax[0].plot(train_pred, label="Training Loss (prediction)", marker="^", color="tab:blue", linewidth=.4, markersize=1.)
        ax[0].plot(val_pred, label="Validation Loss (prediction)", marker="2", color="tab:orange", linewidth=.4, markersize=1.)
        ax[0].axvline(benchmark["epoch"], color="red", linestyle="--")
        ax[0].set_title("Predictive loss")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Dice loss [a.u.]")
        ax[0].set_aspect("auto")

        ax[1].plot(train_exp, label="Training Loss (explanation)", marker="o", color="tab:purple", linewidth=.4, markersize=1.)
        ax[1].plot(val_exp, label="Validation Loss (explanation)", marker="s", color="tab:red", linewidth=.4, markersize=1.)
        ax[1].axvline(benchmark["epoch"], color="red", linestyle="--", label="Evaluated epoch")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Cosine similarity loss [a.u.]")
        ax[1].legend(loc="upper right", fontsize=12)
        ax[1].set_title("Explanatory loss")
        ax[1].set_aspect("auto")

        ax[2].imshow(np.ones_like(segmentation), cmap="gray", vmin=0, vmax=1)
        ax[2].imshow(segmentation, cmap=segmentation_cmap, alpha=segmentation_alpha, zorder=10)
        ax[2].imshow(target_batch[random_index][0].squeeze(), cmap=target_cmap, alpha=target_alpha)
        ax[2].set_title("Segmentation")
        ax[2].legend(handles=[
            plt.Line2D([0], [0], color="black", lw=4, label="Target"),
            plt.Line2D([0], [0], color="lime", lw=4, label="Segmentation")
        ], loc="upper right", fontsize=12)

        ax[2].set_xticks([])
        ax[2].set_yticks([])
        bbox0 = ax[0].get_position()
        bbox1 = ax[1].get_position()
        ax[0].set_position([bbox0.x0, bbox0.y0, bbox1.width, bbox1.height])

        fig.tight_layout()
        plt.savefig(os.path.join(plot_folder, model_file.replace(".pt", "benchmark_summary.png")))
        plt.close()