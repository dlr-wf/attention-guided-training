import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.data.datasetup import DataSetup
from src.deeplearning.loss_functions import DiceLoss
from src.deeplearning.model_architectures import AGT_UNet
from src.explainability.methods.GradCAMpp import GradCAMpp
from src.explainability.metrics.correctness import IncrementalDeletion
from src.explainability.metrics.obfuscate import Obfuscate, ObfuscationTester


plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 20
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["savefig.dpi"] = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run_base = os.path.join("results", "runs")
runs = os.listdir(run_base)
for run in runs:
    base_path = os.path.join(run_base, run)
    print(f"Processing {run} ...")

    metadata_path = os.path.join(base_path, "metadata.json")
    metadata = json.load(open(metadata_path))
    in_ch, out_ch = metadata["in_channels"], metadata["out_channels"]
    init_features = metadata["init_features"]
    dropout_prob = metadata["dropout_prob"]

    # post-hoc "early stopping" so to say
    val_loss = np.load(os.path.join(base_path, "data", "validation_loss_prediction.npy"))
    min_val_epoch = int(np.argmin(val_loss))

    model_dir = os.path.join(base_path, "models")
    model_files = os.listdir(model_dir)
    selected_models = [f for f in model_files if ("pretrained" in f or "final" in f or f"agt_{min_val_epoch}" in f)] # we only evaluate the important 3 model instances for brevity, but could evaluate the evolution of the model over all epochs if desired

    for model_file in selected_models:
        print(f"> Evaluating {model_file}...")
        model = AGT_UNet(in_ch=in_ch, out_ch=out_ch, dropout_prob=dropout_prob, init_features=init_features)
        model.load_state_dict(torch.load(os.path.join(model_dir, model_file), map_location=device))
        model.to(device).eval()

        data_setup = DataSetup(experiment_name="S_160_4.7", transforms_name="validation")
        dataloader_val = data_setup.dataloaders["validation"]
        full_input_data = torch.cat([x for x, _, _ in dataloader_val], dim=0)
        reduced_input_data = full_input_data[::10]

        explainer = GradCAMpp(
            model=model,
            layers=metadata["explanation_layers"],
            tip_selection=metadata["tip_selection"],
            scaling=metadata["heatmap_normalization"]
        )
        
        obfuscation_calibration = ObfuscationTester(model, DiceLoss(), scales=np.linspace(0.1, 2., 5), runs=3, datapoints=30) # increasing the number of sampled scales may enable finding a more ideal scale, runs increases statistical significance and datapoints the resolution. Theese numbers are on the more coarse side to keep this analysis brief
        best_scale, interp_results = obfuscation_calibration(dataloader_val)
        interpolator = Obfuscate(best_scale)

        # Plot interpolation
        percentages = np.linspace(0, 1, 30)
        x_ticks = [0, 0.25, 0.5, 0.75, 1]
        x_labels = [0, 25, 50, 75, 100]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for scale, dev, mean, std in interp_results:
            ax.plot(percentages, 1 - mean, label=f"Scale {scale:.2f} (dev: {dev:.2f})")
            ax.fill_between(percentages, 1 - mean - std, 1 - mean + std, alpha=0.2)
        ax.plot(percentages, 1 - 0.85 * percentages, linestyle="--", color="black", label="Ideal")
        
        ax.set_ylim(0,1)
        ax.set_xlim(0,1)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_title("Obfuscation scale calibration")
        ax.set_xlabel("Pixels obfuscated [\%]")
        ax.set_ylabel("Dice coefficient [a.u.]")
        ax.grid()
        ax.legend(
            loc='upper left',
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.,
            frameon=True
        )
        fig.tight_layout()
        plt.savefig(f"{base_path}/plots/obfuscation_calibration_{model_file.replace('.pt', '')}.png")
        plt.close()

        # Correctness (Co1)
        correctness = IncrementalDeletion(model, DiceLoss(), interpolator, explainer)
        (corr_mean, corr_std), (corr_scores, corr_pcts) = correctness(reduced_input_data, percentage_range=(0, 0.3, 30))
        np.save(f"{base_path}/data/{model_file}_correctness_scores.npy", corr_scores)

        # plot correctness
        scores_mean = np.mean(corr_scores, axis=0)
        scores_std = np.std(corr_scores, axis=0)
        x_ticks = [0, 0.1, 0.2, 0.3]
        x_labels = [0, 10, 20, 30]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(corr_pcts, scores_mean, marker="o", linewidth=0.3)
        ax.fill_between(corr_pcts, scores_mean - scores_std, scores_mean + scores_std, alpha=0.2)
        
        ax.set_ylim(0,1)
        ax.set_xlim(0,0.3)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_title("Correctness (Incremental deletion)")
        ax.set_xlabel("Pixels obfuscated [\%]")
        ax.set_ylabel("Dice coefficient [a.u.]")
        ax.grid()
        fig.tight_layout()
        plt.savefig(f"{base_path}/plots/correctness_{model_file.replace('.pt', '')}.png")
        plt.close()

        # Summary plot
        rand_idx = np.random.randint(0, len(dataloader_val.dataset))
        val_input, val_target, _ = dataloader_val.dataset[rand_idx]
        val_input = val_input.unsqueeze(0).to(device)

        heatmap, pred = explainer(input_data=val_input, agt_data=val_input)
        seg = torch.sigmoid(pred).detach().cpu().squeeze()
        seg_bin = torch.where(seg > 0.5, 1, 0)
        heatmap = heatmap.detach().cpu().squeeze()
        val_target = val_target.squeeze().cpu()

        segmentation_cmap = ListedColormap(["none", "lime"])
        target_cmap = ListedColormap(["none", "black"])

        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(np.ones_like(seg_bin), cmap="gray", vmin=0, vmax=1)
        ax1.imshow(seg_bin, cmap=segmentation_cmap, alpha=0.8)
        ax1.imshow(val_target, cmap=target_cmap, alpha=0.8)
        handles = [plt.Line2D([0], [0], color="black", lw=4, label="Target"),
                   plt.Line2D([0], [0], color="lime", lw=4, label="Segmentation")]
        ax1.legend(handles=handles, loc="upper right", fontsize=12)
        
        ax1.set_title("Segmentation")
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(heatmap, cmap="coolwarm", vmin=0, vmax=1)        
        ax2.imshow(seg_bin, cmap=segmentation_cmap, alpha=0.8)
        ax2.imshow(val_target, cmap=target_cmap, alpha=0.8)
        handles = [plt.Line2D([0], [0], color="black", lw=4, label="Target"),
                   plt.Line2D([0], [0], color="lime", lw=4, label="Segmentation")]
        ax2.legend(handles=handles, loc="upper right", fontsize=12)

        ax2.set_title("Explanation")
        ax2.axis("off")

        epochs = np.load(f"{base_path}/data/training_epochs.npy")
        loss_pred_train = np.load(f"{base_path}/data/training_loss_prediction.npy")
        loss_pred_val = np.load(f"{base_path}/data/validation_loss_prediction.npy")
        loss_exp_train = np.load(f"{base_path}/data/training_loss_explanation.npy")
        loss_exp_val = np.load(f"{base_path}/data/validation_loss_explanation.npy")

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.plot(epochs, loss_pred_train, label="Training Loss (prediction)", marker="^", color="tab:blue", linewidth=.4, markersize=1.)
        ax3.plot(epochs, loss_pred_val, label="Validation Loss (prediction)", marker="2", color="tab:orange", linewidth=.4, markersize=1.)
        ax3.plot(epochs, loss_exp_train, label="Training Loss (explanation)", marker="o", color="tab:purple", linewidth=.4, markersize=1.)
        ax3.plot(epochs, loss_exp_val, label="Validation Loss (explanation)", marker="s", color="tab:red", linewidth=.4, markersize=1.)
        
        if metadata["explanation_weight"] > 0:
            ax3.fill_betweenx([0, 1], 0, metadata["pretrain_epochs"], color="tab:blue", alpha=0.1, label="Pretraining hase")
            ax3.fill_betweenx([0, 1], metadata["pretrain_epochs"], metadata["pretrain_epochs"] + metadata["agt_epochs"], color="tab:orange", alpha=0.1, label="AGT phase")
        else:
            ax3.fill_betweenx([0, 1], 0, metadata["agt_epochs"], color="tab:blue", alpha=0.1, label="Conventional Training phase")
        
        ax3.set_ylim(0,1)
        ax3.set_xlim(0, metadata["pretrain_epochs"] + metadata["agt_epochs"])
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss [a.u.]")
        ax3.legend(fontsize=8)
        ax3.set_title("Loss curves")
        
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, "plots", f"{model_file.replace('.pt', '')}_explanation_summary.png"))
        plt.close()

        # write to json for later use
        results = {
            "Obfuscation scale": best_scale,
            "correctness": (corr_mean, corr_std),
        }
        with open(os.path.join(base_path, f"{model_file.replace('.pt', '')}_explanation_evaluation.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"> Results saved to {model_file.replace('.pt', '')}_explanation_evaluation.json !")