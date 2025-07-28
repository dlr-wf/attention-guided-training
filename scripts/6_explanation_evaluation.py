import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(".")
from src.data.datasetup import DataSetup
from src.deeplearning.loss_functions import DiceLoss
from src.deeplearning.model_architectures import AGT_UNet
from src.explainability.methods.GradCAMpp import GradCAMpp

from src.explainability.metrics.obfuscate import Obfuscate
from src.explainability.metrics.continuity import Continuity
from src.explainability.metrics.compactness import Compactness
from src.explainability.metrics.obfuscate import ObfuscationTester
from src.explainability.metrics.correctness import IncrementalDeletion
from src.explainability.metrics.completeness import IncrementalInsertion

plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 20
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["savefig.dpi"] = 300

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
data_setup = DataSetup(experiment_name="S_160_4.7",
                        transforms_name="validation")
dataloader_train, dataloader_val = data_setup.dataloaders["training"], data_setup.dataloaders["validation"]

# get the explainer class
explainer = GradCAMpp(model=model, layers=["down1", "down2", "down3", "down4", "base"]) # normalize to 0-1 or keep original values

# get data samples
input_data_full = torch.cat([x for x, _, _ in dataloader_val], dim=0) if data_setup.experiment_hasTargets else torch.cat([x for x in dataloader_val], dim=0)

# thin out the data for faster evaluation (technically using all samples is more statistically rigorous, but takes longer + requires more memory -> batching)
dataloader_val.dataset.inputs = dataloader_val.dataset.inputs[::10]
input_data_reduced = input_data_full[::10] 


# Step 1: Determine how to interpolate the explanations in downstream tasks
# determine interpolation scale
# performs multiple random deletions, determines the effect on the output and chooses the scales that controllably changes the output along the diagonal
obfuscation_calibration = ObfuscationTester(
                                              model=model,
                                              criterion=DiceLoss(),
                                              scales=[0.1, 0.5, 1., 1.5, 2.], # interpolation scales to be tested
                                              runs=3,  # how many runs to average over (more runs -> more statistically significant results)
                                              datapoints=30 # how many different % obfuscations to test (more datapoints -> more accurate results)
                                             )                                                 
best_scale, results = obfuscation_calibration(dataloader=dataloader_val)

percentages = np.linspace(0, 1, 30)
fig, ax = plt.subplots(figsize=(10, 6))
for scale, dev, mean, stds in results:
    ax.plot(percentages, 1 - mean, label=f"Scale {scale:.2f} (dev: {dev:.2f})")
    ax.fill_between(percentages, 1 - mean - stds, 1- mean + stds, alpha=0.2)

# note 0.85 is an emprical correction factor we include since we expect that the residual Dice score of a fully obfuscated sample is not 0, but rather around ~0.15 - 0.2
# This can also be seen in the Dice coefficient between the binarized prediction and the continuous version of that prediction for 0 obfuscation, since it likely does not reach 1, due to the non-zero residual probablity of non-segmented pixels
ax.plot(percentages, 1 - 0.85 * percentages, label="Approx. ideal curve", linestyle="--", color="black")

ax.set_title("Obfuscation calibration results")
ax.set_xlabel("Pixel obfuscated [\%]")
ax.set_ylabel("Dice coefficient [a.u.]")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks(np.linspace(0, 1, 5, endpoint=True))
ax.set_xticklabels([f"{int(x*100)}" for x in np.linspace(0, 1, 5, endpoint=True)])
ax.grid()
ax.legend(
    loc='upper left',
    bbox_to_anchor=(1.01, 1.0),
    borderaxespad=0.,
    frameon=True
)
fig.tight_layout()
plt.savefig(f"{base_path}/plots/obfuscation_calibration.png")
plt.close()


# Step 2: Evaluate the explanations for correctness (more commonly known as faithfulness)
# set up evaluation class for Co1 (correctness)
obfuscator = Obfuscate(scale=best_scale)
correctness = IncrementalDeletion(
                                  model=model,
                                  criterion=DiceLoss(),
                                  obfuscator=obfuscator,
                                  explainer=explainer
                                 )
(correctness_aucs_mean, correctness_aucs_std), (correctness_scores_perSample, correctness_percentages)  = correctness(input_data=input_data_reduced,
                               percentage_range=(0, 0.3, 30)) # percentage range to be tested (start, end, steps)

# save results
np.save(f"{base_path}/data/correctness_scores_perSample.npy", correctness_scores_perSample)
np.save(f"{base_path}/data/correctness_percentages.npy", correctness_percentages)

scores_mean = np.mean(correctness_scores_perSample, axis=0)
scores_std = np.std(correctness_scores_perSample, axis=0)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(correctness_percentages, scores_mean)
ax.fill_between(correctness_percentages, 
                scores_mean - scores_std, 
                scores_mean + scores_std, 
                alpha=0.2)
ax.set_title("Correctness evaluation results")
ax.set_xlabel("Pixels obfuscated [\%]")
ax.set_ylabel("Dice coefficient [a.u.]")
ax.set_xlim(0, 0.3)
ax.set_ylim(0, 1)
ax.set_xticks(np.linspace(0, 0.3, 3, endpoint=True))
ax.set_xticklabels([f"{int(x*100)}" for x in np.linspace(0, 0.3, 3, endpoint=True)])
ax.grid()
fig.tight_layout()
plt.savefig(f"{base_path}/plots/correctness_evaluation.png")
plt.close()

# Step 3: Evaluate the explanations for completeness
# set up evaluation class for Co2 (completeness)
completeness = IncrementalInsertion(
                                    model=model,
                                    criterion=DiceLoss(),
                                    obfuscator=obfuscator,
                                    explainer=explainer
                                   )
(completeness_aucs_mean, completeness_aucs_std), (completeness_scores_perSample, completeness_percentages) = completeness(input_data=input_data_reduced,
                                 percentage_range=(0, 0.3, 30)) # percentage range to be tested (start, end, steps)

# save results
np.save(f"{base_path}/data/completeness_scores_perSample.npy", completeness_scores_perSample)
np.save(f"{base_path}/data/completeness_percentages.npy", completeness_percentages)

fig, ax = plt.subplots(figsize=(10, 6))
scores_mean = np.mean(completeness_scores_perSample, axis=0)
scores_std = np.std(completeness_scores_perSample, axis=0)

ax.plot(completeness_percentages, scores_mean, label="Completeness Scores")
ax.fill_between(completeness_percentages, 
                scores_mean - scores_std, 
                scores_mean + scores_std, 
                alpha=0.2, label="Std Deviation")
ax.set_title("Completeness evaluation Results")
ax.set_ylabel("Dice coefficient [a.u.]")
ax.set_xlabel("Pixels un-obfuscated [\%]")
ax.set_xlim(0, 0.3)
ax.set_ylim(0, 1)
ax.set_xticks(np.linspace(0, 0.3, 3, endpoint=True))
ax.set_xticklabels([f"{int(x*100)}" for x in np.linspace(0, 0.3, 3, endpoint=True)])
ax.grid()
fig.tight_layout()
plt.savefig(f"{base_path}/plots/completeness_evaluation.png")
plt.close()


# Step 4: Evaluate the explanations for continuity
# set up evaluation class for Co4 (continuity)
continuity = Continuity(model=model,
                        explainer=explainer)
continuity_scores = continuity(input_data_full) # call with full dataset to ensure that we have an ordered set of contiguous samples
continuity_mean = np.mean(continuity_scores)
continuity_std = np.std(continuity_scores)

# save results
np.save(f"{base_path}/data/continuity_scores.npy", continuity_scores)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(continuity_scores, label="Continuity Scores")
ax.axhline(y=continuity_mean, color="r", linestyle="--", label="Mean Score")
ax.fill_between(range(len(continuity_scores)), 
                continuity_mean - continuity_std, 
                continuity_mean + continuity_std, 
                color="r", alpha=0.2, label="Std Deviation")
ax.set_title("Continuity Evaluation Results")
ax.set_xlabel("Sample Index [a.u.]")
ax.set_ylabel("Pairwise structural similarity [a.u.]")
ax.legend()
ax.set_ylim(0, 1)
ax.grid()
fig.tight_layout()
plt.savefig(f"{base_path}/plots/continuity_evaluation.png")
plt.close()



# Step 5: Evaluate the explanations for compactness
# set up evaluation class for Co7 (compactness)
compactness = Compactness(
                          model=model,
                          criterion=DiceLoss(),
                          obfuscator=obfuscator,
                          explainer=explainer
                         )
compactness_values, (compactness_scores, compactness_percentages) = compactness(input_data=input_data_reduced) # values contain percentages where threshold was reached where the scores correspond to the incremental insertion results
compactness_mean = np.mean(compactness_values)
compactness_std = np.std(compactness_values)

scores_mean = np.mean(compactness_scores, axis=0)
scores_std = np.std(compactness_scores, axis=0)

# save results
np.save(f"{base_path}/data/compactness_scores.npy", compactness_scores)
np.save(f"{base_path}/data/compactness_percentages.npy", compactness_percentages)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(compactness_percentages, scores_mean, label="Compactness Scores")
ax.fill_between(compactness_percentages, 
                scores_mean - scores_std, 
                scores_mean + scores_std, 
                alpha=0.2)
ax.axvline(x=compactness_mean, color="r", linestyle="--", label="Mean compactness")
ax.axhline(y=0.8, color="black", linestyle="--", label="Threshold (80\%)")
ax.fill_betweenx([0, 1],
                 compactness_mean - compactness_std, 
                 compactness_mean + compactness_std, 
                 color="r", alpha=0.2, label="Std Deviation")
ax.set_title("Compactness Evaluation Results")
ax.set_xlabel("Un-obfuscation Percentage [a.u.]")
ax.set_ylabel("Earliest threshold percentage [a.u.]")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks(np.linspace(0, 1, 5, endpoint=True))
ax.set_xticklabels([f"{int(x*100)}" for x in np.linspace(0, 1, 5, endpoint=True)])
ax.grid()
ax.legend()
plt.savefig(f"{base_path}/plots/compactness_evaluation.png")
plt.close()


# summarize results
results = {"Obfuscation cale": best_scale,
            "correctness": (correctness_aucs_mean, correctness_aucs_std),
            "completeness": (completeness_aucs_mean, completeness_aucs_std),
            "continuity": (continuity_mean, continuity_std),
            "compactness": (compactness_mean, compactness_std)}
json.dump(results, open(f"{base_path}/explanation_evaluation.json", "w"), indent=2)

print(f"Evaluation finished! Results saved to {base_path}/explanation_evaluation.json") 