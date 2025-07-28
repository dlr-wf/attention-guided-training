"""
This script manages the training of multiple AGT runs with different configurations.
"""
import torch

from src.utils.train_agt import train_agt  

# define the metadata configurations -> use e.g., itertools.product if you want to explore combinations of multiple parameters
explanation_types = ["binary williams", "gradual williams", "binary misleading", "multi-gradual misleading", "reference"]

# perform each run n times for statistical significance
n_repeats = 10
performed_runs = 0 # keep track for indexing
for n in range(n_repeats):
    for type in explanation_types:
        
        run_params = {
            "experiment_name": f"AGT_{performed_runs}", # simple indexing, metadata should be sufficient to identify the run
            "device": "cuda" if torch.cuda.is_available() else "cpu",

            # Data setup
            "data_augmentation": "full",
            "tip_size": 2,
            "explanation_type": type,
            "explanation_lower_bound": 75,
            "explanation_upper_bound": 200,

            # Model
            "model": "AGT_UNet", # just for documentation, does not affect the run
            "in_channels": 2,
            "out_channels": 1,
            "init_features": 64,
            "dropout_prob": 0.3,

            # Training
            "optimizer": "Adam", # just for documentation, does not affect the run
            "learning_rate": 5e-4,
            "amsgrad": True,
            "predictive_loss": "DiceLoss", # just for documentation, does not affect the run
            "explanatory_loss": "CSILoss", # just for documentation, does not affect the run
            "explanation_weight": 2. if type != "reference" else 0., # reference runs do not receive explanatory feedback
            "pretrain_epochs": 30,
            "agt_epochs": 370,
            
            # Explanation setup
            "explanation_layers": ["down1", "down2", "down3", "down4", "base"],
            "explanation_method": "GradCAM++",
            "tip_selection": False,
            "heatmap_normalization": True,
        }
        train_agt(run_params=run_params)
        performed_runs += 1