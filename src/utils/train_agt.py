import os
import json
import torch
import numpy as np
import shutil
from torch import optim

from src.data.datasetup import DataSetup
from src.deeplearning.model_architectures import AGT_UNet
from src.explainability.methods.GradCAMpp import GradCAMpp
from src.deeplearning.loss_functions import DiceLoss, CSILoss

def train_agt(run_params):
    """
    Wrapper function to use AGT training with specified parameters.
    Results are stored in a hirachical folder structure based on the experiment name for consecutive analysis.
    
    Args:
        run_params (dict): Dictionary containing all parameters for the training run.
        Required keys include:
            - experiment_name (str): Name of the experiment.
            - device (str): Device to use for training ('cuda' or 'cpu').
            - data_augmentation (str): Type of data augmentation to apply.
            - tip_size (int): Size of the tips for the AGT model.
            - explanation_type (str): Type of explanation to use.
            - explanation_lower_bound (int): Lower bound for explanations.
            - explanation_upper_bound (int): Upper bound for explanations.
            - model (str): Model architecture to use.
            - in_channels (int): Number of input channels.
            - out_channels (int): Number of output channels.
            - init_features (int): Initial number of features in the model.
            - dropout_prob (float): Dropout probability for the model.
            - optimizer (str): Optimizer to use ('Adam', etc.).
            - learning_rate (float): Learning rate for the optimizer.
            - amsgrad (bool): Whether to use AMSGrad in the optimizer.
            - explanation_weight (float): Weight for the explanation loss.
            - n_epochs_pretrain (int): Number of epochs for pretraining.
            - n_epochs_agt (int): Number of epochs for AGT training.
            - explanation_layers (list): Layers to apply explanations on.
            - explanation_method (str): Method for generating explanations ('GradCAM', 'GradCAM++').
            - tip_selection (bool): Whether to select tips during explanation generation.
            - heatmap_normalization (bool): Whether to normalize heatmaps.
    
    Returns:
        None
    """
    # start from clean slate
    base_path = os.path.join("results", "runs", run_params["experiment_name"])
    shutil.rmtree(base_path) if os.path.exists(base_path) else None
    
    # create expected folder structure
    os.makedirs(os.path.join(base_path,"data"), exist_ok=True)
    os.makedirs(os.path.join(base_path,"models"), exist_ok=True)
    os.makedirs(os.path.join(base_path,"plots"), exist_ok=True)
    
    # store metadata
    json.dump(run_params, open(os.path.join(base_path,"metadata.json"), "w"), indent=2)
    
    if run_params["explanation_type"] == "reference":
        run_params["explanation_type"] = "gradual williams" # default to gradual williams for reference runs since we do not provide explanatory feedback anyway

    # prepare training environment
    device = torch.device(run_params["device"])
    ds = DataSetup(
                   transforms_name=run_params["data_augmentation"],
                   tip_size=run_params["tip_size"], 
                   explanation_type=run_params["explanation_type"],
                   explanation_lower_bound=run_params["explanation_lower_bound"], 
                   explanation_upper_bound=run_params["explanation_upper_bound"]
                  )
    dataloader_train, dataloader_val = ds.dataloaders["training"], ds.dataloaders["validation"]
    assert ds.experiment_hasTargets, "The chosen dataset does not contain targets for training. Please check the dataset setup."

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
    
    loss_function_prediction = DiceLoss()
    loss_function_explanation = CSILoss()


    # Training loop -> Compare to scripts/3_attention_guided_training.py
    epochs = [] # convenience for plotting
    train_loss_predictions = []
    train_loss_explanations = []
    valid_loss_predictions = [] 
    valid_loss_explanations = []
            
    explanation_weight = run_params["explanation_weight"]
    
    # pretrain the model to allow for salient explanation basis once introducing the explanatory feedback
    print(f"Pretraining for {run_params["pretrain_epochs"]} epochs...")
    for epoch in range(run_params["pretrain_epochs"]):
        loss_epoch, val_loss_epoch = [], []
        
        model.train()
        for input_data, target_data, target_explanations in dataloader_train:
            optimizer.zero_grad()
            
            input_data, target_data = input_data.to(device), target_data.to(device)

            prediction = model(input_data)
            loss = loss_function_prediction(torch.sigmoid(prediction), target_data)
            
            loss.backward()
            optimizer.step()
            
            loss_epoch.append(loss.item())

        model.eval()
        for input_data, target_data, target_explanations in dataloader_val:
            input_data, target_data = input_data.to(device), target_data.to(device)

            predictions = model(input_data)

            val_loss = loss_function_prediction(torch.sigmoid(predictions), target_data)
            
            val_loss_epoch.append(val_loss.item())

        epochs.append(epoch)
        train_loss_predictions.append(np.mean(loss_epoch))
        valid_loss_predictions.append(np.mean(val_loss_epoch))
        train_loss_explanations.append(0)
        valid_loss_explanations.append(0)

        padded_epoch = str(epoch).zfill(len(str(run_params["pretrain_epochs"] + run_params["agt_epochs"])))
        print(f"Epoch {padded_epoch} done! Training loss: {train_loss_predictions[-1]:.4f} - Validation loss: {valid_loss_predictions[-1]:.4f}")

    # save model after pretraining
    torch.save(model.state_dict(), os.path.join(base_path, "models", f"model_pretrained.pt"))
    print("Pretraining done!")
    
    # AGT training
    print(f"Running AGT for {run_params['agt_epochs']} epochs...")
    for epoch in range(run_params["agt_epochs"]):
        loss_epoch, val_loss_epoch = [], []
        loss_epoch_expl, val_loss_epoch_expl = [], []
    
        model.train()
        for input_data, target_data, target_explanations in dataloader_train:
            optimizer.zero_grad()
            
            input_data, target_data, target_explanations = input_data.to(device), target_data.to(device), target_explanations.to(device)
        
            if run_params["explanation_method"].lower() == "gradcam++":
                explainer = GradCAMpp(model, 
                                    run_params["explanation_layers"], 
                                    tip_selection=run_params["tip_selection"], 
                                    scaling=run_params["heatmap_normalization"])
            else:
                raise ValueError(f"Currently we only import GradCAM++ for simplicity, if you want to use {run_params['explanation_method']} please check if it is available in src/explainability/methods/ and import from there accordingly.")
            
            
            heatmaps, predictions = explainer(input_data=input_data, agt_data=input_data)
    
            loss_prediction = loss_function_prediction(torch.sigmoid(predictions), target_data)
            loss_explanations = loss_function_explanation(heatmaps, target_explanations)

            if explanation_weight > 0:
                loss = loss_prediction + explanation_weight * loss_explanations
            else:
                loss = loss_prediction

            loss.backward()
            optimizer.step()

            loss_epoch.append(loss_prediction.item())
            loss_epoch_expl.append(loss_explanations.item())
        
        model.eval()
        for input_data, target_data, target_explanations in dataloader_val:
            input_data, target_data, target_explanations = input_data.to(device), target_data.to(device), target_explanations.to(device)
            
            if run_params["explanation_method"].lower() == "gradcam++":
                explainer = GradCAMpp(model, 
                                    run_params["explanation_layers"], 
                                    tip_selection=run_params["tip_selection"], 
                                    scaling=run_params["heatmap_normalization"])
            else:
                raise ValueError(f"Currently we only import GradCAM++ for simplicity, if you want to use {run_params['explanation_method']} please check if it is available in src/explainability/methods/ and import from there accordingly.")    
            
            heatmaps, predictions = explainer(input_data=input_data, agt_data=input_data)
            
            val_loss = loss_function_prediction(torch.sigmoid(predictions), target_data)
            val_loss_expl = loss_function_explanation(heatmaps, target_explanations)

            val_loss_epoch.append(val_loss.item())
            val_loss_epoch_expl.append(val_loss_expl.item())

        epochs.append(epoch + run_params["pretrain_epochs"])
        train_loss_predictions.append(np.mean(loss_epoch))
        train_loss_explanations.append(np.mean(loss_epoch_expl))
        valid_loss_predictions.append(np.mean(val_loss_epoch))
        valid_loss_explanations.append(np.mean(val_loss_epoch_expl))

        # save model depening on the best validation loss and every 20 epochs regardless
        if (epoch % 20 == 0) or (valid_loss_predictions[-1] <= min(valid_loss_predictions[run_params["pretrain_epochs"]-1:])):
            torch.save(model.state_dict(), f"{base_path}/models/model_agt_{epochs[-1]}.pt")
        
        padded_epoch = str(epochs[-1]).zfill(len(str(run_params["agt_epochs"] + run_params["pretrain_epochs"])))
        print(f"Epoch {padded_epoch} done! Training prediction loss: {train_loss_predictions[-1]:.4f}, training explanation loss: {train_loss_explanations[-1]:.4f} - Validation prediction loss: {valid_loss_predictions[-1]:.4f}, validation explanation Loss: {valid_loss_explanations[-1]:.4f}")

    print("AGT training done!")
    
    
    # save final model
    torch.save(model.state_dict(), f"{base_path}/models/model_final.pt")

    # save training history
    np.save(f"{base_path}/data/training_epochs.npy", epochs)
    np.save(f"{base_path}/data/training_loss_prediction.npy", train_loss_predictions)
    np.save(f"{base_path}/data/training_loss_explanation.npy", train_loss_explanations)
    np.save(f"{base_path}/data/validation_loss_prediction.npy", valid_loss_predictions)
    np.save(f"{base_path}/data/validation_loss_explanation.npy", valid_loss_explanations)