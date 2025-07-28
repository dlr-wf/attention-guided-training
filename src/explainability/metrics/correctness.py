import torch
import numpy as np
from typing import Tuple

from src.explainability.metrics.baseclass import Co12Base


class IncrementalDeletion(Co12Base):
    def __init__(self, 
                model: torch.nn.Module, 
                criterion: callable, 
                obfuscator: callable, 
                explainer: callable
                ):
        """
        Implementation of the Incremental Deletion metric.
        
        Args:
            model (torch.nn.Module): The model to be tested.
            criterion (callable): The loss function to evaluate the model's predictions.
            obfuscator (callable): The obfuscation method to apply to the input data.
            explainer (callable): The explainer method to compute heatmaps.
        """
        self.model = model
        self.criterion = criterion
        self.device = next(model.parameters()).device
        self.obfuscator = obfuscator
        self.explainer = explainer

    def __call__(self, 
                 input_data: torch.Tensor, 
                 percentage_range: Tuple[float, float, int] = (0, 0.03, 30), 
                 ) -> Tuple:
        """
        Obfuscate the input data incrementally based on the attention heatmap and compute the model's performance.
        A steep decrease in performance indicates that the model relies on the obfuscated features, which in turn indicates that the heatmap is faithful.
        
        Args:
            input_data (torch.Tensor): Input data to be obfuscated.
            percentage_range (Tuple[float, float, int]): Range of percentages for obfuscation.
        
        Returns:
            Tuple: A tuple containing the mean and standard deviation of the AUC scores, and the scores and percentages.
        
        """
        self.model.eval()
        
        batch_size = input_data.size(0)
        scores = []


        for i in range(batch_size):
            current_input = input_data[i].unsqueeze(0).to(self.device)
            
            heatmap, target_data = self._compute_heatmap(current_input)
            
            # skip inputs where the segmentation failed, as the explanations are likely to be wrong/not representative and should be excluded from the score calculation
            if heatmap is None:
                continue
            
            # generate interpolated inputs 
            new_inputs, percentages = self._generate_interpolated_inputs(current_input, heatmap, percentage_range)
            batch_scores = []
            
            with torch.no_grad():
                for j in range(new_inputs.size(0)):
                    pred = torch.sigmoid(self.model(new_inputs[j].unsqueeze(0).to(self.device))).detach()
                    score = self._compute_dice_score(pred, target_data)
                    batch_scores.append(score)
            
            scores.append(batch_scores)


        scores = np.array(scores)

        auc_mean, auc_std = self._calculate_auc_with_uncertainty(percentages, scores)

        return (auc_mean, auc_std), (scores, percentages)


    def _compute_heatmap(self, input_data):
        pred = self.model(input_data)
        
        # set target data as the prediction of the unperturbed input as reference for the decrease in performance after deletion
        if pred.size(1) == 1:
            pred = torch.sigmoid(pred).detach()
            target_data = torch.where(pred > 0.5, 1, 0) # simulate the ground truth data (binary mask)
        else:
            pred = torch.softmax(pred, dim=1)
            target_data = torch.argmax(pred, dim=1) # simulate the ground truth data (binary mask)
            target_data = torch.eye(pred.size(1))[target_data].to(self.device) # one-hot encode the target data for multi-class

        if target_data.sum() == 0: # if no crack tip was segmented, skip the input
            return None, None
        
        # compute heatmap using the provided explainer class -> has to be a callable class
        heatmap = self.explainer(input_data)
        return heatmap, target_data



    def _generate_interpolated_inputs(self, input_data, heatmap, percentage_range):
        percentages = np.linspace(percentage_range[0], percentage_range[1], percentage_range[2])
        
        # calculate thresholds for the heatmap with the threshold seperating the top 1-percentile% of the heatmap pixels
        thresholds = torch.quantile(heatmap, 1 - torch.from_numpy(percentages).to(self.device).to(torch.float32))
        thresholds = thresholds.view(-1, 1, 1, 1) # add dimensions for broadcasting
        
        masks = (heatmap > thresholds).to(self.device)
        interpolated_input = self.obfuscator(input_data)
        
        new_inputs = torch.where(masks, interpolated_input, input_data)

        return new_inputs, percentages