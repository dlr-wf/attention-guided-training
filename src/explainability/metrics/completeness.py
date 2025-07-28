import torch
import numpy as np
from typing import Tuple, Dict

from src.explainability.metrics.baseclass import Co12Base

class IncrementalInsertion(Co12Base):
    def __init__(self, 
                model: torch, 
                criterion: callable, 
                obfuscator: callable, 
                explainer: callable):
        """
        Implementation of the Incremental Insertion metric.
        
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
                 ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate the completeness metric by incrementally inserting features based on the attention heatmap and computing the model's performance.
        If the model's performance increases significantly and reaches almost full predictive performance, it indicates that the model relies almost exclusively the inserted features, which in turn indicates that the heatmap contains an (almost) complete set of relevant features.
        
        Args:
            input_data (torch.Tensor): Input data to be evaluated.
            percentage_range (Tuple[float, float, int]): Range of percentages for insertion.
            
        Returns:
            Dict[str, Tuple[float, float]]: A dictionary containing the mean and standard deviation of the AUC scores, and the scores and percentages.

        """
        self.model.eval()
        scores = []

        for sample in input_data:
            current_input = sample.unsqueeze(0).to(self.device)
            
            heatmap, target_data = self._compute_heatmap(current_input)
            
            # skip inputs where the segmentation failed, as the explanations are likely to be wrong/not representative and should be excluded from the score calculation
            if heatmap is None:
                continue
            
            # generate interpolated inputs 
            new_inputs, percentages = self._generate_interpolated_inputs(current_input, heatmap, percentage_range)
            batch_scores = []
            
            with torch.no_grad():
                for j in range(new_inputs.size(0)):
                    pred = torch.sigmoid(self.model(new_inputs[j].unsqueeze(0))).detach()
                    score = self._compute_dice_score(pred, target_data)
                    batch_scores.append(score)
            
            scores.append(batch_scores)

        scores = np.array(scores)
        auc_mean, auc_std = self._calculate_auc_with_uncertainty(percentages, scores)
        return (auc_mean, auc_std), (scores, percentages)
    

    def _generate_interpolated_inputs(self, input_data, heatmap, percentage_range):
        percentages = np.linspace(percentage_range[0], percentage_range[1], percentage_range[2])
        
        # calculate thresholds for the heatmap with the threshold seperating the top 1-percentile% of the heatmap pixels
        thresholds = torch.quantile(heatmap, 1 - torch.from_numpy(percentages).to(self.device).to(torch.float32))
        thresholds = thresholds.view(-1, 1, 1, 1) # add dimensions for broadcasting
        
        masks = (heatmap < thresholds).to(self.device)
        interpolated_input = self.obfuscator(input_data)
        
        new_inputs = torch.where(masks, interpolated_input, input_data)

        return new_inputs, percentages