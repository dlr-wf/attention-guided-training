import torch
import numpy as np

from src.explainability.metrics.baseclass import Co12Base

class Compactness(Co12Base):
    def __init__(self, 
                model: torch.nn.Module, 
                criterion: callable, 
                obfuscator: callable,
                explainer: callable):
        """
        Implementation of the Compactness metric.
        
        Args:
            model (torch.nn.Module): The model to be tested.
            criterion (callable): The loss function to evaluate the model's predictions.
            obfuscator (callable): The obfuscation method to apply to the input data.
            explainer (callable): The explainer method to compute heatmaps.
        """
        self.model = model
        self.criterion = criterion
        self.obfuscator = obfuscator
        self.explainer = explainer
        self.device = next(model.parameters()).device

    def __call__(self, 
                input_data: torch.Tensor, 
                num_steps: int = 100, 
                threshold: float = 0.8):
        """
        Calculates the compactness metric by evaluating when the model's performance reaches a certain threshold after un-obfuscating the input data.
        If the threshold is reached with the least possible amount of features provided, we postulate that the heatmap is compact, as it contains only the most relevant features and could be truncated to a compact representation without loss of information.
        
        Args:
            input_data (torch.Tensor): Input data to be evaluated.
            num_steps (int): Number of steps for interpolation.
            threshold (float): Threshold for the model's performance to consider the heatmap compact.
        
        Returns:
            List: A list containing the earliest percentage at which the threshold is reached for each sample.
            Tuple: A tuple containing the scores and percentages used for the evaluation.
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
            percentage_range = (0, 1, num_steps) # we always want to interpolate from 0 to 100% of the heatmap to ensure that we surpass the threshold
            new_inputs, percentages = self._generate_interpolated_inputs(current_input, heatmap, percentage_range)
            batch_scores = []
            
            with torch.no_grad():
                for j in range(new_inputs.size(0)):
                    pred = torch.sigmoid(self.model(new_inputs[j].unsqueeze(0))).detach()
                    score = self._compute_dice_score(pred, target_data)
                    batch_scores.append(score)
            
            scores.append(batch_scores)
        scores = np.array(scores)
        
        # Calculate earliest threshold percentage with uncertainty for each layer/explainer
        earliest_list = []
        for score_list in scores:
            earliest_percentage = self._find_earliest_threshold(percentages, score_list, threshold)
            earliest_list.append(earliest_percentage)
        
        return earliest_list, (scores, percentages)


    def _generate_interpolated_inputs(self, input_data, heatmap, percentage_range):
        percentages = np.linspace(percentage_range[0], percentage_range[1], percentage_range[2])
        
        # calculate thresholds for the heatmap with the threshold seperating the top 1-percentile% of the heatmap pixels
        thresholds = torch.quantile(heatmap, 1 - torch.from_numpy(percentages).to(self.device).to(torch.float32))
        thresholds = thresholds.view(-1, 1, 1, 1) # add dimensions for broadcasting
        
        masks = (heatmap < thresholds).to(self.device)
        interpolated_input = self.obfuscator(input_data)
        
        new_inputs = torch.where(masks, interpolated_input, input_data)

        return new_inputs, percentages


    def _find_earliest_threshold(self, 
                                percentages: np.ndarray,
                                scores,
                                threshold: float = 0.9):
        earliest_mean = 1.0
        
        for p, s_mean in zip(percentages, scores):
            if s_mean > threshold and earliest_mean == 1.0:
                earliest_mean = p

        return earliest_mean