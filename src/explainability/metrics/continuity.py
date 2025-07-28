import torch
import numpy as np

from src.explainability.metrics.baseclass import Co12Base
from src.explainability.metrics.ssim import ssim

class Continuity(Co12Base):
    def __init__(self, 
                model: torch.nn.Module, 
                explainer: callable):
        """
        Task specific implementation of the Continuity metric.
        
        Args:
            model (torch.nn.Module): The model to be tested.
            explainer (callable): The explainer method to compute heatmaps.
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.explainer = explainer 

    def __call__(self, 
                 input_data : torch.Tensor) -> np.ndarray:
        """
        Compute the continuity metric for the given input data.
        Here we know that the input data is a continous sequence of pairwise similar images, hence we postulate that the heatmaps should also be similar.
        
        Args:
            input_data (torch.Tensor): Input data to be evaluated. Needs to be a sorted sequence of pairwise similar images.
        
        Returns:
            scores (np.ndarray): Array of SSIM scores between consecutive heatmaps.
        """
        self.model.eval()
        
        heatmaps = []
        sample_indices = []
        
        for idx, sample in enumerate(input_data):
            
            sample = sample.to(self.device)
            heatmap, pred = self._compute_heatmap(sample.unsqueeze(0))

            if heatmap is None:
                continue
            
            # Compute and store heatmap
            heatmaps.append(heatmap)
            sample_indices.append(idx)

        heatmaps = torch.cat(heatmaps, dim=0)
        scores = self._calculate_ssim_scores(heatmaps)

        return scores

    
    def _calculate_ssim_scores(self, heatmaps):
        scores = []
        for i in range(len(heatmaps) - 1):
            score = ssim(heatmaps[i].unsqueeze(0), heatmaps[i+1].unsqueeze(0)).item()
            scores.append(score)
        return np.array(scores)