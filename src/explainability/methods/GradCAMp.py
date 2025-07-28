import torch
import torch.nn.functional as F
from src.explainability.methods.baseCAM import BaseCAM

class GradCAMP(BaseCAM):
    """
        GradCAM+ implementation following the paper GradCAM++ is equivalent to GradCAM with positive gradients Miguel Lerma, et al. (https://arxiv.org/pdf/2205.10838)
    
        Args:
            model (torch.nn.Module): model to be used
            layers (list): list of layers to be used
            tip_selection (bool): determine whether to use segmentated area as score or the global average
            scaling (bool): determine if we normalize the heatmap or keep the quantitative values

        Returns:
            torch.Tensor: heatmap attached to the computational graph. Make sure to detach() before using it as numpy array.
    """
    def __init__(self, 
                model : torch.nn.Module,
                layers : list[str] | str,
                tip_selection : bool = False,
                scaling : bool = True) -> None:
        super().__init__(model, layers, tip_selection)

        # determine if we normalize the heatmap or keep the quantitative values
        self.scaling = scaling
    
    def __call__(self, 
                input_data : torch.Tensor,
                target_data : torch.Tensor = None) -> torch.Tensor:
        """ Define the forward pass of the GradCAM+ algorithm """
        self.model.eval()
        
        # Forward pass to populate self.features
        output = self.model(input_data)
        
        # Compute the gradients
        grads = self.gradients(output, target_data, use_activation=False)

        # create empty heatmap tensor to aggregate the results from each layer. Dimension: (batch_size, 1, w, h) for spatial input shape (w,h)
        heatmap = torch.zeros(size=(input_data.shape[0], 1, *input_data.shape[-2:]), device=self.device)
        # 
        for l in self.layers:
            heatmap += self.single_layer(input_data, grads[l], self.features[l])
        

        heatmap = self.normalize(heatmap) if self.scaling else heatmap
        return heatmap

    def single_layer(self, input_data, grads, feature):
        """ Compute the heatmap for a single layer following the GradCAM+ algorithm (https://arxiv.org/pdf/2205.10838)"""
        
        # weights as the spatial average of the positive gradients (eq. 14 in the paper)
        grads_pos = torch.relu(grads)
        weights = torch.mean(grads_pos, dim=(-2, -1), keepdim=True)

        # weighted sum of the feature maps (not in the paper explicitely but following the general CAM approach)
        heatmap = torch.sum(weights * feature, dim=1, keepdim=True)

        # Only keep positive attentions
        heatmap = torch.relu(heatmap)

        # interpolate the heatmap to the input shape
        return self.interpolate(heatmap, input_data)
