import torch
import torch.nn.functional as F
from src.explainability.methods.baseCAM import BaseCAM

class GradCAMEW(BaseCAM):
    """
        GradCAM-Element-wise.
        Following the paper "Explainable Models with Consistent Interpretations" by Vipin Pillai et al.
        Args:
            model (torch.nn.Module): model to be used
            layers (list): list of layers to be used
            input_shape (tuple): spatial input shape (w,h) of the model (default: (256, 256))
            tip_selection (bool): determine whether to use segmentated area as score or the global average

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

    def __call__(self, input_data, target_data=None):
        """ Define the forward pass of the GradCAM-EW algorithm """
        self.model.eval()

        # Forward pass to populate self.features
        output = self.model(input_data)
        
        # Compute the gradients
        grads = self.gradients(output, target_data, use_activation=False)

        # create empty heatmap tensor to aggregate the results from each layer. Dimension: (batch_size, 1, w, h) for spatial input shape (w,h)
        heatmap = torch.zeros(size=(input_data.shape[0], 1, *input_data.shape[-2:]), device=self.device)
        # compute the heatmap for each layer
        for layer in self.layers:
            heatmap += self._single_layer(input_data, grads[layer], self.features[layer])
        
        heatmap = self.normalize(heatmap) if self.scaling else heatmap
        return heatmap

    
    def _single_layer(self, input_data, grads, feature):
        """ GradCAM-EW algorithm for a single layer """
        
        weights = grads # eq. 3 in: Explainable Models with Consistent Interpretations Vipin Pillai, Hamed Pirsiavash
        
        # compute weighted sum of the feature maps
        heatmap = torch.sum(weights * feature, dim=1, keepdim=True) # eq. 3 in the GC-EW paper

        # Only keep positive attentions
        heatmap = torch.relu(heatmap) # eq. 3 in the GC-EW paper

        # interpolate the heatmap to the input shape
        return self.interpolate(heatmap, input_data)