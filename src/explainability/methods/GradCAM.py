import torch
import torch.nn.functional as F
from src.explainability.methods.baseCAM import BaseCAM

class GradCAM(BaseCAM):
    """ GradCAM Class 
    
    GradCAM: Visual Explanations from Deep Networks via Gradient-based Localization
    https://arxiv.org/abs/1610.02391

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


    def __call__(self,
                input_data : torch.Tensor,
                agt_data : torch.Tensor = None,
                target_data : torch.Tensor = None) -> torch.Tensor:
        """ Define the forward pass of the GradCAM algorithm """

        # Forward pass to populate self.features
        if agt_data is not None:
            input_data = agt_data
        
        self.model.eval()
        output = self.model(input_data)

        # Compute the gradients
        grads = self.gradients(output, target_data, use_activation=False)

        # create empty heatmap tensor to aggregate the results from each layer. Dimension: (batch_size, 1, w, h) for spatial input shape (w,h)
        heatmap = torch.zeros((input_data.shape[0], 1, input_data.shape[-2], input_data.shape[-1])).to(self.device)
        
        # compute the heatmap for each layer
        for l in self.layers:
            heatmap += self.single_layer(input_data, grads[l], feature=self.features[l])
            #print(heatmap.max(), heatmap.min())
        
      # normalize the heatmap if scaling is set to True
        heatmap = self.normalize(heatmap) if self.scaling else heatmap

        if agt_data is None:
            return heatmap
        return heatmap, output

    def single_layer(self, input_data, grads, feature):
        """ Compute the heatmap for a single layer following the GradCAM algorithm (https://arxiv.org/pdf/1610.02391)"""
        
        # weights as the spatial average of the gradients (eq. 1 in the paper)
        weights = torch.mean(grads, dim=(-2, -1), keepdim=True) # output shape: (len(features), 1, 1, 1)
        
        # weighted sum of the feature maps (eq. 2 in the paper)
        heatmap = torch.sum(weights * feature, dim=1, keepdim=True)
        #print(weights.max(), weights.min())

        # apply ReLU to the heatmap (eq. 2 in the paper)
        heatmap = torch.relu(heatmap)

        # interpolate the heatmap to the input data spatial shape
        return self.interpolate(heatmap, input_data)