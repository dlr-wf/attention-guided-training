import torch
import torch.nn.functional as F
from src.explainability.methods.baseCAM import BaseCAM

class GradCAMpp(BaseCAM):
    """ GradCAM++ Class following the paper GradCAM++ improved visual explanations for deep convolutional networks Aditya Chattopadhyay, et al. (https://arxiv.org/abs/1710.11063)

        Args:
            model (torch.nn.Module): model to be used
            layer (str): layer to be used
            tip_selection (bool): determine whether to use segmentated area as score or the global average
            scaling (bool): determine if we normalize the heatmap or keep the quantitative values
        
        Returns:
            torch.Tensor: heatmap attached to the computational graph. Make sure to detach() before using it as numpy array.
    
    """
    def __init__(self, 
                model : torch.nn.Module,
                layers : list[str] | str,
                tip_selection : bool = False,
                scaling : bool = True):
        super().__init__(model, layers, tip_selection)
        # determine if we normalize the heatmap or keep the quantitative values
        self.scaling = scaling

    def __call__(self,
                input_data : torch.Tensor,
                agt_data : torch.Tensor = None,
                target_data : torch.Tensor = None) -> torch.Tensor:
        """ Define the forward pass of the GradCAM++ algorithm """

        # Forward pass to populate self.features
        if agt_data is None:
            self.model.eval()
            output = self.model(input_data)
        else:
            input_data = agt_data
            output = self.model(input_data)
        
        # Compute the gradients
        grads = self.gradients(output, target_data, use_activation=True) # CARE! check activation usage (paper vs jacobgil) (see 3.3 Computation Analysis in the paper)

        # create empty heatmap tensor to aggregate the results from each layer. Dimension: (batch_size, 1, w, h) for spatial input shape (w,h)
        heatmap = torch.zeros((input_data.shape[0], 1, input_data.shape[-2], input_data.shape[-1])).to(self.device)
        # compute the heatmap for each layer
        for l in self.layers:
            heatmap += self.single_layer(input_data, grads[l], feature=self.features[l])
        
        # normalize the heatmap if scaling is set to True
        heatmap = self.normalize(heatmap) if self.scaling else heatmap

        if agt_data is None:
            return heatmap
        return heatmap, output
    
    def single_layer(self, input_data, grads, feature):
        """ Compute the heatmap for a single layer following the GradCAM++ algorithm (https://arxiv.org/abs/1710.11063)"""
        
        # compute the alpha values (eq. 19 2nd part of the denominator in the paper)
        sum_activations = torch.sum(feature, dim=(-2,-1), keepdim=True)

        # compute the alpha values (eq. 19 in the paper)
        alpha = grads**2 / (2 * (grads**2) + (sum_activations * (grads**3)) + 1e-7)

        # compute the weights (eq. 5 in the paper)
        positive_grads = torch.relu(grads)
        weighted_activations = positive_grads * alpha
        weights = torch.sum(weighted_activations, dim=(-2, -1), keepdim=True)

        # weighted sum of the feature maps (eq. 20 in the paper)
        heatmap = torch.sum(weights * feature, dim=1, keepdim=True)
        heatmap = torch.relu(heatmap)

        # interpolate the heatmap to the input shape
        return self.interpolate(heatmap, input_data)
    
