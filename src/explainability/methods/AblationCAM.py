import torch
import torch.nn.functional as F
from src.explainability.methods.baseCAM import BaseCAM
from src.deeplearning.loss_functions import DiceLoss

class AblationCAM(BaseCAM):
    """
    AblationCAM implementation as described in the paper "Ablation-CAM: Visual Explanations for Deep Convolutional Network via Gradient-free Localization" by Desai et al.
    
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
                layers : list[str],
                ablation_mode : str = "zero",
                tip_selection : bool = False,
                scaling : bool = True) -> None:
        super().__init__(model, layers, tip_selection=False)
        # determine how we ablate the feature maps during the forward pass
        self.ablation_mode = ablation_mode
        # score calculation for estimating the decrease in confidence
        self.criterion = DiceLoss()
        # determine whether to scale the heatmap
        self.scaling = scaling

    def __call__(self, 
                input_data : torch.Tensor) -> torch.Tensor:
        # set the model to evaluation mode
        self.model.eval()

        # determine baseline
        with torch.no_grad():
            baseline_predictions = self.model(input_data)
            baseline_predictions = torch.where(torch.sigmoid(baseline_predictions) > 0.5, 1, 0)

        # remove the hooks to avoid memory leaks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        # create empty heatmap tensor to aggregate the results from each layer. Dimension: (batch_size, 1, w, h) for spatial input shape (w,h)
        heatmap = torch.zeros((input_data.shape[0], 1, *input_data.shape[-2:]), device=self.device)

        # compute the heatmap for each layer
        for layer in self.layers:
            with torch.no_grad():
                heatmap += self._single_layer(input_data, self.features[layer], baseline_predictions, layer)
        
        heatmap = self.normalize(heatmap) if self.scaling else heatmap
        return heatmap

    def _single_layer(self,
                    input_data : torch.Tensor,
                    feature : torch.Tensor,
                    baseline_predictions : torch.Tensor,
                    layer_index : str) -> torch.Tensor:
        
        # interpolate the feature maps to the input shape
        activations_upsampled = self.interpolate(feature, input_data)
        
        # empty heatmap tensor to aggregate the results from each feature map. Dimension: (batch_size, 1, w, h) for spatial input shape (w,h)
        heatmap = torch.zeros((input_data.shape[0], 1, *input_data.shape[-2:]), device=self.device)

        # calculate score after ablating each feature map
        for feature_idx in range(feature.size(1)):
            hook = self.apply_ablation(layer_index, feature_idx)
            ablated_predictions = torch.sigmoid(self.model(input_data))
            hook.remove() # remove the hook to avoid memory leaks

            weights = 1 - self.criterion(ablated_predictions, baseline_predictions) # deviating from the paper to use the DiceLoss as the score calculation for segmentation models (vs. classification scores)
            
            # compute the weighted sum of the feature maps
            heatmap += torch.relu(torch.sum(weights.unsqueeze(-1) * activations_upsampled[:, feature_idx:feature_idx+1], dim=1, keepdim=True)).to(self.device)

        return self.normalize(heatmap)

    def apply_ablation(self, layer_index, feature_map_indices):
        """
        Register a forward hook to ablate the feature maps in the target layer during the forward pass.
        """
        def forward_hook(module, input, output):
            # ablate the feature maps by setting them to zero/average/min during the forward pass
            if self.ablation_mode == "zero":
                output[:, feature_map_indices, :, :] = 0
            elif self.ablation_mode == "avg":
                output[:, feature_map_indices, :, :] = output.mean()
            elif self.ablation_mode == "min":
                output[:, feature_map_indices, :, :] = output.min()
            return output

        target_layer = self.model._modules[layer_index]
        return target_layer.register_forward_hook(forward_hook)

    def normalize(self, heatmap):
        heatmap = torch.relu(heatmap)
        return super().normalize(heatmap)