import torch
import torch.nn.functional as F
from src.explainability.methods.baseCAM import BaseCAM

class EigenCAM(BaseCAM):
    """ EigenCAM Class following the paper Eigen-CAM: Class Activation Map using Principal Components (https://arxiv.org/abs/2007.12391)

        Args:
            model (torch.nn.Module): model to be used
            layers (list): list of layers to be used
            tip_selection (bool): (useless, just for convenience) determine whether to use segmentated area as score or the global average
            scaling (bool): determine if we normalize the heatmap or keep the quantitative values
        
        Returns:
            torch.Tensor: heatmap (not attached to the computational graph, cannot be used for LSX or other backpropagation methods)
    """
    def __init__(self, 
                model : torch.nn.Module,
                layers : list[str] | str,
                tip_selection : bool = False,
                scaling : bool = True) -> None:
        super().__init__(model, layers, tip_selection=False)
        self.scaling = scaling

    def __call__(self,
                input_data : torch.Tensor,
                target_data : torch.Tensor = None) -> torch.Tensor:
        """ Define the forward pass of the EigenCAM algorithm """
        self.model.eval()

        with torch.no_grad(): # we don't need gradients for this
            _ = self.model(input_data)  # Forward pass to populate self.features

            # create empty heatmap tensor to aggregate the results from each layer. Dimension: (batch_size, 1, w, h) for spatial input shape (w,h)
            heatmap = torch.zeros((input_data.shape[0], 1, *input_data.shape[-2:]), device=self.device)
            # compute the heatmap for each layer
            for layer in self.layers:
                heatmap += self.single_layer(input_data, self.features[layer])
        
        heatmap = self.normalize(heatmap) if self.scaling else heatmap
        return heatmap

    def single_layer(self, input_data, feature):
        """ Compute the heatmap for a single layer following the EigenCAM algorithm (https://arxiv.org/abs/2007.12391)"""

        # prepare storage as we can't compute the eigenvalues for the whole batch at once
        heatmap = torch.zeros((input_data.shape[0], 1, *input_data.shape[-2:]), device=self.device)
        
        # compute the heatmap for each batch element
        for b in range(input_data.shape[0]):
            O_k_b = feature[b] # get the feature map for the current batch element
            
            # flatten the feature map and transpose it to have the spatial dimensions as the first dimension
            #O_k_r = O_k_b.flatten().reshape(O_k_b.shape[0], -1).transpose(0, 1)
            O_k = O_k_b.reshape(O_k_b.shape[0], -1).transpose(0, 1)
            
            # center the feature map (following JacobGil's implementation)
            O_k = O_k - O_k.mean(dim=0)
            
            # compute the eigenvector for the largest eigenvalue (eq. 2 in the paper)
            _, _, Vh = torch.linalg.svd(O_k, full_matrices=True)
            
            # calculate projection of the feature map onto the eigenvector (eq. 3 in the paper)
            Leig = (O_k @ Vh[0, :]).reshape(O_k_b.shape[1:])
            
            #Leig = torch.relu(Leig) # technically not suggested in the paper but following the general CAM approach

            # interpolate the heatmap to the input shape
            heatmap[b] = self.interpolate(Leig.unsqueeze(0).unsqueeze(0), input_data[b].unsqueeze(0))

        return heatmap