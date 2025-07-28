import torch
import torch.nn.functional as F
from src.explainability.methods.baseCAM import BaseCAM
from src.deeplearning.loss_functions import DiceLoss

class ScoreCAM(BaseCAM):
    """
    ScoreCAM implementation as described in the paper "Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks" by Wang et al.
    (for the old score calculation method using GAP instead of Dice Loss)

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
                input_shape=(256, 256),
                max_batch_size=16,
                tip_selection=False,
                scaling=True):
        super().__init__(model, layers, tip_selection=False)
        self.input_shape = input_shape
        self.max_batch_size = max_batch_size
        self.criterion = DiceLoss()
        self.tip_selection = tip_selection
        self.scaling = scaling


    def __call__(self, input_data, target_data=None, prediction_data=None):
        self.model.eval() # we can eval here because due to the nature of the method with the vast creation of input samples and the forward pass, we can't use the heatmap for LSX/Backpropagation anyhow
        
        # register hooks if not already done / e.g. when using multiple calls of the class
        if len(self.hooks) < 1:
            self.register_hooks()

        with torch.no_grad(): # gradient free score/weight calculation
            # Forward pass to populate self.features + calculate baseline scores -> in practice we neglect the baseline scores as they provide only a constant offset
            baseline_predictions = torch.where(torch.sigmoid(self.model(input_data)) > 0.5, 1, 0) 
            
            # set up empty heatmap tensor to aggregate the results from each layer. Dimension: (batch_size, 1, w, h) for spatial input shape (w,h)
            heatmap = torch.zeros(size=(input_data.shape[0], 1, *input_data.shape[-2:]), device=self.device)

            # remove the hooks to avoid memory leaks (this requires re-registering the hooks for the next sample)
            for hook in self.hooks:
                hook.remove()
            self.hooks = []

            for layer in self.layers:
                heatmap += self.single_layer(input_data, self.features[layer], baseline_predictions)
        
        heatmap = self.normalize(heatmap) if self.scaling else heatmap
        return heatmap
    

    def single_layer(self, input_data, feature, baseline_predictions):

        # get the feature maps from the layer and rename
        activations = feature
        # upsample the feature maps to the input shape
        # eq. 7 and eq. 8 in the paper
        activations_upsampled = self.interpolate(activations, input_data)

        # normalize the feature maps to [0,1] -> eq. 7 and eq. 8 in the paper
        activations_min = activations_upsampled.amin(dim=(-2, -1), keepdim=True)
        activations_max = activations_upsampled.amax(dim=(-2, -1), keepdim=True)
        activations_upsampled_normalized = (activations_upsampled - activations_min) / (activations_max - activations_min + 1e-17)

        # expand the normalized feature maps for multiplication with the input data -> for each feature map + input data pair we get a new set of inputs to estimate the scores
        activations_upsampled_normalized_expanded = activations_upsampled_normalized.unsqueeze(2)
        if input_data.size(1) > 1:
            activations_upsampled_normalized_expanded = activations_upsampled_normalized_expanded.expand(-1, -1, input_data.size(1), -1, -1)

        # compute the new inputs by element-wise multiplication of the normalized feature maps and the input data
        new_inputs = activations_upsampled_normalized_expanded * input_data.unsqueeze(1).expand(-1, activations_upsampled_normalized_expanded.size(1), -1, -1, -1)

        # storage for the heatmap
        heatmap = torch.zeros(size=(1, 1, *input_data.shape[-2:]), device=self.device)
        for b, inputs in enumerate(new_inputs):
            # calculate the score for each feature map -> eq. 6 in the paper
            scores = self.safe_batch_forward(inputs)
            scores = scores.to(self.device).nan_to_num(0).unsqueeze(-1)
            weights = F.softmax(scores, dim=0).unsqueeze(-1)
            heatmap += torch.sum(weights * activations_upsampled[b], dim=0)

        heatmap = torch.relu(heatmap)
        return self.interpolate(heatmap, input_data)


    def safe_batch_forward(self, inputs):
        """ mini-batch forward pass to avoid memory issues """
        results = torch.tensor([], device=self.device)
        # iterate over the input samples in mini-batches
        for i in range(0, inputs.size(0), 16):
            start = i
            end = min(i + 16, inputs.size(0))
            mini_batch = inputs[start:end]

            if self.tip_selection:
                output = self.model(mini_batch)
                predictions = torch.where(torch.sigmoid(output) > 0.5, 1, 0) # mask for the tip selection
                mini_batch_result = torch.mean(predictions*output, dim=(1, 2, 3)) # GAP with mask to calculate the score
            else:
                mini_batch_result = torch.mean(self.model(mini_batch).detach(), dim=(1, 2, 3)) # GAP to calculate the score
            results = torch.cat((results, mini_batch_result), dim=0) # concatenate the results to the entire batch
        return results