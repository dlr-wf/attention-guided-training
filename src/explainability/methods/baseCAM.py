import torch
import torch.nn.functional as F

class BaseCAM:
    """
        Collection of methods to implement CAM methods for the U-Net model.
    """
    def __init__(self,
                 model : callable, 
                 layers : list[str],
                 tip_selection : bool = False) -> None:
        
        # store the model and layer
        self.model = model
        self.layers = layers if isinstance(layers, list) else [layers]

        # automatically determine the device of the model
        self.device = next(model.parameters()).device

        # store the feature maps in a dictionary with the layer name as key
        self.features = {}

        # store the hooks for removal
        self.hooks = []

        # register the hooks
        self.register_hooks()        

        # decide between GAP loss and segmentation only loss
        self.tip_selection = tip_selection


    def register_hooks(self) -> None:
        """ Register the hooks for the forward pass of the model. """
        for layer in self.layers:
            self.hooks.append(
                self.model._modules.get(layer).register_forward_hook(
                    lambda module, input, output, layer_name=layer: self.save_feature(module, input, output, layer_name)
                )
            )
    

    def save_feature(self, module : torch.nn.Module, input : torch.Tensor, output : torch.Tensor, layer_name : str) -> None:
        """ Save the feature maps of the forward pass. """
        output.requires_grad_(True) # set the gradient to True to ensure the results are differentiable
        self.features[layer_name] = output # store the feature maps in the dictionary



    def normalize(self, heatmap : torch.Tensor) -> torch.Tensor:
        """ Normalize the heatmap to be between 0 and 1 using Min-Max scaling with smoothing 1e-9. """
        mins_per_sample = torch.amin(heatmap, dim=(-2, -1), keepdim=True) # get the minimum value per sample and channel
        maxs_per_sample = torch.amax(heatmap, dim=(-2, -1), keepdim=True) # get the maximum value per sample and channel
        heatmap = (heatmap - mins_per_sample) / (maxs_per_sample - mins_per_sample + 1e-7) # normalize the heatmap
        return heatmap


    def interpolate(self, heatmap : torch.Tensor, input_data : torch.Tensor) -> torch.Tensor:
        """ Interpolate the heatmap to the input data size. """
        heatmap = F.interpolate(heatmap, size=input_data.shape[-2:], mode='bilinear', align_corners=False)
        return heatmap
    
    def get_loss(self, output : torch.Tensor, target_mask : torch.Tensor | None, use_activation : bool = False) -> torch.Tensor:
        """ Get the loss from the output and target data. """
       
        if self.tip_selection == False:
            # get score as GAP of the output following the seggradcam paper
            loss = torch.mean(output)
        elif target_mask == None:
            # determine the target data as the segmentation output of the model
            if output.size(1) == 1:
                # binary segmentation
                target_mask= torch.where(torch.sigmoid(output) > 0.5, 1., 0.)
            else:
                # multi-class segmentation
                target_mask = torch.where(torch.softmax(output, dim=1) > 0.5, 1., 0.)
            
            # calculate the loss as the mean of the segmented pixel values
            loss = torch.mean(output*target_mask)
            # apply the exponential function if the activation is used (only for GradCAM++)
            loss = torch.exp(loss) if use_activation else loss
        else:
            # calculate the loss as the mean of the pixel values in the target mask
            loss = torch.mean(output*target_mask)
            # apply the exponential function if the activation is used (only for GradCAM++)
            loss = torch.exp(loss) if use_activation else loss
        
        return loss


    def gradients(self, output, target_data, use_activation=False):
        """ Get the gradients of the loss with respect to the feature maps. """
        loss = self.get_loss(output, target_data, use_activation)

        # get the gradients for each layer
        grads = {}
        for l in self.layers:
            # calculates the gradient of the loss with respect to the feature maps (dS/dA)
            grad = torch.autograd.grad(outputs=loss,
                                        inputs=self.features[l],
                                        create_graph=True,
                                        retain_graph=True)
            grads[l] = grad[0] # get the first element of the gradient list 
        return grads

    def save_grads(self, module, grad_input, grad_output, layer_name):
        """ Save the gradients of the model. """
        print(f"Saving gradients for layer {layer_name} - {grad_input[0].shape}")
        self.hook_grads[layer_name] = grad_output[0]
        

    def __del__(self):
        """ Remove the hooks after the object is deleted to prevent memory leaks during inference. """
        for hook in self.hooks:
            hook.remove()