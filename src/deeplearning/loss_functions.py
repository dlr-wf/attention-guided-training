from torch import nn
from torch.nn.functional import cosine_similarity

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        """
        Dice loss for binary segmentation tasks.
        
        Args:
            eps (float): Small value to avoid division by zero.
        """
        super().__init__()
        self.eps = eps

    def forward(self, prediction, target):
        """
        Computes the Dice loss between the predicted and target tensors.

        Args:
            prediction (torch.Tensor): Prediction tensor with continous values.
            target (torch.Tensor): Target tensor with binary values (0 or 1).

        Raises:
            AssertionError: If the prediction and target tensors do not have the same size or length.

        Returns:
            torch.Tensor: Dice loss value.
        """
        if not prediction.size() == target.size() and len(prediction.size()) == 4:
            raise AssertionError("'prediction' and 'target' need to have length 4 and the same size")
        
        prediction = prediction[:, 0].contiguous().view(-1)
        target = target[:, 0].contiguous().view(-1)
        intersection = (prediction * target).sum()
        dsc = (2. * intersection + self.eps) / (prediction.sum() + target.sum() + self.eps)
        return 1. - dsc

class CSILoss(nn.Module):
    def __init__(self, 
                 weight_factor : float = 1.) -> None:
        """
        Cosine similarity loss.
        
        Args:
            weight_factor (float): Factor to scale the loss.
        """
        super().__init__()
        self.weight_factor = weight_factor
      
    def forward(self, prediction, target):
        """
        Calculates the cosine similarity loss between the predicted and target tensors.
        
        Args:
            prediction (torch.Tensor): Prediction tensor with arbitrary values.
            target (torch.Tensor): Target tensor with arbitrary values.
        
        Raises:
            AssertionError: If the prediction and target tensors do not have the same size or length.
            
        Returns:
            torch.Tensor: Cosine similarity loss value.
        """
        loss = self.weight_factor * (1 - cosine_similarity(prediction.flatten(1), target.flatten(1), dim=1))
        return loss.mean()
    