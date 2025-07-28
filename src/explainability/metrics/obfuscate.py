import torch
import numpy as np

class Obfuscate:
    def __init__(self, 
                 scale : float):
        """
        Default obfuscation class that adds random noise to the input tensor.
        
        Args:
            scale (float): Scale of the noise to be added to the input tensor.
        """
        self.scale = scale

    def __call__(self, input_tensor):
        """
        Generates an obfuscated version of the input tensor by adding random noise.
        
        Args:
            input_tensor (torch.Tensor): Input tensor to be obfuscated.
        
        Returns:
            torch.Tensor: Obfuscated tensor with added noise.
        """
        noise = torch.randn_like(input_tensor) * self.scale
        obfuscated_tensor = input_tensor + noise
        return obfuscated_tensor

class ObfuscationTester:
    def __init__(self, 
                 model : callable, 
                 criterion : callable, 
                 scales : list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2], 
                 runs : int = 5, 
                 datapoints : int = 20):
        """
        Initialize the ObfuscationTester with model, criterion, scales, runs, and datapoints.
        
        Args:
            model (callable): The model to be tested.
            criterion (callable): The loss function to evaluate the model's predictions.
            scales (list): List of scales for obfuscation.
            runs (int): Number of runs for each scale.
            datapoints (int): Number of datapoints for interpolation.
        """
        self.model = model
        self.criterion = criterion
        self.scales = scales
        self.runs = runs
        self.datapoints = datapoints
        self.device = next(model.parameters()).device
        self.scores = {scale: [] for scale in scales}
        self.percentages = torch.linspace(0, 1, self.datapoints, device=self.device)

    def __call__(self, dataloader):
        """
        Evaluate the effect of random pixel-obfuscation on the model's predictions for a given dataset.
        
        Args:
            dataloader (DataLoader): DataLoader providing the input data.
        
        Returns:
            best_scale (float): The scale with the best performance.
            results (list): List of tuples containing scale, deviation, mean scores, and std scores.
        """
        self.model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)): # usually the dataloader returns (input, target) pairs or (input, target, explanation) triples but if we distinguish by simple > n checking, we might miss cases where only inputs are returned as actual batched inputs
                    inputs = batch[0] if (len(batch) == 3 or len(batch) == 2) else batch
                else:
                    inputs = batch # Assuming the dataloader returns only inputs if no target or explanation is provided
                
                inputs = inputs.to(self.device) # Move to device
                self._test_batch(inputs) 

        best_scale, deviation, results = self._evaluate_scales()
        return best_scale, results

    def _random_mask(self, input_shape, percentage):
        """
        Generate a random mask rather than a attention map defined ranking for obfuscation.
        This mask will randomly select pixels to obfuscate based on the given percentage as a means to estimate a random baseline obfuscation influence.
        """
        mask = torch.rand(input_shape, device=self.device) > (1 - percentage)
        return mask

    def _test_batch(self, 
                    inputs : torch.Tensor):
        """
        Obfuscate a batch of inputs.
        
        Args:
            inputs (torch.Tensor): Batch of input data.
        
        Returns:
            None
        """
        batch_size, channels, height, width = inputs.shape
        assert channels == 2, f"Expected input with 2 channels, got {channels}"

        # generate new targets using the unobfuscated predictions to allow cases where there is no target data / poorly trained models
        targets = torch.where(torch.sigmoid(self.model(inputs)) > 0.5, 1., 0.)

        # Generate all masks once for all scales and runs
        masks = torch.stack([self._random_mask((batch_size, channels, height, width), p) for p in self.percentages])  
        # Shape: (datapoints, batch_size, channels, height, width)
    
        for scale in self.scales:
            obfuscation = Obfuscate(scale=scale)
            scale_scores = []

            # Precompute fully obfuscated inputs (reminder: interpolation <-> points replaced with noisy data) for the current scale
            obfuscated_inputs = obfuscation(inputs)

            for _ in range(self.runs):
                run_scores = []
                for mask in masks:
                    new_inputs = torch.where(mask, obfuscated_inputs, inputs)
                    preds = torch.sigmoid(self.model(new_inputs))
                    dice_scores = self.criterion(preds, targets).mean().item()
                    run_scores.append(dice_scores)

                scale_scores.append(run_scores)

            self.scores[scale].append(scale_scores)

    def _evaluate_scales(self):
        """
        Evaluate the scores for each scale and find the best scale based on the deviation from an ideal curve.
        The ideal curve is defined as a decreasing function from 1 to 0.15 over the datapoints as an empirically defined target curve.
        """
        ideal_curve = 1 - 0.85 * np.linspace(0, 100, self.datapoints) / 100 # 0.85 is an empirical correction factor. Ideally we have the score reach both 1 and 0
        results = []
        for scale, scores in self.scores.items():
            scores_array = np.array(scores)
            mean_scores = scores_array.mean(axis=(0, 1))
            std_scores = scores_array.std(axis=(0, 1))
            deviation = np.mean((mean_scores - ideal_curve) ** 2)
            results.append((scale, deviation, mean_scores, std_scores))

        best_scale, best_deviation, _, _ = min(results, key=lambda x: x[1])
        return best_scale, best_deviation, results

