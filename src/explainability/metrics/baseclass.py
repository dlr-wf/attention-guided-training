import numpy as np, torch, scipy as scp

class Co12Base:
    def __init__(self, 
                model, 
                dataloader, 
                interpolator, 
                explainer):
        
        self.model = model
        self.dataloader = dataloader
        self.interpolator = interpolator
        self.explainer = explainer
    
    def _compute_heatmap(self, input_data):
        pred = self.model(input_data)
        
        pred = torch.sigmoid(pred)
        target_data = torch.where(pred > 0.5, 1, 0) # simulate the ground truth data (binary mask)

        if target_data.sum() == 0:
            return None, None
        
        # compute heatmap using the provided explainer class -> has to be a callable class
        heatmap = self.explainer(input_data).detach()
        return heatmap, target_data
    
    def get_sample(self, index):
        return self.dataloader[index]
    
    def get_batch(self, batch_size, start_index=None):
        if start_index:
            return self.dataloader[start_index:start_index+batch_size]
        else:
            indices = np.random.choice(len(self.dataloader), batch_size)
            return self.dataloader[indices]

    def interpolate_sample(self, sample):
        return self.interpolator(sample)

    def interpolate_batch(self, batch):
        return self.interpolate_sample(batch)
    
    def _compute_dice_score(self, pred, target):
        dice_score = 1 - self.criterion(pred, target).item()
        return dice_score

    
    def _calculate_auc_with_uncertainty(self, percentages: np.ndarray, scores: np.ndarray):
        # generate samples from normal distributions within the given mean and standard deviation
        #samples = np.random.normal(scores_mean, scores_std, (n_samples, len(scores_mean)))

        # Calculate AUC for each sample set
        aucs = np.array([scp.integrate.trapezoid(score_set, percentages) for score_set in scores])

        # Normalize AUCs
        aucs_normalized = aucs / (percentages[-1] - percentages[0])

        # Calculate mean and standard deviation of AUCs
        auc_mean = np.mean(aucs_normalized)
        auc_std = np.std(aucs_normalized)

        return auc_mean, auc_std