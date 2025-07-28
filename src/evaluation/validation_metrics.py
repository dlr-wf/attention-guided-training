import torch
import numpy as np

def validate_model(model, device, dataloader, criterion):
    """ Validate the model on validation dataset """
    with torch.no_grad():
        model.eval()
        validation_loss = []
        deviation = []
        reliability= []

        for data, target, explanation in dataloader:
            input_data = data.to(device)
            target_seg = target[0].to(device)

            pred = model(input_data)
            class_loss = criterion(torch.sigmoid(pred), target_seg)


            validation_loss.append(class_loss.item())
            deviation.append(get_deviation(pred, target_seg))
            reliability.append(get_reliability(pred, target_seg))

    return np.mean(validation_loss), np.mean(deviation), np.mean(reliability)


def get_deviation(outputs, labels):
    """
    Calculates the sum of Euklidean distances of the predictions to the **labels** (if both exist).
    The prediction is calculated by taking the mean of all segmentated pixels.
    If there is no segmented pixel or no labeled pixel, the sample is skipped.

    :param outputs (torch.tensor) of shape (B x 1 x W x H) (after sigmoid/softmax)
    :param labels (torch.tensor) of the same shape
    :return: dist_euklid (float) sum of Euklidean distances between predictions and labels
    """
    if not outputs.size() == labels.size():
        raise AssertionError("Output and Labels need to have the same torch.Size!")

    batch_size = outputs.shape[0]
    is_crack_tip = torch.where(outputs >= 0.5, 1, 0)

    dist_euklid = []
    for i in range(batch_size):
        prediction_i = torch.nonzero(is_crack_tip[i], as_tuple=False)[:, -2:] / 1.
        label_i = torch.nonzero((labels[i] == 1), as_tuple=False)[:, -2:] / 1.
        # skip the unlabeled or unsegmented
        if len(label_i) == 0 or len(prediction_i) == 0:
            continue
        prediction_i = torch.mean(prediction_i, dim=0)
        label_i = torch.mean(label_i, dim=0)
        dist = torch.sqrt(torch.sum((prediction_i - label_i) ** 2)).item()
        dist_euklid.append(dist)

    return np.asarray(dist_euklid).mean()


def get_reliability(outputs, labels):
    """
    Calculates the reliability of a segmentation model's output batch and the corresponding labels.

    :param outputs: (torch.tensor) output of the model (after Sigmoid/Softmax) (B x H x W)
    :param labels: (torch.tensor) corresponding labels (B x H x W) with 1's and 0's
    :return: score (int) reliability score (1.0 = 100 %)
    """
    if not outputs.size() == labels.size():
        raise AssertionError("Output and Labels need to have the same torch.Size!")

    batch_size = outputs.shape[0]
    is_crack_tip = torch.where(outputs >= 0.5, 1, 0)

    unpredicted = 0
    for i in range(batch_size):
        prediction_i = torch.nonzero(is_crack_tip[i], as_tuple=False)[:, -2:] / 1.
        label_i = torch.nonzero((labels[i] == 1), as_tuple=False)[:, -2:] / 1.
        # skip the unlabeled or unsegmented
        if len(label_i) > 0 and len(prediction_i) == 0:
            unpredicted += 1

    score = 1. - unpredicted / batch_size

    return score


from scipy.ndimage import label
def get_no_label_reliability(segs):
    """
    Variation of the original reliability metric which works on the outputs only.
    It calculates the reliability of a segmentation model's output batch by checking if there are any predicted crack tips. More importantly if there is only a single crack tip predicted in each sample.
    Also it checks crack tip distances between consecutive samples to check if the crack tip was segmented unrealistically far away.

    :param seg: (torch.tensor) output of the model segmentations (where sigmoid > 0.5) #(after Sigmoid/Softmax) (B x H x W) in order (non-shuffled)

    :return: score (int) reliability score (1.0 = 100 %)
    """
    batch_size = segs.size(0)
    is_crack_tip = segs#torch.where(outputs > 0.5, 1, 0)

    invalid_samples = 0
    invalid_sample_list = []

    non_segmented_samples = 0
    non_segmented_sample_list = []

    multiple_segmented_samples = 0
    multiple_segmented_sample_list = []

    for i in range(batch_size):

        # check if there are any crack tips segmented
        if torch.sum(is_crack_tip[i]) == 0:
            non_segmented_samples += 1
            non_segmented_sample_list.append(i)
            
            invalid_samples += 1
            invalid_sample_list.append(i)
            continue
        
        # check if there are multiple crack tips segmented
        labeled, num_features = label(is_crack_tip[i].cpu().numpy())
        if num_features > 1:
            multiple_segmented_samples += 1
            multiple_segmented_sample_list.append(i)
            
            invalid_samples += 1
            invalid_sample_list.append(i)
            continue

    score = 1.0 - invalid_samples / batch_size

    return score, invalid_samples, invalid_sample_list, non_segmented_sample_list, multiple_segmented_sample_list#, far_away_segmented_sample_list