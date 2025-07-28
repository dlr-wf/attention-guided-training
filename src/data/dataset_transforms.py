import PIL
import math
import torch
import random
import numpy as np
from torchvision.transforms import Compose
import torchvision.transforms.functional as F


class TransformSet:
    """
    Class to store sets of transformations for data Preprocessing, depending on the type of data and experiment.
    
    Properties:
        transforms (dict): Dictionary with transformation sets. Possible keys are:
            - "full": Full set of transformations including data augmentation.
            - "minimal": Minimal set of transformations without data augmentation.
            - "validation": Set of transformations for validation data.
            - "no crack tips": Set of transformations for datasets without defined crack tip targets.
    """
    def __init__(self, 
                 tip_size : int = 1):
        """
        Dictionaries with the transformation sets are initialized.
        
        Args:
            tip_size (int): Size of the area around the crack tip to enhance. Default is 1.
        
        Raises:
            ValueError: If the transforms_name is not valid.
        """
        
        self.transforms = {
            "full": Compose([
                EnhanceTip(size=tip_size), # enlarge the single-pixel tip to ensure that the tip doesn't accidentally vanish due during another transformation
                InputNormalization(), 
                CrackTipNormalization(),
                RandomCrop(size=[120, 180], left=[10, 30]),
                RandomRotation(degrees=10),
                Resize(size=224),
                RandomFlip(),
                ToCrackTipMasks(),
            ]),

            "minimal": Compose([
                EnhanceTip(size=tip_size),
                InputNormalization(),
                CrackTipNormalization(),
                Resize(size=224),
                ToCrackTipMasks(),
            ]),

            "validation": Compose([
                EnhanceTip(size=1),
                InputNormalization(),
                CrackTipNormalization(),
                ToCrackTipMasks(),
            ]),

            "no targets": Compose([
                InputNormalization(),
                ToData(),
            ]),
        }
           
############################################################################################################
# Adapted from: https://github.com/dlr-wf/explainable-crack-tip-detection
############################################################################################################

class CrackTipNormalization:
    def __call__(self, 
                 sample : dict) -> dict:
        """
            Normalize the crack tips (ground truth data).
            
            Args:
                sample (dict): Sample dictionary. Required keys are 'target' and 'tip'.
                
            Returns:
                dict: Sample dictionary with normalized crack tips.
        """
        target, tip = sample["target"], sample["tip"]

        center = torch.tensor(target.shape, dtype=torch.float32) / 2.
        tip_normalized = (tip - center) / center
        sample["tip"] = tip_normalized

        return sample

class InputNormalization:
    def __init__(self, 
                 means : list | None = None, 
                 stds : list | None = None) -> None:
        """
        Initialize the InputNormalization class.
        
        Args:
            means (list-like or None): List-like of means for each channel. If None, the mean will be calculated from the input data.
            stds (list-like or None): List-like of standard deviations for each channel. If None, the std will be calculated from the input data.
        """
        self.means = np.asarray(means).reshape((-1, 1, 1)) if means is not None else None
        self.stds = np.asarray(stds).reshape((-1, 1, 1)) if stds is not None else None

    def __call__(self, sample):
        """
        Normalize the input image.
        
        Args:
            sample (dict): Sample dictionary. Required key is 'input'.
        
        Returns:
            dict: Sample dictionary with normalized input image.
        """
        img = np.asarray(sample["input"])
        means = img.mean(axis=(1, 2), keepdims=True) if self.means is None else self.means
        stds = img.std(axis=(1, 2), keepdims=True) if self.stds is None else self.stds
        img = (img - means) / (stds + 1e-12)
        sample["input"] = torch.tensor(img, dtype=torch.float32)

        return sample
    
class EnhanceTip:
    def __init__(self, 
                 size : int = 1):
        """
        Initialize the EnhanceTip class.
        
        Args:
            size (int): Size of the area around the crack tip to enhance. Default is 1.
        """
        self.size = size

    def __call__(self, sample):
        """
        Enhance the crack tip position with width 1.
        This is necessary because otherwise the crack tip position might vanish with the transformations 'Resize' or 'RandomRotation' which involve interpolation.
            
        Args:
            sample (dict): Sample dictionary. Required keys are 'target' and 'tip'.
        
        Returns:
            dict: Sample dictionary with enhanced crack tip position.
        """
        target, tip = sample['target'], sample['tip']
        # Write '2's of width 'size' indicating crack tip around the target tip position
        # This produces (size + 1)x(size + 1) square crack tip pixels instead of just 1.
        size = self.size
        if size == 1:
            target[int(tip[0].item() - 1), int(tip[1].item() - 1):int(tip[1].item() + 2)] = 2
            target[int(tip[0].item()), int(tip[1].item() - 1):int(tip[1].item() + 2)] = 2
            target[int(tip[0].item() + 1), int(tip[1].item() - 1):int(tip[1].item() + 2)] = 2
        else:
            row_start = int(max(tip[0].item() - size -1, 0))
            row_end = int(min(tip[0].item() + size + 1, target.shape[0]))
            col_start = int(max(tip[1].item() - size - 1, 0))
            col_end = int(min(tip[1].item() + size + 1, target.shape[1]))
            # Set the area around the tip to '2'
            target[row_start:row_end, col_start:col_end] = 2

        sample["target"] = target

        return sample
    

class RandomCrop:


    def __init__(self, 
                 size : int | tuple | list, 
                 left : list | None = None):
        """
        Initialize the RandomCrop class.
        
        Args:
            size (int, tuple, or list): Size of the crop.
                        if int: crop size is ('size', 'size')
                        if tuple: crop size is 'size'
                        if list: crop size is randomly chosen in interval 'size'
            left (list or None): If not None, left-side of the crop is randomly chosen in interval 'left'.
        """
        if left is not None:
            assert isinstance(left, list) and len(left) == 2
        self.left = left
        assert isinstance(size, (int, tuple, list))
        if isinstance(size, int):
            self.size = (size, size)
            self.random_size = False
        elif isinstance(size, tuple):
            assert len(size) == 2
            self.size = size
            self.random_size = False
        else:
            assert len(size) == 2
            self.size = size
            self.random_size = True

    def __call__(self, sample):
        """
        Crop randomly the image & labels in a sample.
        
        Args:
            sample (dict): Sample dictionary. Required keys are 'input', 'target', and 'tip'. Optionally 'explanation'.
            
        Returns:
            dict: Sample dictionary with cropped input, target, and tip.
        """
        # check if we have a target i.e. a training dataset
        input, target, tip = sample['input'], sample['target'], sample['tip']

        height, width = input.shape[1:3]
        if self.random_size:
            new_height = np.random.randint(self.size[0], self.size[1])
            new_width = new_height
        else:
            new_height, new_width = self.size

        top = np.random.randint(0, height - new_height)
        if self.left is not None:
            left = np.random.randint(self.left[0], self.left[1])
        else:
            left = np.random.randint(0, width - new_width)

        sample["input"] = input[:, top: top + new_height, left: left + new_width]
        sample["target"] = target[top: top + new_height, left: left + new_width]
        sample["tip"] = tip - torch.tensor([top, left])

        # check if we have an explanation
        if "explanation" in sample.keys():
            explanation = sample["explanation"]
            sample["explanation"] = explanation[:,None, top: top + new_height, left: left + new_width]

        return sample
    
class RandomRotation:
    def __init__(self, 
                 degrees : int | tuple):
        """
        Initialize the RandomRotation class.
        
        Args:
            degrees (int or tuple): Range of degrees to select from. If int, the range is (-degrees, degrees).
                                    If tuple, the range is (degrees[0], degrees[1]).
        """
        assert isinstance(degrees, (int, tuple))
        if isinstance(degrees, int):
            assert 0 <= degrees <= 90
            self.degree_min = -degrees
            self.degree_max = degrees
        else:
            assert len(degrees) == 2
            self.degree_min, self.degree_max = degrees
            assert self.degree_min >= -45 and self.degree_max <= 45

    def __call__(self, 
                 sample : dict) -> dict:
        """
        Rotate the input, target and crack tip randomly with the same angle.
        
        Args:
            sample (dict): Sample dictionary. Required keys are 'input', 'target', and 'tip'. Optionally 'explanation'.
            
        Returns
            dict: Sample dictionary with rotated input, target, and tip.
        """
        input, target, tip = sample["input"], sample["target"], sample["tip"]
        in_size = target.shape[0]

        # Rotate and crop to avoid padding at the corners
        angle = random.uniform(self.degree_min, self.degree_max)

        input = F.rotate(input, angle)
        target = F.rotate(target.unsqueeze(0), angle, interpolation=PIL.Image.NEAREST).squeeze()
        tip_rot = self.rotate_point(origin=(in_size / 2., in_size / 2.),
                               point=(tip[0].item(), tip[1].item()),
                               angle_rad=np.deg2rad(angle))

        crop_size = self.calculate_crop_size(np.deg2rad(angle), in_size)
        input = F.center_crop(input, [crop_size, crop_size])
        target = F.center_crop(target, [crop_size, crop_size])
        tip = torch.tensor(tip_rot) - \
              torch.tensor([in_size - crop_size, in_size - crop_size]) / 2.
        

        if "explanation" in sample.keys():
            explanation = sample["explanation"]
            explanation = F.rotate(explanation,angle)
            explanation = F.center_crop(explanation, [crop_size, crop_size])
            sample["explanation"] = explanation

        sample["input"] = input
        sample["target"] = target
        sample["tip"] = tip
        return sample
    
    def rotate_point(self, 
                     origin : tuple, 
                     point : tuple, 
                     angle_rad : float) -> list:
        """
        Rotate a point counterclockwise by a given angle around a given origin.
        The angle should be given in radians.
        
        Args:
            origin (tuple): The origin point (ox, oy) around which to rotate.
            point (tuple): The point (px, py) to rotate.
            angle_rad (float): The angle in radians to rotate the point.
        
        Returns:
            list: The rotated point [qx, qy].
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle_rad) * (px - ox) - math.sin(angle_rad) * (py - oy)
        qy = oy + math.sin(angle_rad) * (px - ox) + math.cos(angle_rad) * (py - oy)

        return [qx, qy]


    def calculate_crop_size(self, 
                            angle_rad : float, 
                            in_size : int) -> int:
        """
        Calculates the crop size if rotation of angle is applied.
        The angle should be given in radians.
        
        Args:
            angle_rad (float): The angle in radians to rotate the point.
            in_size (int): The size of the input image.
        
        Returns:
            int: The crop size after rotation.
        """
        sin_a, cos_a = abs(math.sin(angle_rad)), abs(math.cos(angle_rad))
        cos_2a = cos_a * cos_a - sin_a * sin_a
        crop = (in_size * cos_a - in_size * sin_a) / cos_2a

        return int(crop)


class RandomFlip:
    def __init__(self, 
                 flip_probability=0.5):
        """
        Initialize the RandomFlip class.
        
        Args:
            flip_probability (float): Probability of flipping the input and target. Default is 0.5.
        """
        assert 0 <= flip_probability <= 1, "Flip probability must be between 0 and 1."
        self.flip_probability = flip_probability

    def __call__(self, sample):
        """
        Flip randomly up/down of an input & target in a sample.
        
        Args:
            sample (dict): Sample dictionary. Required keys are 'input', 'target', and 'tip'. Optionally 'explanation'.
        
        Returns:
            dict: Sample dictionary with flipped input, target, and tip.
        """
        if random.random() <= self.flip_probability:
            sample["input"] = torch.flip(sample["input"], dims=[1])
            sample["target"] = torch.flip(sample["target"], dims=[0])
            sample["tip"][0] = sample["target"].shape[-1] - sample["tip"][0]

            if "explanation" in sample.keys():
                sample["explanation"] = torch.flip(sample["explanation"], dims=[1])

        return sample
    
class Resize:
    def __init__(self, 
                 size : int | tuple):
        """
        Initialize the Resize class.
        
        Args:
            size (int or tuple): Size to resize the image to.
                        if int: resize to ('size', 'size')
                        if tuple: resize to 'size'
        """
        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            assert size > 0
            self.out_sizes = [size, size]
        else:
            assert len(size) == 2 and isinstance(size[0], int) and isinstance(size[1], int)
            assert size[0] >= 0, size[1] >= 0
            self.out_sizes = list(size)

    def __call__(self, 
                 sample : dict) -> dict:
        """
        Resize the input, target, and crack tip in a sample.
        
        Args:
            sample (dict): Sample dictionary. Required keys are 'input', 'target', and 'tip'. Optionally 'explanation'.
        
        Returns:
            dict: Sample dictionary with resized input, target, and tip.
        """
        input, target, tip = sample["input"], sample["target"], sample["tip"]
        in_sizes = torch.tensor(input.size()[1:])

        input_resized = F.resize(input, size=self.out_sizes)
        target_resized = F.resize(img=target.unsqueeze(0),
                                 size=self.out_sizes,
                                 interpolation=PIL.Image.NEAREST).squeeze()
        tip_resized = tip * torch.tensor(self.out_sizes) / in_sizes

        if "explanation" in sample.keys():
            explanation = sample["explanation"].squeeze()
            explanation = F.resize(explanation.unsqueeze(0), size=self.out_sizes, interpolation=PIL.Image.NEAREST)
            sample["explanation"] = explanation
        sample["input"] = input_resized
        sample["target"] = target_resized
        sample["tip"] = tip_resized

        return sample


class ToCrackTipMasks:
    def __call__(self, 
                 sample : dict) -> tuple:
        """
        Extract input and crack tip mask.
        
        Args:
            sample (dict): Sample dictionary. Required keys are 'input' and 'target'.
        
        Returns:
            tuple: Tuple of input tensor and crack tip mask tensor. Optionally explanation tensor.
        """
        input, target = sample["input"], sample["target"]
        target = torch.where(target == 2, 1., 0.) # 2+1 classes (0 = background, 1 = crack, 2 = crack tip)
        target = target.unsqueeze(0)

        if "explanation" in sample.keys():
            explanation = sample["explanation"]
            return input, target, explanation

        return input, target

class ToCrackTipsAndMasks:
    def __call__(self, 
                 sample : dict) -> tuple:
        """
        Extract input and tuple of crack tip mask and coordinates.

        Args:
            sample (dict): Sample dictionary. Required keys are 'input', 'target', and 'tip'.
        
        Returns:
            tuple: Tuple of input tensor, crack tip mask tensor, and crack tip coordinates. Optionally explanation tensor.
        """
        image, target, tip = sample["input"], sample["target"], sample["tip"]
        mask = torch.where(target == 2, 1., 0.)
        mask = mask.unsqueeze(0)

        if "explanation" in sample.keys():
            explanation = sample["explanation"]
            return image, (mask, tip), explanation

        return image, (mask, tip)

class ToData:
    def __call__(self, 
                 sample : dict) -> torch.Tensor:
        """
        For datasets that have no defined targets, just return the input data.
        
        Args:
            sample (dict): Sample dictionary. Required key is 'input'.
        """
        return sample['input']