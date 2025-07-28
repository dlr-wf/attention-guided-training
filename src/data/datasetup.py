import os
import torch
from torch.utils.data.dataloader import DataLoader

from crackpy.crack_detection.data.datapreparation import import_data
from crackpy.crack_detection.data.interpolation import interpolate_on_array
from crackpy.crack_detection.utils.utilityfunctions import get_nodemaps_and_stage_nums

from src.data.dataset_params import StoredExperiments
from src.data.dataset_transforms import TransformSet
from src.data.domain_knowledge import WilliamsDomainKnowledge
from src.data.custom_datasets import CrackTipDataset, ExplCrackTipDataset

class DataSetup:
    def __init__(self, 
                data_folder="data/",
                experiment_name="S_160_4.7",
                transforms_name="minimal",
                explanation_type="gradual williams",
                explanation_upper_bound=500,
                explanation_lower_bound=100,
                tip_size=1):
        
        # storage classes to group predefined parameter sets
        self.experiment_infos = StoredExperiments()
        trfs = TransformSet(tip_size=tip_size)

        # get variables from the predefined sets
        self.experiment_name = experiment_name
        self.experiment_size = self.experiment_infos.sizes[experiment_name]
        self.experiment_hasTargets = self.experiment_infos.exists_target[experiment_name]
        self.experiment_nodemap_nums = self.experiment_infos.nodemap_nums[experiment_name]

        # domain knowledge generation
        domain_knowledge = WilliamsDomainKnowledge(
                                                   type=explanation_type, 
                                                   upper_threshold=explanation_upper_bound, 
                                                   lower_threshold=explanation_lower_bound
                                                  )

        # safeguard to prevent using transforms that are not compatible with the dataformat
        # -> we just neglect any transforms specific to training data augmentation and target processing, just normalizing the data instead
        if self.experiment_hasTargets:
            self.transforms = trfs.transforms[transforms_name.lower()]
        else:
            self.transforms = trfs.transforms["no targets"]
        
        # get data
        base_path = os.path.join(data_folder, experiment_name) # expected data structure data/<experiment_name>/raw/...
        if not os.path.exists(os.path.join(base_path, "raw", "Nodemaps")):
            print(f"Data for experiment {experiment_name} not found. Downloading...")
            import requests
            from pathlib import Path
            from zipfile import ZipFile
            os.makedirs(base_path, exist_ok=True)

            # download from zenodo
            zenodo_url = f"https://zenodo.org/records/5740216/files/{experiment_name}.zip?download=1"
            with requests.get(zenodo_url, stream=True) as r:
                print(f"Downloading data for experiment {experiment_name} from {zenodo_url} to {base_path}...")
                r.raise_for_status()
                zip_path = os.path.join(base_path, f"{experiment_name}.zip")
                with open(zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_folder)
            os.remove(zip_path)
            print(f"Data for experiment {experiment_name} downloaded and extracted.")

        # raw experimental data is not recorded on a regular grid, so we interpolate it to a regular grid for ML purposes
        if not os.path.exists(os.path.join(base_path, "processed")):
            print(f"Preparing data for experiment {experiment_name}...")
            self.stages_to_nodemaps, self.nodemaps_to_stages = get_nodemaps_and_stage_nums(
                                                                                           folder_path=os.path.join(base_path, "raw", "Nodemaps"), 
                                                                                           which =self.experiment_nodemap_nums
                                                                                          )

            for side in ['left', 'right']: # left for training, right for validation
                input_data, ground_truth = import_data(
                                                       nodemaps=self.stages_to_nodemaps, 
                                                       data_path=os.path.join(base_path,"raw"), 
                                                       side=side, 
                                                       exists_target=self.experiment_hasTargets,
                                                      )

                # interpolate semi-structured nodemap data on a more conventional regular grid
                interpolation_size = self.experiment_size if side == 'right' else self.experiment_size * -1
                _, interp_disps, _ = interpolate_on_array(input_by_nodemap=input_data, interp_size=interpolation_size, pixels=256)

                inputs = self.numpy_to_tensor(interp_disps, dtype=torch.float32)
                inputs = self.dict_to_list(inputs)

                # save inputs
                os.makedirs(os.path.join(base_path, "processed"), exist_ok=True)
                torch.save(inputs, os.path.join(base_path, "processed", f'InputData_{side}.pt'))

                if not self.experiment_hasTargets:
                    continue
                
                # get targets
                targets = self.numpy_to_tensor(ground_truth, dtype=torch.float32)
                targets = self.dict_to_list(targets)
                
                # save targets
                os.makedirs(os.path.join(base_path, "processed"), exist_ok=True)
                torch.save(targets, os.path.join(base_path, "processed", f'TargetData_{side}.pt'))
                
        # tensors to datasets and dataloaders
        datasets = {}
        
        # only get taret/label paths if there are actually labels for this experiment        
        if self.experiment_hasTargets:
            train_label_path = os.path.join(base_path, "processed", 'TargetData_right.pt')
            val_label_path = os.path.join(base_path, "processed", 'TargetData_left.pt')
        else:
            train_label_path = None
            val_label_path = None

        # input data paths (are always present)
        train_input_path = os.path.join(base_path, "processed", 'InputData_right.pt')
        val_input_path = os.path.join(base_path, "processed", 'InputData_left.pt')
        
        # create datasets
        if self.experiment_hasTargets:
            datasets["training"] = ExplCrackTipDataset(
                                                       inputs=train_input_path, 
                                                       targets=train_label_path, 
                                                       transform=self.transforms, 
                                                       critic_ai=domain_knowledge
                                                      )
            datasets["validation"] = ExplCrackTipDataset(
                                                          inputs=val_input_path, 
                                                          targets=val_label_path, 
                                                          transform=trfs.transforms["validation"], 
                                                          critic_ai=domain_knowledge
                                                         )
        else:
            datasets["training"] = CrackTipDataset(
                                                   inputs=train_input_path, 
                                                   targets=None, 
                                                   transform=trfs.transforms["no targets"]
                                                  )
            datasets["validation"] = CrackTipDataset(
                                                     inputs=val_input_path, 
                                                     targets=None, 
                                                     transform=trfs.transforms["no targets"]
                                                    )

        if self.experiment_hasTargets and transforms_name.lower() != "minimal":
            dataloaders = {x: DataLoader(datasets[x], batch_size=16, shuffle=(x=="training"), num_workers=4) for x in datasets.keys()}
        else:
            dataloaders = {x: DataLoader(datasets[x], batch_size=16, shuffle=False, num_workers=4) for x in datasets.keys()}

        self.dataloaders=dataloaders

    def numpy_to_tensor(self, numpy_dict, dtype):
        """Convert a dict of numpy arrays into a dict of 'unsqueezed' tensors of 'dtype'."""
        return {key: torch.tensor(value.copy(), dtype=dtype).unsqueeze(0) for key, value in numpy_dict.items()}
    
    def dict_to_list(self, dictionary):
        """Convert a dictionary into a list by loosing the keys."""
        return [value for key, value in dictionary.items()]