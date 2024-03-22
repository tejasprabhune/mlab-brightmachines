import torch
import yaml
import pathlib
from pick_place import models, datasets

class Configs:
    """
    Config utility functions for reproducible models.
    """

    @staticmethod
    def load_config(config_path: str) -> dict:
        """
        Load config file.

        Args:
            config (str): Config path
        
        Returns:
            Dictionary of configuration
        """
        config_path = pathlib.Path(config_path)
        return yaml.safe_load(config_path.read_text())

    @staticmethod
    def load_model(config: dict, ckpt: str = None) -> torch.nn.Module:
        """
        Load trained model.
        
        Args:
            ckpt (str): Checkpoint path
            config (dict): Configuration dict
        
        Returns:
            model (torch.nn.Module): Model based on config
        """
        model_name = config.get("model", "SiameseCNN")
        model_params = config.get("model_params", {})
        model = getattr(models, model_name)(**model_params)


        if ckpt is not None:
            ckpt = pathlib.Path(ckpt)
            model.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state"])
        
        return model

    
    @staticmethod
    def load_criterion(config: dict, criterions: dict):
        """
        Load loss from dict of criterions, with L1Loss as default.

        Args:
            config (dict): Configuration dict
            criterions (dict): All possible losses dict
        """
        loss = config.get("loss", "l1")
        return criterions.get(loss)(**config.get("loss_params", {}))
    
    @staticmethod
    def load_optimizer(config: dict, model: torch.nn.Module):
        optim_params = config.get("optim_params", {})
        optimizer = torch.optim.Adam(model.parameters(), **optim_params)

        return optimizer
    
    @staticmethod
    def load_dataset(config: dict):
        """
        Load train, val, and test datasets based on config.

        Args:
            config (dict): Configuration dict
        """

        dataset_name = config.get("dataset", "ImagePositionDataset")
        dataset_params = config.get("dataset_params", {})
        dataset = getattr(datasets, dataset_name)(**dataset_params)

        train_split = config.get("train_split", 0.8)
        val_split = config.get("val_split", 0.1)

        num_train_samples = int(len(dataset) * train_split)
        num_val_samples = int(len(dataset) * val_split)
        num_test_samples = len(dataset) - num_train_samples - num_val_samples

        (train_data, val_data, test_data) = torch.utils.data.random_split(
            dataset,
        	[
                num_train_samples, 
                num_val_samples, 
                num_test_samples
            ],
        	generator=torch.Generator().manual_seed(config.get("seed", 0))
        )

        return train_data, val_data, test_data
    
    @staticmethod
    def load_dataloader(config: dict, dataset: torch.utils.data.Dataset):
        """
        Load dataloader for a dataset based on config.

        Args:
            config (dict): Configuration dict
            dataset: Dataset to create DataLoader from
        """

        dataloader_params: dict = config.get("dataloader_params", {})

        dataset_name = config.get("dataset", "ImagePositionDataset")
        dataset_class = getattr(datasets, dataset_name)
        collate_fn_name = dataloader_params.get("collate_fn", None)
        if type(collate_fn_name) == str:
            dataloader_params["collate_fn"] = getattr(
                dataset_class, 
                collate_fn_name
            )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            **dataloader_params
        )

        return dataloader
    
    @staticmethod
    def generate_checkpoint_dir(config: dict, val_loss: float):
        model_name = config.get("model_key", "scnn")
        features = config.get("model_params")["features"]
        loss = config.get("loss")

        return pathlib.Path(f"{model_name}_{features}_{loss}_{val_loss:.2f}.pth")

if __name__ == "__main__":

    # Sanity Checks
    config = Configs.load_config("../ckpts/scnn_l1/scnn_l1.yml")

    print(config)
    print(Configs.load_model(config, "../ckpts/scnn_l1/scnn_l1_0.15.pth"))

    criterions = {
        "l1": torch.nn.L1Loss
    }

    print(Configs.load_criterion(config, criterions))

    dataset = Configs.load_dataset(config)

    print(Configs.load_dataloader(config, dataset))