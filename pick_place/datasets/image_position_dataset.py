import json
import pathlib
from random import randint

import PIL
import PIL.Image

import numpy as np
import torch
import torchvision

class ImagePositionDataset(torch.utils.data.Dataset):

    """
    Dataset for pairwise image + position vector
    prediction.

    Expected directory structure:

    dataset/
        sample.json
        xy_sample.json
        xyr_sample.json
        data_xyr/
            _________.png
            ...
        data_xy/
            _________.png
    """

    def __init__(
        self,
        dataset_path: str,
        label_name: str = "xy",
        img_width: int = 2592,
        img_height: int = 1944,
        use_transform=True
    ):
        self.use_transform = use_transform
        self.dataset_path = pathlib.Path(dataset_path)
        if label_name:
            label_file = f"{label_name}_sample.json"
        else:
            label_file = "sample.json"

        self.labels_path = self.dataset_path / label_file

        with open(self.labels_path) as f:
            self.all_labels = json.load(f)
        self.labels = []

        self.image_dir = self.dataset_path / f"data_{label_name}"

        for label in self.all_labels:
            if (self.image_dir / pathlib.Path(label["filename"] + ".png")).is_file():
                self.labels.append(label)
        #print(len(self.labels))

        self.img_width = img_width
        self.img_height = img_height
        
    def __len__(self):
        return len(self.labels)
    
    def __str__(self):
        return f"Image Position Dataset [{len(self)}]"
    
    def __getitem__(self, idx):
        image_file_name, label = self.get_label_data(idx)
        image = PIL.Image.open(self.image_dir / f"{image_file_name}.png")
        image = self.get_image(image_file_name)
        if image is None:
            return None
        
        if self.use_transform:
            idx2 = randint(0, len(self.labels) - 1)
            ref_image_file_name, ref_image_label = self.get_label_data(idx2)
            
            image = image.repeat(3, 1, 1)
            ref_image = self.get_image(ref_image_file_name, rgb=True)
            if ref_image is None:
                return None
            label: torch.Tensor = ref_image_label - label
            return image.float(), ref_image.float(), label.float()
        else:
            return image.float(), label.float()
    
    def get_image(self, image_file_name, rgb=False):
        """Return Image object using file name or None if not found."""
        try:
            image = PIL.Image.open(self.image_dir / f"{image_file_name}.png")
            image = torchvision.transforms.functional.resize(
                image, [self.img_height, self.img_width]
            )
            image = torch.from_numpy(np.array(image)).unsqueeze(0)
            if rgb:
                image = image.repeat(3, 1, 1)
            return image
        except:
            return None
        
    def get_label_data(self, idx):
        """Return image file name and label vector corresponding to index."""
        label_json = self.labels[idx]
        return label_json["filename"], torch.tensor([
            label_json["x"],
            label_json["y"],
            label_json["r"]
        ])

        
    def get_image_label_pairs(self, idx1, idx2=None):
        """
        Returns two image-label pairs

        Params:
        idx1: index of one image-label pair in the dataset
        idx2 (Optional): index of another image-label pair in the dataset 
            (defaults to a random integer if not provided)
        """
        if not idx2:
            idx2 = randint(0, len(self.labels) - 1)

        return self[idx1], self[idx2]

    def collate_fn(batch):
        """Handles missing images during loading."""
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

if __name__ == "__main__":
    image_position_dataset = ImagePositionDataset(dataset_path="/data/brightmachines/dataset/", label_name="xyr")
    print(image_position_dataset[0])