import argparse
import pathlib
from tqdm import tqdm

import torch

from pick_place.models import scnn
from pick_place.datasets.image_position_dataset import ImagePositionDataset
from pick_place.utils.configs import Configs
from pick_place.criterions import scnn_loss

class Trainer():
    def __init__(
        self,
        train_steps,
        val_steps,
        epochs,
        train_dataloader,
        val_dataloader,
        model,
        criterion,
        optimizer,
        config,
        config_dir,
        device=torch.device("cpu")
    ):
        self.train_steps = train_steps
        self.val_steps = val_steps
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.config_dir = pathlib.Path(config_dir)
        self.device = device

        self.model.to(self.device)
        self.criterion.to(self.device)

    def train(self):
        epoch_tqdm = tqdm(range(self.epochs))
        for epoch in epoch_tqdm:
            epoch_tqdm.set_description(f"Epoch {epoch}")

            total_train_loss = self.train_loop(epoch)
            total_val_loss = self.val_loop()

            avg_train_loss = total_train_loss / self.train_steps
            avg_val_loss = total_val_loss / self.val_steps

            print(f"Train Loss: {avg_train_loss:.2f}")
            print(f"Val Loss: {avg_val_loss:.2f}")

            self.save_checkpoint(avg_val_loss)
            
    def train_loop(self, epoch):
        self.model.train()
        total_train_loss = 0
        train_tqdm = tqdm(self.train_dataloader)
        train_tqdm.set_description(f"Epoch {epoch}")
        for (image, ref_image, y) in train_tqdm:
            loss = self.train_step(image, ref_image, y)
            total_train_loss += loss
            train_tqdm.set_postfix(loss=loss)
        return total_train_loss
    
    def train_step(self, image, ref_image, y):
        image, ref_image, y = (
            image.to(self.device), 
            ref_image.to(self.device),
            y.to(self.device)
        )

        y_hat = self.model(image, ref_image)
        loss = self.criterion(y_hat, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    @torch.no_grad()
    def val_loop(self):
        self.model.eval()
        total_val_loss = 0
        val_tqdm = tqdm(self.val_dataloader)
        val_tqdm.set_description("Validation")
        for (image, ref_image, y) in val_tqdm:
            val_loss = self.val_step(image, ref_image, y)
            total_val_loss += val_loss
            val_tqdm.set_postfix(val_loss=val_loss)
        return total_val_loss
    
    @torch.no_grad()
    def val_step(self, image, ref_image, y):
        image, ref_image, y = (
            image.to(self.device), 
            ref_image.to(self.device),
            y.to(self.device)
        )

        y_hat = self.model(image, ref_image)
        val_loss = self.criterion(y_hat, y)
        return val_loss.item()
    
    def save_checkpoint(self, val_loss: float):
        model_dict = {
            "model_state": self.model.state_dict(),
            "opt_state": self.optimizer.state_dict()
        }
        ckpt_path = Configs.generate_checkpoint_dir(self.config, val_loss)
        torch.save(model_dict, self.config_dir.parent / ckpt_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()

    config_dir = pathlib.Path(args.config)
    config = Configs.load_config(config_dir)

    train_dataset, val_dataset, test_dataset = Configs.load_dataset(config)

    train_dataloader = Configs.load_dataloader(config, train_dataset)
    val_dataloader = Configs.load_dataloader(config, val_dataset)

    model = Configs.load_model(config=config)

    criterions = {
        "l1": torch.nn.L1Loss,
        "scnn": scnn_loss.SCNNLoss
    }
    criterion = Configs.load_criterion(config, criterions)
    optimizer = Configs.load_optimizer(config, model)

    train_steps = len(train_dataloader)
    val_steps = len(val_dataloader)

    trainer = Trainer(
        train_steps=train_steps,
        val_steps=val_steps,
        epochs=config.get("epochs", 50),
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        config_dir=config_dir,
        device=torch.device(0)
    )

    trainer.train()