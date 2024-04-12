import torch
import torch.nn
import torch.nn.functional as F

from pick_place.utils.configs import Configs

config = Configs.load_config("../ckpts/scnn_l1_3_crop/scnn_l1_3.yml")

model = Configs.load_model(config)

train_dataset, val_dataset, test_dataset = Configs.load_dataset(config)

train_dataloader = Configs.load_dataloader(config, train_dataset)
val_dataloader = Configs.load_dataloader(config, val_dataset)

model_input = next(iter(train_dataloader))[0][0].unsqueeze(0)

onnx_program = torch.onnx.export(model, model_input, "../ckpts/scnn_l1_3/scnn_l1_3_c.onnx")
print(onnx_program)