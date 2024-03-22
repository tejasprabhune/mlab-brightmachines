# ML@B x Bright Machines

To set up the module, run
```
python install -e .
```

## Training (work-in-progress)

To train a Siamese CNN, create/use a config file similar to 
`ckpts/scnn_l1/scnn_l1.yml`. Then run the following:

```
python bin/train.py -c ../ckpts/scnn_l1/scnn_l1.yml
```

## Code Walkthrough

`bin/train.py` provides a wrapper `Trainer` class that enables reproducible,
configurable model training using config `.yml` files. For now, the main
training functionality consists of a normal `torch` training loop. A checkpoint
is saved after every epoch based on the configuration file.

`utils/configs.py` contains a static `Configs` class that enables all
configuration, including loading a config, model (new or pretrained), 
criterion, optimizer, dataset, and dataloader. The corresponding functions
are all commented in the file and examples of use are in `bin/train.py` within
the `if __init__ == "__main__"` section.

Models are located in `models/`. Currently, the baseline Res-Net 
has not been ported to the configurable `Trainer` implementation, but the model
itself is at `models/resnet.py` and the training file is in `./train_resnet.py`
(effectively deprecated code). Similarly, the Siamese CNN model code is
located at `models/scnn.py` and both config-based and legacy training code is
available at `bin/train.py` and `./train_scnn.py`, respectively.

The dataset in use for these models is the `ImagePositionDataset` in
`datasets/image_position_dataset.py`. It provides two images, one at the 
current index and one random reference image with the transformation
$T$ between them.
