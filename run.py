import argparse
import yaml
from omegaconf import OmegaConf
from sklearn.model_selection import ParameterGrid

import os
import random
import logging
import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from utils.checkpointing import get_checkpoint_path
from utils.model_factory import instantiate


def run(cfg_yaml):
    
    cfg = OmegaConf.from_dotlist([f"{k}={v}" for k,v in cfg_yaml.items()])
    # Interpolate fields
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)
    
    print(OmegaConf.to_yaml(cfg))
    
    logger = TensorBoardLogger(save_dir=cfg.logging.path, version=cfg.logging.name, name="")    
    
    dm = instantiate(cfg.dataset)
    
    # setup callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # saves last-K checkpoints based on epoch metric make sure you log it inside your LightningModule
    checkpoint_callback = ModelCheckpoint(save_top_k=cfg.checkpoint.save_top_k, monitor="epoch",
                                          mode="max", filename="model-{epoch}")
    
    pl.seed_everything(cfg.random.seed, workers=True)
    trainer = pl.Trainer(**cfg.trainer, logger=logger,
                         callbacks=[lr_monitor, checkpoint_callback])
    
    model = instantiate(cfg.model, cfg=cfg)
    
    trainer.fit(model, datamodule=dm, ckpt_path=get_checkpoint_path(cfg))
    
    trainer.test(datamodule=dm, ckpt_path=cfg.checkpoint.ckpt_path)


def main(config_path: str, overrides: list = []):
    
    if not torch.cuda.is_available():
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! CUDA is NOT available !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
    with open(config_path) as f:
        cfg_yaml = yaml.unsafe_load(f)
    cfg_yaml = {k: 'null' if v is None  else v for k,v in cfg_yaml.items()}
    # This creates the fields for grid out of the tuple fields in the config
    cfg_yaml = {k: list(v) if isinstance(v, tuple) else [v] for k,v in cfg_yaml.items()}
    
    param_grid = ParameterGrid(cfg_yaml)
    for param_set in param_grid:
        run(param_set)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False, description="Experiment")
    parser.add_argument('--config', type=str, 
                        help='Path to the experiment configuration file', 
                        default='config/config.yaml')
    parser.add_argument("overrides", nargs="*",
                        help="Any key=value arguments to override config values (use dots for.nested=overrides)", )
    args = parser.parse_args()

    main(config_path=args.config, overrides=args.overrides)
    