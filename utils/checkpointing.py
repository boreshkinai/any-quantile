import os
from glob import glob

def get_checkpoint_path(cfg):
    if cfg.checkpoint.resume_ckpt is None:
        return None
    if os.path.isfile(cfg.checkpoint.resume_ckpt):
        return cfg.checkpoint.resume_ckpt 
    if cfg.checkpoint.resume_ckpt == 'last':
        job_dir = os.path.join(cfg.logging.path, cfg.logging.name)
        return get_latest_checkpoint(os.path.join(job_dir, 'checkpoints/*.ckpt'))
    else:
        assert False, f"Checkpoint {cfg.checkpoint.resume_ckpt} is not found, supported options: null, last, valid file path" 

def get_latest_checkpoint(checkpoint_dir: str = None) -> str:
    if checkpoint_dir is None:
        return None
    checkpoints = glob(checkpoint_dir)

    checkpoint_epochs = {c.split("epoch=")[-1].split(".")[0]: c for c in checkpoints}
    checkpoint_epochs = {int(c): dir for c, dir in checkpoint_epochs.items() if c.isdigit()}
    if len(checkpoint_epochs) == 0:
        return None

    max_epoch = max(checkpoint_epochs.keys())
    if os.path.isfile(checkpoint_epochs[max_epoch]):
        return checkpoint_epochs[max_epoch]
    else:
        return None
    