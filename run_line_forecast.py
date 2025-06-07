import os

from hydra import compose, initialize
from omegaconf import OmegaConf

from train import train


def main():
    """Train S4 on the synthetic line forecasting dataset."""
    overrides = [
        "pipeline=informer",
        "model=s4",
        "dataset=line",
        "dataset.pred_len=1",
        "loader.batch_size=32",
        "trainer.max_epochs=10",
        "wandb=null",  # disable wandb logging by default
    ]
    with initialize(config_path="configs"):
        cfg = compose(config_name="config.yaml", overrides=overrides)
        print(OmegaConf.to_yaml(cfg))
        train(cfg)


if __name__ == "__main__":
    main()
