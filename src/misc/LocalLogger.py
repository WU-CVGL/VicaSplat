import os
from pathlib import Path
from typing import Any, Optional
from torch.utils.tensorboard import SummaryWriter
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_only
from PIL import Image

LOG_PATH = Path("outputs/local")


class LocalLogger(Logger):
    def __init__(self, log_path: Path = LOG_PATH) -> None:
        super().__init__()
        self.experiment = None
        self.log_path = log_path
        os.system(f"rm -r {log_path}")

    @property
    def name(self):
        return "LocalLogger"

    @property
    def version(self):
        return 0

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        pass

    @rank_zero_only
    def log_image(
        self,
        key: str,
        images: list[Any],
        step: Optional[int] = None,
        **kwargs,
    ):
        # The function signature is the same as the wandb logger's, but the step is
        # actually required.
        assert step is not None
        for index, image in enumerate(images):
            path = self.log_path / f"{key}/{index:0>2}_{step:0>6}.png"
            path.parent.mkdir(exist_ok=True, parents=True)
            Image.fromarray(image).save(path)


class LocalTensorboardLogger(TensorBoardLogger):
    
    @rank_zero_only
    def log_image(
        self,
        key: str,
        images: list[Any],
        step: Optional[int] = None,
        dataformats: str = "HWC",
        **kwargs,
    ):
        writer: SummaryWriter = self.experiment
        assert step is not None
        if len(images) > 1:
            for index, image in enumerate(images):
                log_key = f"{key}/{index:0>2}"
                writer.add_image(log_key, image, step, dataformats=dataformats)
        else:
            log_key = key
            writer.add_image(log_key, images[0], step, dataformats=dataformats)