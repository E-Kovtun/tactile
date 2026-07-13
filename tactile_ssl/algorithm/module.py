from typing import Dict, Any, Tuple, Optional, Union
from abc import ABC, abstractmethod
import torch


class Module(ABC):
    @abstractmethod
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(self, num_iterations_per_epoch: int, num_epochs: int) -> Tuple[
        torch.optim.Optimizer,
        Optional[Dict],
        Optional[Dict],
    ]:
        raise NotImplementedError

    def on_fit_start(self, train_dataloader, val_dataloader, trainer_instance=None):
        pass

    def get_checkpoint_state(self) -> Dict[str, Any]:
        """Return algorithm state that is not part of ``nn.Module.state_dict``."""
        return {}

    def load_checkpoint_state(
        self,
        state: Optional[Dict[str, Any]],
        global_step: int,
        current_epoch: int,
    ) -> None:
        """Restore non-module state after model and optimizer loading."""
        pass

    def on_train_epoch_end(self, trainer_instance=None):
        pass

    def on_validation_epoch_end(self, trainer_instance=None):
        pass

    def on_train_batch_end(self, outputs: Dict, batch: Dict, batch_idx: int, trainer_instance=None):
        pass

    def on_validation_batch_end(self, outputs: Dict, batch: Dict, batch_idx: int, trainer_instance=None):
        pass

    def on_train_batch_start(self, batch: Dict, batch_idx: int):
        pass

    def on_train_epoch_start(self, trainer_instance=None):
        pass
