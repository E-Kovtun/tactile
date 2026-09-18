import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.trace_func = trace_func
        self.save_checkpoint_flag = False

    def __call__(self, val_loss):
        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.save_checkpoint_flag = True
            self.update_val_state(val_loss)
        elif val_loss < self.best_val_loss - self.delta:
            # Significant improvement detected
            self.best_val_loss = val_loss
            self.save_checkpoint_flag = True
            self.update_val_state(val_loss)
            self.counter = 0  # Reset counter since improvement occurred
        else:
            # No significant improvement
            self.counter += 1
            self.save_checkpoint_flag = False
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def update_val_state(self, val_loss):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
        self.val_loss_min = val_loss

    def state_dict(self):
        """Return the patience state needed for an exact training resume."""
        def scalar(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return float(value.detach().cpu().item())
            return float(value)

        return {
            "counter": int(self.counter),
            "best_val_loss": scalar(self.best_val_loss),
            "early_stop": bool(self.early_stop),
            "val_loss_min": scalar(self.val_loss_min),
            "save_checkpoint_flag": bool(self.save_checkpoint_flag),
        }

    def load_state_dict(self, state):
        """Restore patience without changing the configured patience/delta."""
        self.counter = int(state.get("counter", 0))
        self.best_val_loss = state.get("best_val_loss")
        self.early_stop = bool(state.get("early_stop", False))
        self.val_loss_min = float(state.get("val_loss_min", np.inf))
        self.save_checkpoint_flag = bool(state.get("save_checkpoint_flag", False))
