from collections import defaultdict
import torch

class BatchMetrics:
    def __init__(self):
        # Initialize metrics storage for each batch
        self.totals = defaultdict(float)
        self.minibatch_metrics = defaultdict(float)

    def accumulate_minibatch_metrics(self, **kwargs):
        """
        Accumulate metrics using a dictionary of new values.
        The keys of new_values should correspond to the attributes of this dataclass.
        This method adds the new value to the existing metric. 
        """
        for key, value in kwargs.items():
            self.minibatch_metrics[key] += value

    def reduce_batch_metrics(self, device):
        """
        Reduce the minibatch metrics across all processes.
        """
        # Create a tensor from the minibatch_metrics values in the order of keys
        keys = list(self.minibatch_metrics.keys())
        tensor = torch.tensor([float(self.minibatch_metrics[k]) for k in keys], device=device)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        # Store reduced values for this batch
        self.totals = {key: value for key, value in zip(keys, tensor.tolist())}
        # Reset minibatch metrics
        self.minibatch_metrics.clear()

    def reset_batch(self):
        """Clear all accumulated metrics before starting a new batch."""
        self.totals.clear()
        self.minibatch_metrics.clear()