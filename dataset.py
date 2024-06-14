import torch
from torch.utils.data.dataset import Dataset


class BCIDataset(Dataset):
    """
    There are two types of BCI datasets:

    1. P300 Stimulus Dataset
        The shapes of the dataset tensors are as follows.
            responses: (batch, channel, sample)
            labels: (batch,)
        The classes are as follows.
            0: P300 signal
            1: non P300 signal

    2. Character Dataset
        The shapes of the dataset tensors are as follows.
            responses: (batch, row_column, epoch, channel, sample)
            labels: (batch,)
        The classes are as follows.
            0-25: A-Z
            26-34: 1-9
            35： _

    The P300 Stimulus Dataset is used to train the stimulus classifier,
    while the Character Dataset is used to evaluate the final character
    classification results.
    """
    def __init__(self, responses: torch.Tensor, labels: torch.Tensor) -> None:
        self.responses = responses
        self.labels = labels
        assert self.responses.size(0) == self.labels.size(0)

    def __getitem__(self, index):
        return self.responses[index, ...], self.labels[index]

    def __len__(self):
        return self.responses.size(0)


__all__ = ['BCIDataset']
