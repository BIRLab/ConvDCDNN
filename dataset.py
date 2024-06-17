import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import h5py
from tqdm import tqdm


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

    def to(self, device):
        self.responses = self.responses.to(device)
        self.labels = self.labels.to(device)


def extract_responses(dataset: h5py.File, subject: str, train_test: str, window_size: int) -> tuple[np.ndarray, ...]:
    """
    Extract all responses from dataset item.

    :param dataset: i.e. dataset["bci_ii"]["train"]
    :param window_size: samples from stimulus begin
    :return: responses (character, row_column, epoch, channel, sample), labels
    """
    stimulus = dataset[subject][train_test]["stimulus"]
    signal = dataset[subject][train_test]["signal"]
    labels = dataset[subject][train_test]["label"][:]

    responses = []
    stimulus_labels = []
    for epoch in tqdm(range(len(labels)), desc=f'extract {subject}'):
        response = np.zeros((12, 15, 64, window_size))
        response_counter = np.zeros(12, dtype=int)
        stimulus_finished_indices = np.argwhere(np.diff(stimulus[0]) < 0).flatten()
        for stimulus_finished_index in stimulus_finished_indices:
            row_column = stimulus[epoch][stimulus_finished_index - 1] - 1
            response[row_column, response_counter[row_column], :, :] = signal[epoch][stimulus_finished_index - 23:stimulus_finished_index + window_size - 23, :].T
            response_counter[row_column] += 1
        target_row, target_column = labels[epoch] // 6 + 7, labels[epoch] % 6 + 1
        target_indices = np.arange(12)
        target_indices = np.logical_or(target_indices == target_row - 1, target_indices == target_column - 1)
        label = np.zeros((12, 15), dtype=int)
        label[target_indices, :] = 1
        responses.append(response)
        stimulus_labels.append(label)
    responses = np.stack(responses, 0)
    stimulus_labels = np.stack(stimulus_labels, 0)
    return responses, stimulus_labels, labels


__all__ = ['BCIDataset', 'extract_responses']
