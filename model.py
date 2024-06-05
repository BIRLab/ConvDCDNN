import torch
from torch import nn
import math


def conv_output_size(input_size, kernel_size, stride, padding, dilation=1):
    """According to https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html"""
    return math.floor((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


class StimulusClassifier(nn.Module):
    def __init__(self, num_channels, num_samples, fusion_channels, kernel_size, kernel_stride, dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(num_channels, fusion_channels, kernel_size, kernel_stride, 0),
            nn.Dropout(dropout),
            nn.Tanh()
        )
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fusion_channels * conv_output_size(num_samples, kernel_size, kernel_stride, 0), 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.output(x)
        return x


class CharClassifier(nn.Module):
    def __init__(self, stimulus_classifier):
        super().__init__()
        self.stimulus_classifier = stimulus_classifier

    def forward(self, x: torch.Tensor):
        num_batch = x.size(0)
        num_epoch = x.size(2)

        # (batch, row_column, epoch, channel, sample)
        x = x.flatten(0, 2)

        # (batch * row_column * epoch, channel, sample)
        x = torch.softmax(self.stimulus_classifier(x), 1)[:, 1]

        # (batch * row_column * epoch,)
        x = x.reshape((num_batch, 12, num_epoch))
        x = x.mean(2)

        # (batch, 12)
        column_score = x[:, :6].reshape((-1, 1, 6))
        row_score = x[:, 6:].reshape((-1, 6, 1))
        x = torch.bmm(row_score, column_score)

        # (batch, 6, 6)
        x = x.reshape((num_batch, 36))

        # (batch, 36)
        return x
