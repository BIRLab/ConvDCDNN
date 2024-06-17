import argparse
import h5py
import torch
from data import dataset_path
from dataset import BCIDataset, extract_responses
from model import StimulusClassifier, CharClassifier
from adcl import ADCL
from preprocess import bandpass, z_score
from focal_loss import focal_loss
from scipy.signal import resample
from early_stopping import EarlyStopping
from pathlib import Path
from metrics import mean_itr
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, Mean
import random
import numpy as np


def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def set_random_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def new_exp_name(exp_dir: Path):
    """generate a new exp name for tensorboard"""
    if not exp_dir.exists():
        exp_id = 1
    else:
        exp_list = [int(i.name[3:]) for i in exp_dir.glob('exp*')]
        if exp_list:
            exp_id = max(exp_list) + 1
        else:
            exp_id = 1
    return f'exp{exp_id}'


def argparse_to_dict(opt: argparse.Namespace):
    d = {}
    for k in dir(opt):
        if not k.startswith('_'):
            v = getattr(opt, k)
            if isinstance(v, Path):
                d[k] = v.as_posix()
            elif isinstance(v, list):
                d[k] = ', '.join(v)
            else:
                d[k] = v
    return d


def parse_opt():
    parser = argparse.ArgumentParser()

    # name
    parser.add_argument('--algorithm', type=str, default='ConvDCDNN', help='algorithm name')

    # hardware
    parser.add_argument('--device', type=str, default=detect_device(), help='torch device')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    # log
    parser.add_argument('--log-dir', type=Path, default=Path('./runs'), help='tensorboard output directory')
    parser.add_argument('--exp', type=str, help='experiment name')
    parser.add_argument('--result-path', type=Path, help='result path')

    # dataset
    parser.add_argument('--subject', type=str, choices=['bci_ii', 'bci_iii_a', 'bci_iii_b'], required=True, help='subject of the dataset')
    parser.add_argument('--selected-channel', nargs='+', type=str, help='selected channel name')
    parser.add_argument('--window-size', type=int, default=240, help='response window size')
    parser.add_argument('--low-frequency', type=float, default=0.1, help='cheby1 bandpass filter low frequency')
    parser.add_argument('--high-frequency', type=float, default=20, help='cheby1 bandpass filter high frequency')
    parser.add_argument('--resampling', type=int, help='resampling number')
    parser.add_argument('--validation', type=float, default=0.1, help='validation dataset ratio')

    # model
    parser.add_argument('--model-path', type=Path, help='pretrain model path')
    parser.add_argument('--fusion-channels', type=int, default=32, help='number of conv1 layer out channel')
    parser.add_argument('--kernel-size', type=int, default=15, help='number of conv2 layer kernel size')
    parser.add_argument('--kernel-stride', type=int, default=15, help='number of conv2 layer stride')
    parser.add_argument('--dropout', type=float, default=0.25, help='dropout rate of mlp layers')

    # train
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--max-epoch', type=int, default=2000, help='max epoch')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')

    # parse arguments
    opt = parser.parse_args()
    if opt.exp is None:
        opt.exp = new_exp_name(opt.log_dir)

    if opt.result_path is None:
        opt.result_path = Path('result') / opt.subject

    return opt


def create_dataset(opt) -> tuple[BCIDataset, BCIDataset, BCIDataset, BCIDataset]:
    # extract dataset
    with h5py.File(dataset_path, 'r') as f:
        train_responses, train_stimulus_labels, train_char_labels = extract_responses(f, opt.subject, 'train', opt.window_size)
        test_responses, test_stimulus_labels, test_char_labels = extract_responses(f, opt.subject, 'test', opt.window_size)
        channels = f.attrs['channels']
        sampling_rate = f.attrs['sampling_rate']

    # select channels
    if opt.selected_channel is not None:
        channel_index = [channels.index(i) for i in opt.selected_channel]
        train_responses = train_responses[:, :, :, channel_index, :]
        test_responses = test_responses[:, :, :, channel_index, :]

    # bandpass filtering
    train_responses = bandpass(train_responses, 4, opt.low_frequency, opt.high_frequency, sampling_rate)
    test_responses = bandpass(test_responses, 4, opt.low_frequency, opt.high_frequency, sampling_rate)

    # resample
    if opt.resampling is not None:
        train_responses = resample(train_responses, opt.resampling, axis=4)
        test_responses = resample(test_responses, opt.resampling, axis=4)

    # normalization
    train_responses, train_mean, train_std = z_score(train_responses, (0, 1, 2, 4))
    test_responses = (test_responses - train_mean) / train_std

    # convert to torch tensor
    train_responses = torch.from_numpy(train_responses).type(torch.float32)
    train_stimulus_labels = torch.from_numpy(train_stimulus_labels).type(torch.int64).flatten()
    train_char_labels = torch.from_numpy(train_char_labels).type(torch.int64)
    test_responses = torch.from_numpy(test_responses).type(torch.float32)
    test_stimulus_labels = torch.from_numpy(test_stimulus_labels).type(torch.int64).flatten()
    test_char_labels = torch.from_numpy(test_char_labels).type(torch.int64)

    # split validation dataset
    train_indices = []
    validation_indices = []
    for i in range(2):
        class_indices = torch.argwhere(train_stimulus_labels == i).flatten()
        class_indices = class_indices[torch.randperm(class_indices.size(0))]
        validation_samples_number = round(class_indices.size(0) * opt.validation)
        train_indices.append(class_indices[validation_samples_number:])
        validation_indices.append(class_indices[:validation_samples_number])
    train_indices = torch.hstack(train_indices)
    validation_indices = torch.hstack(validation_indices)

    # create torch datasets
    train_stimulus_dataset = BCIDataset(train_responses.flatten(0, 2)[train_indices, ...], train_stimulus_labels[train_indices])
    validation_stimulus_dataset = BCIDataset(train_responses.flatten(0, 2)[validation_indices, ...], train_stimulus_labels[validation_indices])
    test_stimulus_dataset = BCIDataset(test_responses.flatten(0, 2), test_stimulus_labels.flatten())
    test_char_dataset = BCIDataset(test_responses, test_char_labels)

    # to device
    train_stimulus_dataset.to(opt.device)
    validation_stimulus_dataset.to(opt.device)
    test_stimulus_dataset.to(opt.device)
    test_char_dataset.to(opt.device)

    return train_stimulus_dataset, validation_stimulus_dataset, test_stimulus_dataset, test_char_dataset


def train_stimulus_classifier(opt: argparse.Namespace, model: StimulusClassifier, train_dataset: BCIDataset, validation_dataset: BCIDataset, writer: SummaryWriter):
    # dataloader
    train_loader = DataLoader(train_dataset, opt.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, opt.batch_size)

    # optimizer
    optimizer = ADCL([
        {'params': model.get_parameter('conv.0.weight'), 'lr': 1e-2},
        {'params': model.get_parameter('conv.0.bias'), 'lr': 1e-5},
        {'params': model.get_parameter('output.1.weight'), 'lr': 1e-3},
        {'params': model.get_parameter('output.1.bias'), 'lr': 1e-6}
    ])

    # loss function
    loss_function = focal_loss(alpha=[0.2, 1.0], gamma=2, device=opt.device)

    # early stopping
    early_stopping = EarlyStopping(tolerance=opt.patience)

    # training loop
    best_loss = float('inf')
    best_f1 = 0.0
    for epoch in range(opt.max_epoch):
        # train model
        model.train()
        with tqdm(train_loader, desc=f'Epoch {epoch}') as pbar:
            for step, data in enumerate(pbar):
                x, y = data

                def closure():
                    optimizer.zero_grad()
                    pred = model(x)
                    if torch.isinf(pred).any():
                        raise ValueError('Training loss is diverging, please decrease the learning rate.')
                    _loss = loss_function(pred, y)
                    _loss.backward()
                    return _loss

                loss = optimizer.step(closure)
                writer.add_scalar('train/loss', loss, len(train_loader) * epoch + step)
                pbar.set_postfix({'loss': loss.item()})

        # evaluate model
        model.eval()
        loss = Mean(device=opt.device)
        accuracy = MulticlassAccuracy(num_classes=2, device=opt.device)
        f1 = MulticlassF1Score(average=None, num_classes=2, device=opt.device)
        with torch.no_grad():
            for x, y in validation_loader:
                pred = model(x)
                loss.update(loss_function(pred, y))
                accuracy.update(pred, y)
                f1.update(pred, y)
        test_loss = loss.compute()
        test_accuracy = accuracy.compute()
        test_f1 = f1.compute()[1]
        writer.add_scalar('validation/loss', test_loss, len(train_loader) * (epoch + 1))
        writer.add_scalar('validation/accuracy', test_accuracy, len(train_loader) * (epoch + 1))
        writer.add_scalar('validation/f1', test_f1, len(train_loader) * (epoch + 1))

        # save best weight
        if test_loss < best_loss:
            torch.save(model.state_dict(), opt.result_path / 'best_loss.pth')
            best_loss = test_loss
        if test_f1 > best_f1:
            torch.save(model.state_dict(), opt.result_path / 'best_f1.pth')
            best_f1 = test_f1

        # check early stopping
        if early_stopping(1 - best_f1):
            break


def test_char(test_dataset: BCIDataset, char_model: CharClassifier, writer: SummaryWriter):
    # test loader
    test_loader = DataLoader(test_dataset, 1)
    char_model.eval()

    # test model
    accuracy_list = []
    with torch.no_grad():
        for i in range(1, 16):
            correct = 0
            with tqdm(test_loader, desc=f'Test {i}') as pbar:
                for x, y in pbar:
                    pred = char_model(x[:, :, :i, :, :])
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            accuracy = correct / len(test_loader)
            accuracy_list.append(accuracy)
            writer.add_scalar('result/accuracy', accuracy, i)

    return accuracy_list


def test_stimulus(opt: argparse.Namespace, test_dataset: BCIDataset, char_model: CharClassifier, writer: SummaryWriter):
    # test loader
    test_loader = DataLoader(test_dataset, opt.batch_size)
    model = char_model.stimulus_classifier
    model.eval()

    # test model
    accuracy = MulticlassAccuracy(num_classes=2, device=opt.device)
    f1 = MulticlassF1Score(average=None, num_classes=2, device=opt.device)
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x)
            accuracy.update(pred, y)
            f1.update(pred, y)
    test_accuracy = accuracy.compute()
    test_f1 = f1.compute()[1]
    return test_accuracy, test_f1


def test(opt: argparse.Namespace, test_stimulus_dataset: BCIDataset, test_char_dataset: BCIDataset, char_model: CharClassifier, writer: SummaryWriter) -> dict:
    # test character recognition accuracy
    char_accuracy = test_char(test_char_dataset, char_model, writer)

    # test p300 accuracy
    p300_accuracy, p300_f1 = test_stimulus(opt, test_stimulus_dataset, char_model, writer)

    return {
        'p300_accuracy': p300_accuracy,
        'p300_f1': p300_f1,
        'mean_itr': mean_itr(char_accuracy),
        **{f'epoch{i + 1}': char_accuracy[i] for i in range(15)}
    }


def main(opt=None):
    # parse arguments
    if opt is None:
        opt = parse_opt()

    # create directory to save result
    opt.result_path.mkdir(parents=True, exist_ok=True)

    # fix random seed for reproducibility
    set_random_seed(opt.seed)

    # load bci dataset
    train_stimulus_dataset, validation_stimulus_dataset, test_stimulus_dataset, test_char_dataset = create_dataset(opt)
    num_channels, num_samples = test_char_dataset.responses.shape[-2:]

    # create models
    model_args = (num_channels, num_samples, opt.fusion_channels, opt.kernel_size, opt.kernel_stride, opt.dropout)
    stimulus_model = StimulusClassifier(*model_args)
    if opt.model_path is not None:
        stimulus_model.load_state_dict(torch.load(opt.model_path))
    char_model = CharClassifier(stimulus_model)

    # create log writer
    writer = SummaryWriter(opt.log_dir / opt.exp)

    # move to device
    stimulus_model.to(opt.device)
    char_model.to(opt.device)
    writer.add_graph(char_model, torch.rand((1, 12, 15, num_channels, num_samples)).to(opt.device))

    # train stimulus models
    train_stimulus_classifier(opt, stimulus_model, train_stimulus_dataset, validation_stimulus_dataset, writer)

    # transfer stimulus classifier weights to char classifier
    char_model.stimulus_classifier.load_state_dict(torch.load(opt.result_path / 'best_f1.pth'))

    # test models
    metric = test(opt, test_stimulus_dataset, test_char_dataset, char_model, writer)

    # log hyperparameter and metric to tensorboard
    writer.add_hparams(argparse_to_dict(opt), metric)

    # finished
    writer.close()


if __name__ == '__main__':
    main()
