import os
from glob import glob
from tqdm import tqdm


data_root = os.path.dirname(__file__)
dataset_path = os.path.join(data_root, 'dataset.hdf5')


if not os.path.exists(dataset_path):
    with open(dataset_path, 'wb') as outfile:
        for part in tqdm(
            sorted(glob('dataset.hdf5.part_*', root_dir=data_root)),
            desc='merge dataset'
        ):
            with open(os.path.join(data_root, part), 'rb') as infile:
                outfile.write(infile.read())


__all__ = ['dataset_path']
