import numpy as np
import pandas as pd
import pydicom
import h5py

from pathlib import Path

DATASET_ROOT = Path('/mnt/d/siim-covid19-detection')
images = h5py.File(DATASET_ROOT / 'train.hdf5', 'w')

study_label = pd.read_csv((DATASET_ROOT / 'train_study_level.csv'))

def extract(study_id):
    ds = pydicom.dcmread(
        next(iter((DATASET_ROOT / 'train' / study_id[:-6]).rglob('*.dcm'))))
    images[study_id[:-6]] = (ds.pixel_array[::2, ::2] * 0.0625).astype('uint8')
    return ds.PatientSex

study_label['Sex'] = study_label['id'].apply(extract)
study_label.to_csv('study_label.csv')

images.close()
