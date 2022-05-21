import numpy as np
import pandas as pd
import pydicom
import h5py
import cv2

from pathlib import Path

DATASET_ROOT = Path('/mnt/d/siim-covid19-detection')
images = h5py.File(DATASET_ROOT / 'train.hdf5', 'w')

study_label = pd.read_csv((DATASET_ROOT / 'train_study_level.csv'))

def extract(study_id):
    ds = pydicom.dcmread(
        next(iter((DATASET_ROOT / 'train' / study_id[:-6]).rglob('*.dcm'))))
    images[study_id[:-6]] = cv2.resize((ds.pixel_array / 16).astype('uint8'), (500,500))
    return ds.PatientSex

study_label['Sex'] = study_label['id'].apply(extract)
study_label.to_csv('study_label.csv')

images.close()
