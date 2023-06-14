import os
from skimage import io, transform
from PIL import Image
import numpy as np
import pandas as pd
import argparse
import skimage
from tqdm import tqdm


def get_MIMIC_img(subject_id, study_id, dicom):
    path = 'xx' # meta MIMIC path
    report_path = 'xx' # report MIMIC path

    sub_dir = 'p' + subject_id[0:2] + '/' + 'p' + subject_id + '/' + 's' + study_id + '/' + dicom + '.jpg'
    report_sub_dir = 'p' + subject_id[0:2] + '/' + 'p' + subject_id + '/' + 's' + study_id + '.txt'
    jpg_path = path + sub_dir
    report_path = report_path + report_sub_dir

    img = Image.open(jpg_path)
    img = np.array(img)
    return img

parser = argparse.ArgumentParser(description='extract_data')
parser.add_argument('--resize', type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    resize = args.resize
        
    metacsv = pd.read_csv('xx') # master csv from MGCA preprocessing stage

    temp_npy = np.zeros((metacsv.shape[0], resize, resize), dtype=np.uint8)
    print(metacsv.shape, temp_npy.shape)

    for i in tqdm(range(temp_npy.shape[0])):
        dicom_idx = metacsv['dicom_id'][i]
        subject_idx = str(int(metacsv['subject_id'][i]))
        study_idx = str(int(metacsv['study_id'][i]))
        
        img = get_MIMIC_img(subject_id=subject_idx, study_id=study_idx, dicom=dicom_idx)
        x, y = np.nonzero(img)
        xl,xr = x.min(),x.max()
        yl,yr = y.min(),y.max()
        img = img[xl:xr+1, yl:yr+1]
        img = ((img - img.min()) * (1/(img.max() - img.min()) * 256))

        img = skimage.transform.resize(img, (resize, resize), 
        order=1, preserve_range=True, anti_aliasing=False)
        img = img.astype(np.uint8)

        temp_npy[i,:,:] = img

        np.save(f'xx', temp_npy) # save to ext_data folder