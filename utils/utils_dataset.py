import torch
import pandas as pd
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
from torchvision.transforms import transforms
from PIL import Image


class IaT_embed_dataset(Dataset):
    def __init__(self, image_data, transform=None, **args):
        self.img_data = image_data

        self.text_csv = args['text']
        self.mode = args['train_test']
        self.transform = transform

    def __len__(self):
        return (self.img_data.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image
        image = self.img_data[idx]
        image = Image.fromarray(image).convert("RGB")

        # get raw text
        findings = self.text_csv['findings'].iloc[idx]
        impression = self.text_csv['impression'].iloc[idx]
        if findings == 'dumb' or type(findings) == float:
            pass
        else:
            impression += findings
        text = impression

        sample = {'image': image, 'raw_text': text}

        if self.transform:
            # for 2 branch contrastive vision model (not useful for CLIP)
            if self.mode == 'train':
                sample['image'] = self.transform[0](sample['image'])

        return sample


class I_T_emb_dataset:

    def __init__(self, image_path, csv_path):
        self.image_path = image_path
        self.csv_path = csv_path

    def get_dataset(self, train_test, T=None):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])

        if train_test == 'train':
            print('Apply Train-stage Transform!')

            Transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop(224),
                transforms.RandomRotation(degrees=(0, 90)),
                transforms.RandomGrayscale(p=0.5),
                transforms.RandomPerspective(distortion_scale=0.5,
                                                p=0.5,
                                                interpolation=3),
                transforms.RandomAffine(degrees=0,
                                        translate=(0.1, 0.1),
                                        scale=(0.9, 1.1),
                                        shear=10,
                                        resample=False,
                                        fillcolor=0),
                transforms.RandomAutocontrast(p=0.5),
                normalize
            ])
        else:
            print('No test stage in pretrain!')

        img_path = np.load(
            self.image_path['img_path'], allow_pickle=True, mmap_mode='r')
        csv_path = pd.read_csv(
            self.csv_path['text_path'], low_memory=False)

        misc_args = {'train_test': train_test,
                   'text': csv_path}

        dataset = IaT_embed_dataset(image_data=img_path,
                                       transform=Transforms,
                                       **misc_args)

        return dataset
