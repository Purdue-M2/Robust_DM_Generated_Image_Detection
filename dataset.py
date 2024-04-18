# File: dataset.py:
import random
import torch
from torch.utils.data import Dataset
# import torch
import torch.nn as nn
import h5py
import numpy as np
import os



class DFADDataset(Dataset):
    def __init__(self, folder):
        super(DFADDataset, self).__init__()
        self.folder = 'clip_' + folder  # Adjusted to match your folder structure
        self.data = []
        self.labels = []
        self.text_data = []

        for doc in range(5300):
            file_path = os.path.join(self.folder, f'{doc:04d}.h5')
            if os.path.exists(file_path):
                with h5py.File(file_path, 'r') as fr:
                    real_img = fr['real'][:]
                    image_gen = [fr[f'image_gen{i}'][:] for i in range(4)]
                    original_prompt = fr['original_prompt'][:]
                    positive_prompt = fr['positive_prompt'][:]

                    # Combine all image data
                    combined_images = np.vstack([real_img] + image_gen)
                    self.data.append(combined_images)

                    # Use original prompts for real images and positive prompts for generated images
                    combined_prompts = np.vstack([original_prompt] + [positive_prompt for _ in range(4)])
                    self.text_data.append(combined_prompts)

                    # Labels: 0 for real images, 1 for generated images
                    labels = np.zeros(len(real_img))  # Real images
                    generated_labels = np.ones(sum(len(gen) for gen in image_gen))  # Generated images
                    self.labels.append(np.concatenate([labels, generated_labels]))

        if len(self.data) > 0:
            self.data = np.vstack(self.data)
            self.text_data = np.vstack(self.text_data)
            self.labels = np.concatenate(self.labels)
        else:
            raise ValueError("No data loaded.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_features = torch.tensor(self.data[idx], dtype=torch.float32)
        text_features = torch.tensor(self.text_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img_features, text_features, label


