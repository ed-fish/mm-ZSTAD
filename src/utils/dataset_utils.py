import os
import glob
import torch
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
import torchaudio.transforms as audio_transforms
from torch.utils.data import Dataset
from torchvision.io import read_video

class ThumosDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, audio_transform=None):
        assert split in ['train', 'val', 'test'], "split must be 'train', 'val', or 'test'"
        
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.audio_transform = audio_transform
        
        self.video_files = sorted(glob.glob(os.path.join(root_dir, split, '*.avi')))
        self.annotation_files = sorted(glob.glob(os.path.join(root_dir, split, '*.csv')))

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        # Load video frames and audio
        video_frames, audio_waveform, _ = read_video(self.video_files[idx], pts_unit='sec')

        # Preprocess video frames
        if self.transform:
            video_frames = torch.stack([self.transform(Image.fromarray(frame)) for frame in video_frames])

        # Preprocess audio
        if self.audio_transform:
            audio_waveform = self.audio_transform(audio_waveform)

        # Load temporal labels
        temporal_labels = pd.read_csv(self.annotation_files[idx])
        start_times = torch.tensor(temporal_labels['start_time'].values, dtype=torch.float)
        end_times = torch.tensor(temporal_labels['end_time'].values, dtype=torch.float)
        action_classes = torch.tensor(temporal_labels['action_class'].values, dtype=torch.long)

        return video_frames, audio_waveform, (start_times, end_times, action_classes)

def get_thumos_dataloader(root_dir, split='train', batch_size=16, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    audio_transform = audio_transforms.MelSpectrogram(sample_rate=16000, n_mels=128)
    
    thumos_dataset = ThumosDataset(root_dir, split=split, transform=transform, audio_transform=audio_transform)
    dataloader = torch.utils.data.DataLoader(thumos_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader