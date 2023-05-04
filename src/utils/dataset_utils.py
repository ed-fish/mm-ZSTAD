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
        
        # self.video_files = sorted(glob.glob(os.path.join(root_dir, split, ext)))
        self.data = pd.read_csv(f"{root_dir}/{split}/thumos_{split}.csv")
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load video frames and audio
        video_path = self.data.loc[idx, 'video_path']
        label = self.data.loc[idx, 'label']
        if self.split in ["test", "val"]:
            start_action = self.data.loc[idx, 'start_action']
            end_action = self.data.loc[idx, 'end_action']
        else:
            start_action = [0]
            end_action = [0]
        video_frames, audio_waveform, framerate = read_video(video_path, pts_unit='sec', output_format="TCHW")
        # video_frames = video_frames[::8, :, :, :]
        
        
        # # Preprocess video frames
        # if self.transform:
        #     video_frames = torch.stack([self.transform(frame) for frame in video_frames])

        # Preprocess audio
        if audio_waveform.size(1) == 0:
            # Create an empty waveform with desired shape, e.g., (1, 1)
            audio_waveform = torch.zeros(2, 100000)
        # else:
        #     # audio_waveform = self.audio_transform(audio_waveform)
        #     audio_waveform = audio_transforms.SlidingWindow(duration=3, overlap=1)(audio_waveform)
        #     print(audio_waveform.shape)
            

        # Load temporal labels
        start_action = start_action.strip("[]").split()
        end_action = end_action.strip("[]").split()
        start_action = [float(n) for n in start_action]
        end_action = [float(n) for n in end_action]
        start_action = torch.tensor(start_action, dtype=torch.float)
        end_action = torch.tensor(end_action, dtype=torch.float)
        action_classes = torch.tensor(label, dtype=torch.long)
        return video_frames, audio_waveform, (start_action, end_action, action_classes)
    

def get_thumos_dataloader(root_dir, split='train', batch_size=1, num_workers=1):
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    audio_transform = audio_transforms.MelSpectrogram(sample_rate=16000, n_mels=64, n_fft=800) 
    thumos_dataset = ThumosDataset(root_dir, split=split, transform=transform, audio_transform=audio_transform)
    dataloader = torch.utils.data.DataLoader(thumos_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader