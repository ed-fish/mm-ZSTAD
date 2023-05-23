import os
import glob
import torch
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
import torchaudio.transforms as audio_transforms
from torch.utils.data import Dataset
from torchvision.io import read_video
from transformers import CLIPProcessor, CLIPModel
import re


def split_camel_case(s):
    return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', s)

# def prepare_labels(class_file):
#     with open(class_file, 'r') as f:
#         class_labels = [line.strip() for line in f.readlines()]

#     return {int(label.split()[0]): f"a video of {split_camel_case(' '.join(label.split()[1:]))}" for label in class_labels}

class BaseCLIPModel:
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model = self.model.eval()

class ThumosDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, audio_transform=None,  model_name='openai/clip-vit-base-patch32', downsample_factor=16):
        assert split in ['train', 'val', 'test'], "split must be 'train', 'val', or 'test'" 
        self.processor = CLIPProcessor.from_pretrained(model_name) 
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.audio_transform = audio_transform
        
        # self.video_files = sorted(glob.glob(os.path.join(root_dir, split, ext)))
        self.data = pd.read_csv(f"{root_dir}/{split}/thumos_{split}.csv")
        self.downsample = downsample_factor
        
        # class_labels_file = os.path.join(root_dir, 'classes.txt')
        # self.class_file = prepare_labels(class_labels_file)
    

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
        frame_rate = framerate["video_fps"]
        audio_fps = framerate["audio_fps"]
        video_frames = video_frames[::self.downsample, :, :, :]
        
        
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
        
        # frame_rate = frame_rate / self.downsample  # Adjust the frame rate according to the downsampled video frames
        # start_frame_indices = (start_action * frame_rate).round().long()
        # end_frame_indices = (end_action * frame_rate).round().long()
        

        # Get text embeddings
        # prompt = get_text_embedding(self.class_file[label], self.processor)
        video_frames = get_image_embedding(video_frames, self.processor)
        label = torch.tensor(label).long()
        return video_path, video_frames, audio_waveform, (start_action, end_action, label)
    
def collate_fn(batch):
    # Sort the batch in the descending order
    sorted_batch = sorted(batch, key=lambda x: x[1].shape[1], reverse=True)

    # Separate video frames, audio, and ground truths
    video_path, sequences, audios, ground_truths = zip(*sorted_batch)

    # Get sequence lengths
    lengths = [len(seq) for seq in sequences]
    # prompts = [p.squeeze() for p in prompts]

    # Padding
    padded_sequences = torch.nn.utils.rnn.pad_sequence([seq for seq in sequences], batch_first=True)
    # padded_audios = pad_audio(audios)

    # Unzip the ground truths
    start_frame_indices, end_frame_indices, action_classes = zip(*ground_truths)
    
    start_frame_indices = torch.nn.utils.rnn.pad_sequence(start_frame_indices, batch_first=True, padding_value=0)
    end_frame_indices = torch.nn.utils.rnn.pad_sequence(end_frame_indices, batch_first=True, padding_value=0)
    # max_length = max([prompt.size(1) for prompt in prompts])
    # prompts = torch.nn.utils.rnn.pad_sequence([torch.cat([prompt, torch.zeros((1, max_length - prompt.size(1)))], dim=1) for prompt in prompts], batch_first=True)
    # prompts = prompts.long()
    action_classes = torch.stack(action_classes)

    # Convert to tensors
    # start_frame_indices = torch.stack(start_frame_indices)
    # end_frame_indices = torch.stack(end_frame_indices)
    # action_classes = torch.stack(action_classes)

    return video_path, padded_sequences, None, (start_frame_indices, end_frame_indices, action_classes)


def get_thumos_dataloader(root_dir, split='train', batch_size=1, num_workers=1, downsample=16):
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    audio_transform = audio_transforms.MelSpectrogram(sample_rate=16000, n_mels=64, n_fft=800) 
    thumos_dataset = ThumosDataset(root_dir, split=split, transform=transform, audio_transform=audio_transform, downsample_factor=downsample)
    dataloader = torch.utils.data.DataLoader(thumos_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader
    
def get_image_embedding(images, clippy, use_tensor=True):
    inputs = clippy(images=images, return_tensors="pt", padding=True)
    return inputs['pixel_values']

def get_text_embedding(texts, clippy, use_tensor=True):
    inputs = clippy(text=texts, return_tensors="pt", padding=True)
    return inputs['input_ids']

