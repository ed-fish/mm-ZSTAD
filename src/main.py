import os
import numpy as np
from PIL import Image
import soundfile as sf
import torch
from models.base import BaseCLAPModel, BaseCLIPModel
from utils import ThumosDataset, get_thumos_dataloader

import re

def split_camel_case(s):
    return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', s)


def cosine_similarity(a, b):
    a_tensor = torch.tensor(a).to(b.device)
    b_tensor = torch.tensor(b).to(b.device)
    return torch.nn.functional.cosine_similarity(a_tensor, b_tensor, dim=0)

clip_model = BaseCLIPModel()
clap_model = BaseCLAPModel()

root_dir = '/home/ed/mm-ZSTAD/data/thumos14'
split = 'val'
train_loader = get_thumos_dataloader(root_dir, split=split, batch_size=1)
video_frames, audio_waveforms, (start, end, labels) = next(iter(train_loader))

# Load class labels and create text prompts
class_labels_file = os.path.join(root_dir, 'classes.txt')
with open(class_labels_file, 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]

class_label_dict = {int(label.split()[0]): f"a video of the action {split_camel_case(' '.join(label.split()[1:]))}" for label in class_labels}

label = labels[-1].item()
text_prompt = class_label_dict[label]

text_embedding_clip = clip_model.get_text_embedding([text_prompt], use_tensor=True)[0]
text_embedding_clap = clap_model.get_text_embedding([text_prompt, ""], use_tensor=True)[0]
frame_features = clip_model.get_image_embedding(video_frames, use_tensor=True)
audio_features = clap_model.get_audio_embedding_from_data(audio_waveforms, use_tensor=True)

frame_similarities = [cosine_similarity(text_embedding_clip, frame_feature) for frame_feature in frame_features]
audio_similarities = [cosine_similarity(text_embedding_clap, audio_feature) for audio_feature in audio_features]

max_frame_similarity_idx = np.argmax(frame_similarities)
min_frame = np.argmin(frame_similarities)
# max_audio_similarity_idx = torch.argmax(audio_similarities)

most_similar_frame = video_frames[0][max_frame_similarity_idx].numpy().transpose(1,2,0)

least_similar_frame = video_frames[0][min_frame].numpy().transpose(1,2,0)
# most_similar_audio = audio_waveforms[max_audio_similarity_idx].numpy()

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Get the original class label (before processing)
original_class_label = class_label_dict[label]

frame_img = Image.fromarray((most_similar_frame).astype(np.uint8))
frame_img.save(os.path.join(output_dir, f'{original_class_label}_most_similar_frame.png'))

frame_img = Image.fromarray((least_similar_frame).astype(np.uint8))
frame_img.save(os.path.join(output_dir, f'{original_class_label}_least_similar_frame.png'))

# audio_waveforms = audio_waveforms.squeeze(0).transpose(1, 0)

# sf.write(os.path.join(output_dir, f'audio.wav'), audio_waveforms.numpy(), samplerate=16000)
