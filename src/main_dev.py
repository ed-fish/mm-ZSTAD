import os
import re
import numpy as np
from collections import defaultdict
import csv
import torch
from models.base import BaseCLIPModel
from utils import get_thumos_dataloader
from tqdm import tqdm

# Configuration
ROOT_DIR = 'data/thumos14'
SPLIT = "val"  # 'train', 'val', or 'test'
BATCH_SIZE = 1
NUM_WORKERS = 8
IOU_THRESHOLD = 0.5
WINDOW_SIZE = 16
STRIDE = 8
THRESHOLD = 28
DOWNSAMPLE_FACTOR = 8

# Check if class labels file exists
CLASS_LABELS_FILE = os.path.join(ROOT_DIR, 'classes.txt')
if not os.path.exists(CLASS_LABELS_FILE):
    raise FileNotFoundError(f"Class labels file {CLASS_LABELS_FILE} not found")

# Prepare the dataset and create the data loader
dataloader = get_thumos_dataloader(ROOT_DIR, split=SPLIT, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, downsample=DOWNSAMPLE_FACTOR)

# Instantiate the CLIP model
DEVICE = "cuda:0"
clip_model = BaseCLIPModel(device=DEVICE)

def split_camel_case(s):
    return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', s)

def prepare_labels(class_file):
    with open(class_file, 'r') as f:
        class_labels = [line.strip() for line in f.readlines()]

    return {int(label.split()[0]): f"a video of {split_camel_case(' '.join(label.split()[1:]))}" for label in class_labels}

def extract_clip_features(video_frames, clip_model):
    batch_size, num_frames, channels, height, width = video_frames.shape
    video_frames_reshaped = video_frames.reshape(batch_size * num_frames, channels, height, width)
    video_features_reshaped = clip_model.get_image_embedding(images={"pixel_values": video_frames_reshaped})
    video_features = video_features_reshaped.reshape(batch_size, num_frames, -1)
    return video_features

def detect_temporal_actions(video_features, text_embeddings, window_size, stride, threshold, downsample_factor, original_fps):
    action_proposals = []
    for i in range(0, video_features.size(0) - window_size, stride):
        window_features = video_features[i:i + window_size]
        window_score = torch.mean(window_features, dim=0)

        # Calculate similarity between window features and text embeddings
        similarities = torch.matmul(window_score, text_embeddings.T)
        max_similarity = torch.max(similarities)

        if max_similarity > threshold:
            # Compute the timestamps using the original video fps and the downsampling factor
            start_timestamp = i * downsample_factor / original_fps
            end_timestamp = (i + window_size) * downsample_factor / original_fps
            action_proposals.append((start_timestamp, end_timestamp, torch.argmax(similarities).item(), max_similarity))

    return action_proposals

def format_proposals(proposals, video_name, class_file):
    formatted_proposals = []
    for proposal in proposals:
        start_time, end_time, class_index, score = proposal
        formatted_proposals.append({
            'video-name': video_name,
            't-start': start_time,
            't-end': end_time,
            'label': class_file[class_index],
            'score': score.item(),
        })
    return formatted_proposals

class_file = prepare_labels(CLASS_LABELS_FILE)
prompts = list(class_file.values())
classes = list(class_file.keys())

# Get text embeddings
text_embeddings = clip_model.get_text_embedding(prompts)

with open('proposals.csv', 'w', newline='') as csvfile:
    fieldnames = ['video-name', 't-start', 't-end', 'label', 'score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for batch_idx, (video_path, video_frames, _, _) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing batches"):
        video_frames = video_frames.to(DEVICE)
        video_features = extract_clip_features(video_frames, clip_model)  # Extract features for the whole batch

        for i, video_path_item in enumerate(video_path):
            action_proposals = detect_temporal_actions(video_features[i], text_embeddings, WINDOW_SIZE, STRIDE, THRESHOLD, DOWNSAMPLE_FACTOR, 30)
            video_name = os.path.basename(video_path_item)
            formatted_proposals = format_proposals(action_proposals, video_name, classes)

            for proposal in formatted_proposals:
                writer.writerow(proposal)
