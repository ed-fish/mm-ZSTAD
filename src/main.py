import torch
import os
import re
import numpy as np
from collections import defaultdict
from models.base import BaseCLIPModel
from utils import get_thumos_dataloader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

# Configuration
root_dir = 'data/thumos14'
split = "val"  # 'train', 'val', or 'test'
batch_size = 2
num_workers = 4
iou_threshold = 0.3
window_size = 16
stride = 8
threshold = 29.0

# Prepare the dataset and create the data loader
dataloader = get_thumos_dataloader(root_dir, split=split, batch_size=batch_size, num_workers=num_workers)

# Instantiate the CLIP model
clip_model = BaseCLIPModel()
device = "cuda:0"

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    clip_model= nn.DataParallel(clip_model)

clip_model.to(device)

if torch.cuda.device_count() > 1:
    clip_model = clip_model.module

    

def extract_clip_features(video_frames, clip_model):
    video_features = clip_model.get_image_embedding(video_frames)
    return video_features.squeeze(1)

def detect_temporal_actions(video_features, text_embeddings, window_size, stride, threshold):
    action_proposals = []
    video_features = video_features.squeeze(0)

    for i in range(0, video_features.size(0) - window_size, stride):
        window_features = video_features[i:i + window_size]
        window_score = torch.mean(window_features, dim=0)

        # Calculate similarity between window features and text embeddings
        similarities = torch.matmul(window_score, text_embeddings.T)
        max_similarity = torch.max(similarities)

        if max_similarity > threshold:
            action_proposals.append((i, i + window_size, torch.argmax(similarities).item()))

    return action_proposals

def calculate_iou(a, b):
    a_start, a_end = a
    b_start, b_end = b
    intersection_start, intersection_end = max(a_start, b_start), min(a_end, b_end)
    intersection = max(0, intersection_end - intersection_start)
    union = (a_end - a_start) + (b_end - b_start) - intersection
    return intersection / union

def evaluate_temporal_actions(batch_action_proposals, batch_ground_truths, iou_threshold, classes):
    batch_results = []

    # Unpack the ground truth data
    start_times, end_times, class_labels = batch_ground_truths

    for action_proposals, starts, ends, class_labels in zip(batch_action_proposals, start_times, end_times, class_labels):
        # Arrange action proposals in ascending order based on start times
        proposals = sorted(action_proposals, key=lambda x: x[0])

        for start, end in zip(starts, ends):
            # Create ground truth list for each instance in the batch
            true_positives, false_positives = 0, 0
            gt_class = class_labels.item()  # get the ground truth class for this instance
            found_true_positive = False

            for proposal in proposals:
                ps, pe, pc = proposal
                pc = classes[pc]
                is_true_positive = False
                breakpoint()

                if pc == gt_class:  # Check if the predicted class matches the ground truth class
                    if calculate_iou((ps, pe), (start.item(), end.item())) > iou_threshold:  # compare with the ground truth start and end times
                        is_true_positive = True
                        break

                if is_true_positive:
                    true_positives += 1
                    found_true_positive = True
                else:
                    false_positives += 1

            precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
            recall = 1 if found_true_positive else 0

            batch_results.append({"precision": precision, "recall": recall})

    return batch_results

# class_labels_file = os.path.join(root_dir, 'classes.txt')
# class_file = prepare_labels(class_labels_file)
# prompts = list(class_file.values())
# classes = list(class_file.keys())

# Get text embeddings
# text_embeddings = clip_model.get_text_embedding(prompts)

class_results = defaultdict(list)

for batch_idx, (video_frames, _, prompts, ground_truths) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing batches"):
    ground_truths = [gt.to(device) for gt in ground_truths]
    video_frames = video_frames.to(device)
    prompts = prompts.to(device)
    
    batch_video_features = extract_clip_features(video_frames, clip_model)  # Extract features for the whole batch
    
    # You might need to adjust the following part further based on how you want to handle the padded parts
    batch_action_proposals = []
    for i in range(batch_size):
        vf = batch_video_features[i]
        action_proposals = detect_temporal_actions(vf.unsqueeze(0), text_embeddings, window_size, stride, threshold)
        batch_action_proposals.append(action_proposals)
        
    batch_evaluation_metrics = evaluate_temporal_actions(batch_action_proposals, ground_truths, iou_threshold, classes)
    
    for i in range(batch_size):
        gt_class = ground_truths[2][i].item()
        class_results[gt_class].append((batch_evaluation_metrics[i]["precision"], batch_evaluation_metrics[i]["recall"]))
    
    print(f"Batch {batch_idx + 1}: {batch_evaluation_metrics}")

    del batch_video_features
    del batch_action_proposals
    torch.cuda.empty_cache()  # Release unused GPU memory
class_aps = {}

for cls, pr_values in class_results.items():
    sorted_pr_values = sorted(pr_values, key=lambda x: x[1])  # Sort by recall
    precisions = [pr[0] for pr in sorted_pr_values]
    recalls = [pr[1] for pr in sorted_pr_values]

    # Compute the AP using the trapezoidal rule
    ap = np.trapz(precisions, recalls)
    class_aps[cls] = ap

# Calculate mAP
mAP = np.mean(list(class_aps.values()))
print(f"mAP: {mAP}")