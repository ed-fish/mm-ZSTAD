import os
import numpy as np
from PIL import Image
import soundfile as sf
import torch
from models.base import BaseCLAPModel, BaseCLIPModel
from utils import ThumosDataset, get_thumos_dataloader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import re
import scipy.signal
from tqdm import tqdm

def calculate_iou(segment1, segment2):
    start1, end1 = segment1
    start2, end2 = segment2

    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)

    intersection = max(0, intersection_end - intersection_start)
    union = (end1 - start1) + (end2 - start2) - intersection

    return intersection / union

def calculate_tp_fp_fn(gt_segments, pred_segments, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0

    for pred_segment in pred_segments:
        is_tp = False
        for gt_segment in gt_segments:
            iou = calculate_iou(pred_segment, gt_segment)
            if iou >= iou_threshold:
                is_tp = True
                break

        if is_tp:
            tp += 1
        else:
            fp += 1

    for gt_segment in gt_segments:
        is_fn = True
        for pred_segment in pred_segments:
            iou = calculate_iou(gt_segment, pred_segment)
            if iou >= iou_threshold:
                is_fn = False
                break

        if is_fn:
            fn += 1

    return tp, fp, fn

def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1_score

def split_camel_case(s):
    return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', s)

def cosine_similarity(a, b):
    a_tensor = torch.tensor(a).to(b.device)
    b_tensor = torch.tensor(b).to(b.device)
    return torch.nn.functional.cosine_similarity(a_tensor, b_tensor, dim=0)

def find_action_segments(similarity_scores, threshold, min_duration, smoothing_window_size=5):
    # Apply Gaussian smoothing
    window = scipy.signal.windows.hann(smoothing_window_size)
    similarity_scores = [s.cpu() for s in similarity_scores]
    smoothed_scores = np.convolve(similarity_scores, window, mode='same') / sum(window)

    # Identify continuous segments above the threshold
    above_threshold = smoothed_scores > threshold
    action_segments = []
    start_idx = None

    for idx, is_above in enumerate(above_threshold):
        if is_above and start_idx is None:
            start_idx = idx
        elif not is_above and start_idx is not None:
            end_idx = idx
            duration = end_idx - start_idx
            if duration >= min_duration:
                action_segments.append((start_idx, end_idx))
            start_idx = None

    return action_segments

def save_frame_action_segments_figure(video_frames, action_segments, original_class_label, output_dir, resize_factor=0.2):
    num_frames = len(video_frames[0])
    num_columns = 10
    num_rows = math.ceil(num_frames / num_columns)

    figsize = (num_columns * 2, num_rows * 2)

    fig, axs = plt.subplots(num_rows, num_columns, figsize=figsize)
    axs = axs.flatten()

    # Create a boolean array with True where the action is happening, False otherwise
    action_present = np.zeros(num_frames, dtype=bool)
    for start, end in action_segments:
        action_present[start:end] = True

    for i, frame in enumerate(video_frames[0]):
        resized_frame = frame.numpy().transpose(1, 2, 0)
        resized_frame = Image.fromarray((resized_frame).astype(np.uint8)).resize((int(resized_frame.shape[1] * resize_factor), int(resized_frame.shape[0] * resize_factor)))
        axs[i].imshow(resized_frame)
        axs[i].set_xticks([])
        axs[i].set_yticks([])

        color = 'green' if action_present[i] else 'red'
        action_box = patches.Rectangle((0, 1.1), 1, 0.2, facecolor=color, transform=axs[i].transAxes, clip_on=False)
        axs[i].add_patch(action_box)

        axs[i].set_title(f"Frame {i}")

    for i in range(num_frames, num_rows * num_columns):
        axs[i].axis('off')

    # Save the figure with a suitable dpi to fit all frames
    plt.savefig(os.path.join(output_dir, f'{original_class_label}_frame_action_segments.png'), bbox_inches='tight', pad_inches=0.1, dpi=300)
    
def convert_segments_to_time(action_segments, frame_rate):
    time_based_segments = []
    for start_frame, end_frame in action_segments:
        start_time = start_frame / frame_rate
        end_time = end_frame / frame_rate
        time_based_segments.append((start_time, end_time))
    return time_based_segments
    
def test_one_sample():
     
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    clip_model = BaseCLIPModel()
    clap_model = BaseCLAPModel()

    root_dir = 'data/thumos14'
    split = 'val'
    train_loader = get_thumos_dataloader(root_dir, split=split, batch_size=1)
    video_frames, audio_waveforms, (start, end, labels) = next(iter(train_loader))

    # Load class labels and create text prompts
    class_labels_file = os.path.join(root_dir, 'classes.txt')
    with open(class_labels_file, 'r') as f:
        class_labels = [line.strip() for line in f.readlines()]

    class_label_dict = {int(label.split()[0]): f"a video of {split_camel_case(' '.join(label.split()[1:]))}" for label in class_labels}

    label = labels[-1].item()
    text_prompt = class_label_dict[label]

    text_embedding_clip = clip_model.get_text_embedding([text_prompt], use_tensor=True)[0]
    text_embedding_clap = clap_model.get_text_embedding([text_prompt, ""], use_tensor=True)[0]
    frame_features = clip_model.get_image_embedding(video_frames, use_tensor=True)
    # audio_features = clap_model.get_audio_embedding_from_data(audio_waveforms, use_tensor=True)

    frame_similarities = [cosine_similarity(text_embedding_clip, frame_feature) for frame_feature in frame_features]

    similarity_threshold = 0.29  # You can optimize this using a validation set
    min_duration = 2  # Minimum duration of an action segment in frames
    smoothing_window_size = 10  # Size of the smoothing window

    # Find action segments
    action_segments = find_action_segments(frame_similarities, similarity_threshold, min_duration, smoothing_window_size)
    time_based_segments = convert_segments_to_time(action_segments, 30) 
    
    formatted_action_segments = [f"{start:.2f}-{end:.2f}" for start, end in time_based_segments]
    formatted_gt_segments = [f"{start.item():.2f}-{end.item():.2f}" for start, end in zip(start.squeeze(), end.squeeze())]

    print("Predicted action segments:")
    print(", ".join(formatted_action_segments))
    print("\nGround truth action segments:")
    print(", ".join(formatted_gt_segments))
    
def test_all_samples():
    iou_threshold = 0.5
    classwise_ap = []
    
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    clip_model = BaseCLIPModel()
    clap_model = BaseCLAPModel()
    
    root_dir = 'data/thumos14'
    split = 'val'
    train_loader = get_thumos_dataloader(root_dir, split=split, batch_size=1)
    video_frames, audio_waveforms, (start, end, labels) = next(iter(train_loader))

    # Load class labels and create text prompts
    class_labels_file = os.path.join(root_dir, 'classes.txt')
    with open(class_labels_file, 'r') as f:
        class_labels = [line.strip() for line in f.readlines()]

    class_label_dict = {int(label.split()[0]): f"a video of {split_camel_case(' '.join(label.split()[1:]))}" for label in class_labels}

    label = labels[-1].item()
    text_prompt = class_label_dict[label]

    for class_id, class_text in class_label_dict.items():
        class_tp, class_fp, class_fn = 0, 0, 0

        for video_frames, audio_waveforms, (start, end, labels) in tqdm(train_loader):
            # Perform the same processing as before to get action_segments
            text_prompt = class_label_dict[class_id]

            text_embedding_clip = clip_model.get_text_embedding([text_prompt], use_tensor=True)[0]
            frame_features = clip_model.get_image_embedding(video_frames, use_tensor=True)

            frame_similarities = [cosine_similarity(text_embedding_clip, frame_feature) for frame_feature in frame_features]
            
            similarity_threshold = 0.29  # You can optimize this using a validation set
            min_duration = 2  # Minimum duration of an action segment in frames
            smoothing_window_size = 10  # Size of the smoothing window

            # Find action segments
            action_segments = find_action_segments(frame_similarities, similarity_threshold, min_duration, smoothing_window_size) 
            time_based_segments = convert_segments_to_time(action_segments, 30) 

            # Get the ground truth segments for the current class
            gt_segments = [(start[i], end[i]) for i, label in enumerate(labels) if label.item() == class_id]

            tp, fp, fn = calculate_tp_fp_fn(gt_segments, time_based_segments, iou_threshold)
            class_tp += tp
            class_fp += fp
            class_fn += fn
            print(class_tp, class_fp, class_fn)

        class_precision, class_recall, class_f1_score = calculate_metrics(class_tp, class_fp, class_fn)
        class_ap = class_recall * class_precision  # Assuming PR curve is rectangular
        classwise_ap.append(class_ap)

    # Calculate the mean average precision (mAP) across all classes
    mAP = sum(classwise_ap) / len(classwise_ap)
    print(f"Mean Average Precision (mAP): {mAP}")

test_all_samples() 


# save_frame_action_segments_figure(video_frames, action_segments, label, output_dir)




# audio_similarities = [cosine_similarity(text_embedding_clap, audio_feature) for audio_feature in audio_features]

# save_frame_similarities_figure(video_frames, frame_similarities, label, output_dir)




# max_frame_similarity_idx = np.argmax(frame_similarities)
# min_frame = np.argmin(frame_similarities)
# # max_audio_similarity_idx = torch.argmax(audio_similarities)

# most_similar_frame = video_frames[0][max_frame_similarity_idx].numpy().transpose(1,2,0)

# least_similar_frame = video_frames[0][min_frame].numpy().transpose(1,2,0)
# # most_similar_audio = audio_waveforms[max_audio_similarity_idx].numpy()


# # Get the original class label (before processing)
# original_class_label = class_label_dict[label]

# frame_img = Image.fromarray((most_similar_frame).astype(np.uint8))
# frame_img.save(os.path.join(output_dir, f'{original_class_label}_most_similar_frame.png'))

# frame_img = Image.fromarray((least_similar_frame).astype(np.uint8))
# frame_img.save(os.path.join(output_dir, f'{original_class_label}_least_similar_frame.png'))

# audio_waveforms = audio_waveforms.squeeze(0).transpose(1, 0)

# sf.write(os.path.join(output_dir, f'audio.wav'), audio_waveforms.numpy(), samplerate=16000)
