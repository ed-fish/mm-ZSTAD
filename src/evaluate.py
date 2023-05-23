import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
import warnings
from collections import defaultdict

# Load ground truth and proposals into pandas DataFrame
gt_dtype = {'video_path': str, 'label': int, 'start_action': float, 'end_action': float}
prop_dtype = {'video-name': str, 't-start': float, 't-end': float, 'label': int, 'score': float}

# Load ground truth and proposals into pandas DataFrame with specified dtypes
gt_df = pd.read_csv("/home/ed/video/mm-ZSTAD/data/thumos14/val/thumos_val_list.csv", dtype=gt_dtype)
prop_df = pd.read_csv("/home/ed/video/mm-ZSTAD/src/output/proposals.csv", dtype=prop_dtype)


# Extract unique labels
proposal_labels = set(prop_df['label'].unique())
groundtruth_labels = set(gt_df['label'].unique())

# Check for labels present in proposals but not in ground truth
missing_in_gt = proposal_labels - groundtruth_labels
if missing_in_gt:
    print(f"Labels present in proposals but missing in ground truth: {missing_in_gt}")

# Check for labels present in ground truth but not in proposals
missing_in_proposals = groundtruth_labels - proposal_labels
if missing_in_proposals:
    print(f"Labels present in ground truth but missing in proposals: {missing_in_proposals}")

# Helper function to compute IoU
def compute_iou(segment1, segment2):
    start1, end1 = segment1
    start2, end2 = segment2
    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    return intersection / union

# IoU thresholds
iou_thresholds = np.arange(0.1, 1.0, 0.05)

# For simplicity, get only the video names from video_path
gt_df['video-name'] = gt_df['video_path'].apply(lambda x: x.split('/')[-1])

# Loop over all IoU thresholds
warnings.filterwarnings("ignore", category=UserWarning)  # Add this line to ignore the warning

ap_values = []

def format_proposals(proposals, video_name, label):
    formatted_proposals = []
    for proposal in proposals:
        start_time, end_time, score = proposal
        formatted_proposals.append({
            'video-name': video_name,
            't-start': start_time,
            't-end': end_time,
            'label': label,
            'score': score.item(),
        })
    return formatted_proposals

for iou_th in iou_thresholds:
    ap_values = []  # Reset ap_values for each IoU threshold

    tp_total = 0
    fp_total = 0
    fn_total = 0

    # Store the proposals for each video and label
    video_proposals = defaultdict(list)

    for label in prop_df['label'].unique():
        tp_class = 0
        fp_class = 0
        fn_class = 0

        for video_name in prop_df['video-name'].unique():
            gt_segments = gt_df[(gt_df['video-name'] == video_name) & (gt_df['label'] == label)][['start_action', 'end_action']].values
            prop_segments = prop_df[(prop_df['video-name'] == video_name) & (prop_df['label'] == label)][['t-start', 't-end', 'score']].values

            # Skip if there are no ground truth or proposal segments for this video and label
            if len(gt_segments) == 0 or len(prop_segments) == 0:
                continue

            # Compute IoU for all pairs of gt and proposal segments
            ious = np.array([[compute_iou(gt, prop[:2]) for prop in prop_segments] for gt in gt_segments])

            tp_fp = ious > iou_th
            tp = tp_fp.sum(axis=1).clip(max=1).sum()
            fp = tp_fp.sum(axis=0).clip(max=1).sum() - tp
            fn = len(gt_segments) - tp

            tp_class += tp
            fp_class += fp
            fn_class += fn

            # Store the proposals for the current video and label
            video_proposals[video_name, label].extend(format_proposals(prop_segments, video_name, label))

        tp_total += tp_class
        fp_total += fp_class
        fn_total += fn_class

        if tp_class + fp_class > 0:
            precision = tp_class / (tp_class + fp_class)
        else:
            precision = np.nan
        recall = tp_class / (tp_class + fn_class)

        y_true = [0] * fn_class + [1] * (tp_class + fp_class)
        y_scores = [1] * (tp_class + fp_class) + [0] * fn_class
        if np.any(gt_df["label"] == label):
            ap = average_precision_score(y_true, y_scores)
        else:
            ap = np.nan
        ap_values.append(ap)

    # Apply non-maximum suppression (NMS) to each video and label
    nms_proposals = []
    for video_label, proposals in video_proposals.items():
        # Sort proposals by score in descending order
        proposals.sort(key=lambda x: x['score'], reverse=True)

        # Apply NMS
        selected_proposals = []
        selected_indices = []
        for i, proposal in enumerate(proposals):
            if iou_th == 0:
                # For IoU threshold 0, select the proposal with the highest score
                if i not in selected_indices:
                    selected_proposals.append(proposal)
                    selected_indices.append(i)
            else:
                # For other IoU thresholds, apply NMS based on IoU
                if i not in selected_indices:
                    selected_proposals.append(proposal)
                    selected_indices.append(i)
                    for j in range(i + 1, len(proposals)):
                        if compute_iou([proposal['t-start'], proposal['t-end']], [proposals[j]['t-start'], proposals[j]['t-end']]) > iou_th:
                            selected_indices.append(j)

        # Add selected proposals to the final list
        nms_proposals.extend(selected_proposals)

    # Compute metrics and mAP for the selected proposals
    tp_fp = len(nms_proposals)
    tp = sum([1 for proposal in nms_proposals if proposal['label'] != 999])  # Modify the condition based on your label representation
    fp = tp_fp - tp
    fn = fn_total
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = np.nan
    recall = tp / (tp + fn)

    map_iou = np.mean(ap_values)
    print(f"mAP at IoU threshold {iou_th:.2f}: {map_iou}")
