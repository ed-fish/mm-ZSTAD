import pandas as pd
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Load the CSV data into a pandas DataFrame
nms_df = pd.read_csv('/home/ed/video/mm-ZSTAD/nms_proposals.csv')

# Group the data by video name
grouped = nms_df.groupby('video-name')

# Dict to store the grouped data
grouped_data = {}

# Iterate over each group
for name, group in grouped:
    # Convert each group (DataFrame) to a list of dicts
    grouped_data[name] = group.to_dict('records')

# Get the first 10 videos to annotate
video_names = list(grouped_data.keys())[:10]

def label_videos(video_path, proposals, output_dir):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_path = f'{output_dir}/{video_path.split("/")[-1]}'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_index in range(frame_count):
        ret, frame = video.read()
        if not ret:
            break

        for proposal in proposals:
            start_frame = int(proposal['t-start'] * fps)
            end_frame = int(proposal['t-end'] * fps)

            if start_frame <= frame_index <= end_frame:
                label = proposal['label']
                score = proposal['score']

                # Draw rectangle and text on the frame
                cv2.rectangle(frame, (10, 10), (200, 60), (0, 0, 255), -1)
                cv2.putText(frame, f"Label: {label}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Score: {score}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out_video.write(frame)

    video.release()
    out_video.release()

for video_name in video_names:
    video_path = f'/mnt/welles/scratch/datasets/thumos/test/{video_name}'
    proposals = grouped_data[video_name]
    label_videos(video_path, proposals, '/home/ed/video/mm-ZSTAD/output/')
