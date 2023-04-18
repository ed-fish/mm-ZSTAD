import pandas as pd
import os

# create test and val formatted dataframe 
for split in ["test", "val"]:

    labels_df = pd.read_csv(f"/mnt/welles/scratch/datasets/thumos/annotations/{split}/labels.csv", sep=",", header=None, skiprows=1, names=['label', 'activity'])
    labels_df['label'] = labels_df['label'].astype('int64')

    print(labels_df)

    def read_temporal_files(path, labels_df):
        data = []
        for file in os.listdir(path):
            if file.endswith(".txt"):
                file_path = os.path.join(path, file)
                temp_df = pd.read_csv(file_path, header=None, sep=' ', names=['video_name', 'none', 'start_action', 'end_action'])
                activity = file.replace(f'_{split}.txt', '')
                label = int(labels_df.loc[labels_df['activity'] == activity, 'label'].values[0])
                temp_df['label'] = label
                data.append(temp_df)
        return pd.concat(data, ignore_index=True)

    temporal_df = read_temporal_files(f"/mnt/welles/scratch/datasets/thumos/annotations/{split}/temporal", labels_df)

    def build_final_dataframe(labels_df, temporal_df):
        final_df = temporal_df.merge(labels_df, on='label', how='left')
        final_df['video_path'] = final_df['video_name'].apply(lambda x: f'/mnt/welles/scratch/datasets/thumos/{split}/{x}.mp4')
        final_df = final_df[['video_path', 'label', 'start_action', 'end_action']]
        return final_df

    final_df = build_final_dataframe(labels_df, temporal_df)
    final_df.to_csv(f'/home/ed/mm-ZSTAD/data/thumos14/{split}/thumos_{split}.csv', index=False)
    
# Create train dataframe

train_dir = "/mnt/welles/scratch/datasets/thumos/train"
labels_df = pd.read_csv("/mnt/welles/scratch/datasets/thumos/annotations/train/labels.csv", sep=",")

def create_train_dataframe(train_dir, labels_df):
    data = []
    for file in os.listdir(train_dir):
        if file.endswith(".avi"):
            label_name = file.split("_")[1]
            label = int(labels_df.loc[labels_df['activity'] == label_name, 'label'].values[0])
            video_path = os.path.join(train_dir, file)
            data.append({"video_path": video_path, "label": label})

    train_df = pd.DataFrame(data)
    return train_df

train_df = create_train_dataframe(train_dir, labels_df)
train_df.to_csv('/home/ed/mm-ZSTAD/data/thumos14/train/train_data.csv', index=False)