import os
import unittest
from torch.utils.data import DataLoader
from src.utils.dataset_utils import ThumosDataset, get_thumos_dataloader

class TestThumosDataset(unittest.TestCase):
    def setUp(self):
        self.root_dir = '/home/ed/mm-ZSTAD/data/thumos14'
        self.assertGreater(len(os.listdir(self.root_dir)), 0, "The root directory is empty")
        self.train_dataset = ThumosDataset(self.root_dir, split='train')
        self.val_dataset = ThumosDataset(self.root_dir, split='val')
        self.test_dataset = ThumosDataset(self.root_dir, split='test')

    def test_dataset_length(self):
        self.assertGreater(len(self.train_dataset), 0, "Training dataset should not be empty.")
        self.assertGreater(len(self.val_dataset), 0, "Validation dataset should not be empty.")
        self.assertGreater(len(self.test_dataset), 0, "Test dataset should not be empty.")

    def test_dataset_item(self):
        for dataset in [self.train_dataset, self.val_dataset, self.test_dataset]:
            video_frames, audio_waveform, (start_times, end_times, action_classes) = dataset[0]
            self.assertIsNotNone(video_frames, "Video frames should not be None.")
            self.assertIsNotNone(audio_waveform, "Audio waveform should not be None.")
            self.assertIsNotNone(start_times, "Start times should not be None.")
            self.assertIsNotNone(end_times, "End times should not be None.")
            self.assertIsNotNone(action_classes, "Action classes should not be None.")

    def test_dataloader(self):
        train_dataloader = get_thumos_dataloader(self.root_dir, split='train', batch_size=16, num_workers=4)
        self.assertIsInstance(train_dataloader, DataLoader, "train_dataloader should be an instance of DataLoader.")

if __name__ == '__main__':
    unittest.main()
