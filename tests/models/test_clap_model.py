import unittest
import numpy as np
import librosa
import torch
from src.models.base.clap_model import CLAPModel, int16_to_float32, float32_to_int16
from torchvision.io import read_video

class TestCLAPModel(unittest.TestCase):
    def setUp(self):
        self.clap_model = CLAPModel()
        
        _, audio_waveform, _ = read_video("/home/ed/mm-ZSTAD/tests/test_data/test.avi", pts_unit='sec')
        self.test_audio = audio_waveform.numpy()

    # def test_audio_embedding_from_filelist(self):
        
    #     audio_embed = self.clap_model.get_audio_embedding_from_filelist(audio_files)
    #     self.assertIsNotNone(audio_embed)
    #     self.assertEqual(audio_embed.shape[1], 512)

    def test_audio_embedding_from_data(self):
        audio_data = self.test_audio
        audio_data = audio_data.reshape(1, -1)
        audio_embed = self.clap_model.get_audio_embedding_from_data(audio_data)
        self.assertIsNotNone(audio_embed)
        self.assertEqual(audio_embed.shape[1], 512)

    def test_text_embedding(self):
        text_data = ["I love the contrastive learning", "I love the pretrain model"]
        text_embed = self.clap_model.get_text_embedding(text_data)
        self.assertIsNotNone(text_embed)
        self.assertEqual(text_embed.shape[1], 512)

if __name__ == '__main__':
    unittest.main()
