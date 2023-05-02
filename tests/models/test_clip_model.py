import unittest
from PIL import Image
import numpy as np
from src.models.base.clip_model import BaseCLIPModel

class TestHuggingFaceCLIPModel(unittest.TestCase):
    def setUp(self):
        self.clip_model = BaseCLIPModel()

    def test_get_image_embedding(self):
        image = Image.open('/home/ed/mm-ZSTAD/tests/test_data/test.png')
        image_embedding = self.clip_model.get_image_embedding([image])
        self.assertEqual(image_embedding.shape, (1, 512))

    def test_get_text_embedding(self):
        text = ["An example sentence."]
        text_embedding = self.clip_model.get_text_embedding(text)
        self.assertEqual(text_embedding.shape, (1, 512))

if __name__ == '__main__':
    unittest.main()