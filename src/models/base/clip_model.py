import torch
from transformers import CLIPProcessor, CLIPModel

class BaseCLIPModel:
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)

    def get_image_embedding(self, images, use_tensor=True):
        images = images.squeeze(0)
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            image_embeddings = self.model.get_image_features(**inputs)
        return image_embeddings.numpy() if not use_tensor else image_embeddings

    def get_text_embedding(self, texts, use_tensor=True):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_embeddings = self.model.get_text_features(**inputs)
        return text_embeddings.numpy() if not use_tensor else text_embeddings
