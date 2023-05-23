import torch
from transformers import CLIPProcessor, CLIPModel

class BaseCLIPModel:
    def __init__(self, model_name='openai/clip-vit-base-patch32', device="cpu"):
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.device = device
        self.model = self.model.eval().to(self.device)

    def get_image_embedding(self, images, use_tensor=True):
        # batch_size = images.shape[0]
        # n_imgs = images.shape[1]
        # channels = images.shape[2]
        # width = images.shape[3]
        # height = images.shape[4]
        # if batch_size == 1:
        #     images = images.squeeze(0)
        # else:
        #     images = images.view(batch_size * n_imgs, channels, width, height)
        # inputs = self.processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            image_embeddings = self.model.get_image_features(**images)
        # image_embeddings = image_embeddings.view(batch_size, n_imgs, 512)
        return image_embeddings.numpy() if not use_tensor else image_embeddings

    def get_text_embedding(self, texts, use_tensor=True):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            text_embeddings = self.model.get_text_features(**inputs)
        return text_embeddings.numpy() if not use_tensor else text_embeddings
