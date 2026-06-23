import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import numpy as np
import faiss

class SigLIPEmbedder:
    def __init__(self, model_id="google/siglip-so400m-patch14-384"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)

    def get_embedding(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            model_output = self.model.get_image_features(**inputs)
        
        emb = model_output.pooler_output.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(emb)
        return emb
