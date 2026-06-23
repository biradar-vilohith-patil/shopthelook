import faiss
import pickle
import numpy as np
import cv2

class ProductSearchEngine:
    def __init__(self, index_path, id_map_path, hash_path):
        self.index = faiss.read_index(index_path)
        with open(id_map_path, 'rb') as f:
            self.image_ids, _ = pickle.load(f)
        with open(hash_path, 'rb') as f:
            self.catalog_hashes = pickle.load(f)

    def avg_color(self, img_np):
        return img_np.mean(axis=(0,1))

    def color_similarity(self, c1, c2):
        distance = np.linalg.norm(c1 - c2)
        return 1 / (1 + distance)

    def search(self, query_emb, top_k=5):
        scores, indices = self.index.search(query_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "product_id": self.image_ids[idx],
                "similarity": float(score)
            })
        return results
