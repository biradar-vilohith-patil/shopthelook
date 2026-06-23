from ultralytics import YOLO
from src.embedder import SigLIPEmbedder
from src.search_engine import ProductSearchEngine

class ShopEngine:
    def __init__(self, index_path, id_map_path, hash_path="catalog_hashes.pkl"):
        # 1. Load YOLO for Object Detection
        self.detector = YOLO('yolov8n.pt') 
        
        # 2. Load the SigLIP Embedder
        self.embedder = SigLIPEmbedder()
        
        # 3. Load the FAISS Search Engine
        self.search_engine = ProductSearchEngine(index_path, id_map_path, hash_path)

    def detect_products(self, image, top_k=3):
        """Runs YOLO detection, crops items, and searches the catalog."""
        results = self.detector(image)
        final_results = []

        # Loop through detected bounding boxes
        for box in results[0].boxes:
            # Extract coordinates and crop the image
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop_img = image.crop((x1, y1, x2, y2))
            
            # Get YOLO label name (e.g., "tie", "handbag")
            class_id = int(box.cls[0])
            label = self.detector.names[class_id]
            conf = float(box.conf[0])
            
            # Skip the human, just grab the clothes/accessories
            if label == "person":
                continue

            # Embed the cropped image
            crop_emb = self.embedder.get_embedding(crop_img)
            
            # Search the catalog
            matches = self.search_engine.search(crop_emb, top_k=top_k)

            # Package the results for the Streamlit UI
            final_results.append({
                "crop": crop_img,
                "label": label,
                "conf": conf,
                "matches": matches
            })

        return final_results