import os
import json
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np

# Paths relative to camera (akshat)
FACES_DIR = os.path.join("data", "faces")
DB_JSON = "persons_db.json"

# Load PyTorch Feature Extractor Model (No XML/dlib required)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Identity()
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

known_embeddings = []
known_names = []

def extract_embedding(img_bgr):
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor_img = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(tensor_img).cpu().numpy().flatten()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        return embedding
    except Exception:
        return None

def load_known_faces():
    global known_embeddings, known_names
    known_embeddings.clear()
    known_names.clear()

    person_mapping = {}
    if os.path.exists(DB_JSON):
        try:
            with open(DB_JSON, "r") as f:
                data = json.load(f)
                for person_id, details in data.items():
                    person_mapping[person_id] = details.get("name", person_id)
        except Exception:
            pass

    if not os.path.exists(FACES_DIR):
        print(f"Directory not found: {FACES_DIR}")
        return

    for file_name in os.listdir(FACES_DIR):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(FACES_DIR, file_name)
            img = cv2.imread(image_path)
            if img is None:
                continue

            embedding = extract_embedding(img)
            if embedding is not None:
                known_embeddings.append(embedding)
                person_id = os.path.splitext(file_name)[0]
                display_name = person_mapping.get(person_id, person_id)
                known_names.append(display_name)

    print(f"Successfully loaded {len(known_embeddings)} target face embeddings.")

# Train/Load on import
load_known_faces()

def identify_face(crop_image):
    if crop_image is None or crop_image.size == 0 or len(known_embeddings) == 0:
        return "UNKNOWN"

    try:
        live_embedding = extract_embedding(crop_image)
        if live_embedding is None:
            return "UNKNOWN"

        similarities = [np.dot(live_embedding, k_emb) for k_emb in known_embeddings]
        best_match_idx = int(np.argmax(similarities))
        best_similarity = similarities[best_match_idx]

        if best_similarity > 0.60:
            return known_names[best_match_idx]

    except Exception:
        pass

    return "UNKNOWN"
