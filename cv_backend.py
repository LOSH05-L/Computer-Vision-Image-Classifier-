import io
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from model import ImageClassifierCNN

app = FastAPI(title="CV Image Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Render free instances run on CPU
DEVICE = torch.device("cpu")
NUM_CLASSES = 2
CLASS_NAMES = ["Class_A_Healthy", "Class_B_Defective"]

# Load model weights on startup
model = ImageClassifierCNN(num_classes=NUM_CLASSES).to(DEVICE)
try:
    model.load_state_dict(torch.load("cnn_model_weights.pth", map_location=DEVICE))
    model.eval()
    print("CNN model weights loaded successfully for API.")
except Exception as e:
    print(f"Warning: Model weights not loaded: {e}")

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read image bytes uploaded from the frontend
        image_bytes = await file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Preprocessing pipeline
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            
        class_label = CLASS_NAMES[predicted_idx.item()] if predicted_idx.item() < len(CLASS_NAMES) else "Unknown"
        confidence_score = round(confidence.item() * 100, 2)
        
        return {
            "status": "success",
            "prediction": class_label,
            "confidence": f"{confidence_score}%"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Computer Vision Classifier Backend is live!"}