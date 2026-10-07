import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from model import ImageClassifierCNN

CLASS_NAMES = ["Class_A_Healthy", "Class_B_Defective"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_inference_model(weights_path="cnn_model_weights.pth", num_classes=2):
    model = ImageClassifierCNN(num_classes=num_classes).to(DEVICE)
    try:
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        model.eval()
        print("Model weights loaded successfully for inference.")
    except FileNotFoundError:
        print("Model weights file not found. Please train the model first using model.py.")
    return model

def predict_image(image_path, model):
    try:
        # Load using PIL to handle webp, jpg, and png seamlessly
        pil_img = Image.open(image_path).convert("RGB")
        img_cv2 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error loading image at {image_path}: {e}")
        return

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
    confidence_score = confidence.item() * 100

    print(f">> Prediction: {class_label} ({confidence_score:.2f}% confidence)")

    # Display image with OpenCV window overlay
    cv2.putText(img_cv2, f"{class_label}: {confidence_score:.1f}%", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("CV Image Classifier Inference", img_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = load_inference_model()
    predict_image("dataset/train/Class_B_Defective/sample_1.webp", model)