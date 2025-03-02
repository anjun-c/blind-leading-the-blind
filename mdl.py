import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import numpy as np

# 🔹 Define Emotion Labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# 🔹 Define the EfficientNet-based Model for FER
class FERModel(nn.Module):
    def __init__(self, num_classes=7):
        super(FERModel, self).__init__()
        # Load EfficientNet-B0 (the smallest EfficientNet model, you can try others like B1, B2 for better performance)
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.model._fc = nn.Linear(self.model._fc.in_features, num_classes)  # Modify last layer for FER

    def forward(self, x):
        return self.model(x)

# 🔹 Load Pre-trained EfficientNet Model
model = FERModel(num_classes=7)
model.load_state_dict(torch.load("efficientnet_fer_model.pth", map_location=torch.device('cpu')))  # Load your custom trained weights
model.eval()  # Set to evaluation mode

# 🔹 Define Image Preprocessing (For EfficientNet)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize image to 224x224 as required by EfficientNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Pre-trained EfficientNet normalization
])

# 🔹 Load OpenCV Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 🔹 Open Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]  # Extract face ROI
        face_tensor = transform(face).unsqueeze(0)  # Preprocess for model

        # 🔹 Predict Emotion
        with torch.no_grad():
            outputs = model(face_tensor)
            _, predicted = torch.max(outputs, 1)
            emotion = emotion_labels[predicted.item()]

        # 🔹 Draw Rectangle & Label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Facial Expression Recognition with EfficientNet", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
