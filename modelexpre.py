import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

# 🔹 Define Emotion Labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# 🔹 Load Pretrained CNN Model (ResNet-50)
class FERModel(nn.Module):
    def __init__(self, num_classes=7):
        super(FERModel, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Modify last layer

    def forward(self, x):
        return self.model(x)

# Load the model (Assuming a pre-trained FER model)
model = FERModel(num_classes=7)
model.load_state_dict(torch.load("fer_model.pth", map_location=torch.device('cpu')))  # Load weights
model.eval()  # Set to evaluation mode

# 🔹 Define Image Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
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

    cv2.imshow("Facial Expression Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
 