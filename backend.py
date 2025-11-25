import torch
import torch.nn as nn
import torchvision
from PIL import Image
import cv2
from ultralytics import YOLO
import torchvision.transforms as transforms

# Class names
CLASS_NAMES = [
    "helmet", "gloves", "vest", "boots", "goggles","none",
    "Person", "no_helmet", "no_goggle", "no_gloves", "no_boots"
]


# Simples CNN class 
class SimpleCNN(nn.Module):
    "Creation of my own customize CNN"
    def __init__(self, num_classes):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(32768, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x

# Transforms
## CNN
cnn_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
## ResNet50
resnet_weights = torchvision.models.ResNet50_Weights.DEFAULT
resnet_transform = resnet_weights.transforms()


# Setting the device
device = "cuda" if torch.cuda.is_available() else "cpu"


# Loading the models
def load_simple_cnn():
    model = SimpleCNN(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load("models/model_1_weights.pth", map_location=device))
    return model.to(device).eval()

def load_resnet50():
    model = torchvision.models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load("models/model_2_weights.pth", map_location=device))
    return model.to(device).eval()

def load_yolo():
    return YOLO("models/model_3/weights/best.pt")


# Inference
def predict_cnn(model, pil_image):
    img_tensor = cnn_transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = torch.sigmoid(model(img_tensor))[0]
    return [CLASS_NAMES[i] for i, v in enumerate(preds) if v > 0.5]

def predict_resnet(model, pil_image):
    img_tensor = resnet_transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = torch.sigmoid(model(img_tensor))[0]
    return [CLASS_NAMES[i] for i, v in enumerate(preds) if v > 0.5]

def predict_yolo(model, pil_image):
    result = model.predict(pil_image)[0]
    annotated = result.plot()
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
