from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import os
import time
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import base64
import io

# Define the CNN model class
class CustomCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Load the model
num_classes = 6
model = CustomCNN(num_classes=num_classes)
device = torch.device("cpu")
model.to(device)

# Load trained model weights (ensure this path is correct)
model_path = 'best_model_CustomCNN.pt'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Prediction function
def predict(image_tensor):
    image_tensor = image_tensor.to(device)  
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class_idx = torch.max(probabilities, 1)
    return predicted_class_idx.item(), confidence.item()

@app.route('/')
def home():
    return render_template("index.html")

'''import os
import random
import shutil

def shuffle_and_save_images(source_folder, destination_folder):
    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Gather all images from the source folder
    all_images = []
    for class_name in os.listdir(source_folder):
        class_path = os.path.join(source_folder, class_name)
        if os.path.isdir(class_path):  # Ensure it's a directory
            images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
            all_images.extend(images)

    # Shuffle the images
    random.shuffle(all_images)

    # Save shuffled images to the destination folder
    for idx, img_path in enumerate(all_images):
        # Create a new filename based on index
        new_filename = f"image_{idx + 1}{os.path.splitext(img_path)[1]}"
        shutil.copy(img_path, os.path.join(destination_folder, new_filename))

    print(f"Shuffled {len(all_images)} images saved to '{destination_folder}'.")

# Example usage
source_folder = 'D:/Micro-Classify/ml_model/notebooks/DIAT-uSAT_dataset'  # Change this to your source folder path
destination_folder = 'D:/webpage-microdopler/Frontend/rpi_website/shuffled_images'  # Change this to your destination folder path

shuffle_and_save_images(source_folder, destination_folder)
'''

def stream_images():
    # image_folder_path = 'D:/Micro-Classify/ml_model/notebooks/DIAT-uSAT_dataset/3_long_blade_rotor' 
    image_folder_path = 'shuffled_images' 
    class_names = ['3 long blade rotor', '3 short blade rotor', 'Bird', 'Bird + mini-helicopter', 'Drone', 'RC Plane']
    
    while True:
        for filename in os.listdir(image_folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'): 
                file_path = os.path.join(image_folder_path, filename)

                # Open and process the image
                image_bytes = Image.open(file_path).convert('RGB')
                image_tensor = transform(image_bytes).unsqueeze(0)

                # Perform prediction
                predicted_idx, confidence_score = predict(image_tensor)

                # Get class name and confidence score as percentage
                prediction_label = class_names[predicted_idx]
                confidence_percentage = confidence_score * 100

                # Encode image to base64 for sending to frontend
                buffered = io.BytesIO()
                image_bytes.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

                # Emit the result back to the client with the image data and prediction result
                socketio.emit('image_stream', {
                    'image': f"data:image/jpeg;base64,{img_str}",
                    'prediction': prediction_label,
                    'confidence': f"{confidence_percentage:.2f}%"
                })

                time.sleep(0.8)  # Wait for 1 second before sending the next image

@socketio.on('connect')
def start_stream():
    socketio.start_background_task(stream_images)  # Start streaming images when a client connects

if __name__ == '__main__':
    socketio.run(app, debug=True)
