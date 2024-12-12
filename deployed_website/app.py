from flask import Flask, request, jsonify, render_template, redirect, session, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import torch
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import os
from dotenv import load_dotenv
from config import Config
import logging
import time


class CustomCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2nd Convolutional Block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 3rd Convolutional Block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 4th Convolutional Block
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layers
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, num_classes)  # Output for 6 classes

        # Dropout for Regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 1st Convolutional Block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # 2nd Convolutional Block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # 3rd Convolutional Block
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # 4th Convolutional Block
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        # Global Average Pooling
        x = self.global_avg_pool(x)

        # Flatten the output
        x = torch.flatten(x, 1)

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# Initialize the model
num_classes = 6
model_cnn = CustomCNN(num_classes=num_classes)
device = torch.device("cpu")
model_cnn.to(device)

# Load the trained model weights
model_path_cnn = 'best_model_CustomCNN.pt'  # Ensure the path matches your file
model_cnn.load_state_dict(torch.load(model_path_cnn, map_location=device))
model_cnn.eval()


def predictcnn(image_tensor, class_names):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model_cnn(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return class_names[predicted.item()], confidence.item() * 100

class CustomVGG(nn.Module):
    def __init__(self, base_model, num_classes):
        super(CustomVGG, self).__init__()
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(base_model.classifier[0].in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def initialize_vgg19(num_classes):
    vgg19_base = models.vgg19(pretrained=True)
    for param in vgg19_base.parameters():
        param.requires_grad = False
    return CustomVGG(vgg19_base, num_classes)


model_vgg = initialize_vgg19(num_classes)
model_vgg.to(device)

model_path_vgg = 'best_model_CustomVGG.pt'
model_vgg.load_state_dict(torch.load(model_path_vgg, map_location=device))
model_vgg.eval()


def predictvgg(image_tensor, class_names):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model_vgg(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return class_names[predicted.item()], confidence.item() * 100

# Define the Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)  # Reduce channels to 1
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=1)  # Learn spatial weights
        self.sigmoid = nn.Sigmoid()  # Normalize attention map to [0, 1]

    def forward(self, x):
        attn = self.conv1(x)  # Reduce channel dimension
        attn = self.conv2(attn)  # Learn spatial relationships
        attn = self.sigmoid(attn)  # Normalize
        return x * attn  # Apply attention

# Define the Custom CNN with integrated Spatial Attention
class CustomCNNWithAttention(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomCNNWithAttention, self).__init__()
        # 1st Convolutional Block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2nd Convolutional Block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 3rd Convolutional Block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 4th Convolutional Block
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Attention Module
        self.attention = SpatialAttention(in_channels=128)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, num_classes)  # Output for num_classes classes

        # Dropout for Regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 1st Convolutional Block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # 2nd Convolutional Block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # 3rd Convolutional Block
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # 4th Convolutional Block
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        # Apply Attention Mechanism
        x = self.attention(x)

        # Global Average Pooling
        x = self.global_avg_pool(x)
        
        # Flatten the output
        x = torch.flatten(x, 1)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
# Load the model
num_classes = 6
model = CustomCNNWithAttention(num_classes=num_classes)
device = torch.device("cpu")
model.to(device)

# Load trained model weights (ensure this path is correct)
model_path = 'customcnnwithAttention_best.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


    
def predictcnnwa(image_tensor, class_names):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class_idx = torch.max(probabilities, 1)
    return class_names[predicted_class_idx.item()], confidence.item() * 100




transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


app = Flask(__name__)

#DATABASE
load_dotenv()
app.secret_key = os.getenv('SECRET_KEY', 'fallback-secret-key')
app.config.from_object(Config)

app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///users.db'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = "False"

db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    username = db.Column(db.String(25), unique = True, nullable = False)
    email = db.Column(db.String(60), unique = True)  
    password_hash = db.Column(db.String(40), nullable = False)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

if not app.debug: 
    logging.basicConfig(level=logging.INFO)



UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template("landingpage.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()

    # Check if an image was uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Check if a model was selected
    if 'model' not in request.form:
        return jsonify({"error": "No model selected"}), 400

    model_selected = request.form['model']  # Get the selected model

    # Log the selected model for verification
    app.logger.info('Model selected: %s', model_selected)

    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Process the image
    image = Image.open(file.stream).convert('RGB')
    image = transform(image).unsqueeze(0)
    class_names = ['3 long blade rotor', '3 short blade rotor', 'Bird', 'Bird + mini-helicopter', 'Drone', 'RC Plane']
    
    # Predict based on selected model
    if model_selected == "cnn":
        app.logger.info('Using Custom CNN model for prediction.')
        prediction, confidence = predictcnn(image, class_names)
        app.logger.info('Prediction: %s, Confidence: %.4f', prediction, confidence)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return jsonify({'prediction': prediction, 'confidence': confidence, 'time':elapsed_time})

    elif model_selected == "vgg":
        app.logger.info('Using Custom VGG model for prediction.')
        prediction, confidence = predictvgg(image, class_names)
        app.logger.info('Prediction: %s, Confidence: %.4f', prediction, confidence)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return jsonify({'prediction': prediction, 'confidence': confidence, 'time':elapsed_time})    
    elif model_selected == "cnnwa":
        app.logger.info('Using Custom CNN with Attention model for prediction.')
        prediction, confidence = predictcnnwa(image, class_names)
        app.logger.info('Prediction: %s, Confidence: %.4f', prediction, confidence)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return jsonify({'prediction': prediction, 'confidence': confidence, 'time': elapsed_time})

    
@app.route('/health', methods=['GET'])
def health_check():
    return "OK", 200

@app.route('/contact')
def contact():
    access_key = os.getenv('WEB3FORMS_ACCESS_KEY')
    return render_template('contact.html', access_key=access_key)


@app.route('/upload')
def upload():
    if "username" not in session:
        flash("Please log in to access this page.", "error")
        return redirect(url_for('loginpage'))
    return render_template('upload.html')

@app.route('/sample_upload')
def sample_upload():
    sample_images = [f for f in os.listdir('static/sample_images') if f.endswith(('.jpg', '.png'))]
    return render_template('sample_upload.html', sample_images=sample_images)

@app.route('/login', methods=['GET','POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    user = User.query.filter_by(username=username).first()

    if user and user.check_password(password):
        session['username'] = username
        return redirect(url_for('home'))
    
    flash("Invalid username or password", "error")
    return redirect(url_for('loginpage'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']  # Capture confirm password

        if password != confirm_password:  # Check if passwords match
            flash("Passwords do not match.", "error")
            return redirect(url_for('signup'))

        user = User.query.filter_by(username=username).first()

        if user:
            flash("Username already exists,Login instead", "error")
            return redirect(url_for('signup'))
        else:
            new_user = User(username=username, email=email)
            new_user.set_password(password)  # Hash password before saving
            db.session.add(new_user)
            db.session.commit()
            flash("Account created successfully!", "success")
            return redirect(url_for('loginpage'))




@app.route('/loginpage')
def loginpage():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    
    with app.app_context():
        db.create_all()
    
    app.run(debug=True)
