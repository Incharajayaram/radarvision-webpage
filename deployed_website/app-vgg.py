from flask import Flask, request, jsonify, render_template
import torch 
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging
import os
from flask import Flask, request, jsonify, render_template, redirect, session, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
from dotenv import load_dotenv
from config import Config


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

num_classes = 6
model = initialize_vgg19(num_classes)
device = torch.device("cpu")
model.to(device)

model_path = 'best_model_CustomVGG.pt'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval() 


def predict(image_tensor, class_names):
    image_tensor = image_tensor.to(device)  
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return class_names[predicted.item()], confidence.item()


app = Flask(__name__, static_folder='static')

if not app.debug: 
    logging.basicConfig(level=logging.INFO)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/')
def home():
    return render_template("landingpage.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/predict', methods=['POST'])
def predict_route():
    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    image = transform(image).unsqueeze(0)
    class_names = ['3 long blade rotor', '3 short blade rotor', 'Bird', 'Bird + mini-helicopter', 'Drone', 'RC Plane']
    prediction, confidence = predict(image, class_names)
    app.logger.info('Prediction: %s, Confidence: %.4f', prediction, confidence)
    return jsonify({'prediction': prediction, 'confidence': confidence})

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
