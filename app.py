from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow import keras
from PIL import Image
import numpy as np
import pickle
import io
import os
from datetime import datetime, timedelta
import secrets
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///biovigil.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = secrets.token_hex(32)

db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.now)  # Changed to use local time
    token = db.Column(db.String(100), unique=True)
    
    # Relationship to scan history
    scans = db.relationship('ScanHistory', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'is_admin': self.is_admin,
            'created_at': self.created_at.isoformat()
        }

class ScanHistory(db.Model):
    __tablename__ = 'scan_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    fruit_type = db.Column(db.String(50), nullable=False)
    prediction = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    is_healthy = db.Column(db.Boolean, nullable=False)
    image_data = db.Column(db.Text)  # Base64 encoded image
    timestamp = db.Column(db.DateTime, default=datetime.now)  # Changed to use local time
    top_predictions = db.Column(db.Text)  # JSON string
    
    def to_dict(self):
        import json
        return {
            'id': self.id,
            'fruit_type': self.fruit_type,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'is_healthy': self.is_healthy,
            'imageUrl': self.image_data,
            'timestamp': self.timestamp.isoformat(),
            'top_predictions': json.loads(self.top_predictions) if self.top_predictions else []
        }

# Global variables for model and class names
model = None
class_names = []

def generate_token():
    return secrets.token_urlsafe(32)

def load_model():
    global model, class_names
    
    try:
        model = keras.models.load_model('fruithealth_model.h5')
        with open('class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
        print(f"✅ Model loaded successfully!")
        print(f"📊 Classes: {class_names}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
        return False

def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Fruit Health Classification API",
        "endpoints": {
            "GET /health": "Check API health and model status",
            "POST /predict": "Upload image for fruit health prediction",
            "GET /predict_local": "Predict using local example.jpg",
            "GET /classes": "Get list of supported classes"
        },
        "model_loaded": model is not None,
        "total_classes": len(class_names) if model else 0
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "model_loaded": model is not None,
        "classes_loaded": len(class_names) if model else 0
    })

@app.route('/classes', methods=['GET'])
def get_classes():
    return jsonify({
        "classes": class_names,
        "total_classes": len(class_names)
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        processed_image = preprocess_image(image)
        
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        predicted_class = class_names[predicted_class_idx]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = [
            {
                "class": class_names[i],
                "confidence": float(predictions[0][i])
            }
            for i in top_3_indices
        ]
        
        is_healthy = "Healthy" in predicted_class
        fruit_type = predicted_class.split('__')[0] if '__' in predicted_class else predicted_class
        
        response = {
            "prediction": predicted_class,
            "confidence": confidence,
            "fruit_type": fruit_type,
            "is_healthy": is_healthy,
            "top_predictions": top_predictions,
            "all_predictions": {
                class_names[i]: float(predictions[0][i]) 
                for i in range(len(class_names))
            }
        }
        
        try:
            auth_token = request.headers.get('Authorization', '').replace('Bearer ', '')
            if auth_token:
                user = User.query.filter_by(token=auth_token).first()
                if user:
                    file.seek(0)
                    image_data = base64.b64encode(file.read()).decode('utf-8')
                    image_url = f"data:image/jpeg;base64,{image_data}"
                    
                    import json
                    scan = ScanHistory(
                        user_id=user.id,
                        fruit_type=fruit_type,
                        prediction=predicted_class,
                        confidence=confidence,
                        is_healthy=is_healthy,
                        image_data=image_url,
                        top_predictions=json.dumps(top_predictions)
                    )
                    db.session.add(scan)
                    db.session.commit()
        except Exception as e:
            print(f"Warning: Failed to save scan history: {e}")
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/predict_local', methods=['GET'])
def predict_local():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        image_path = 'example.jpg'
        if not os.path.exists(image_path):
            return jsonify({"error": "example.jpg not found"}), 404
        
        image = Image.open(image_path)
        processed_image = preprocess_image(image)
        
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        predicted_class = class_names[predicted_class_idx]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = [
            {
                "class": class_names[i],
                "confidence": float(predictions[0][i])
            }
            for i in top_3_indices
        ]
        
        # Determine if it's healthy or rotten
        is_healthy = "Healthy" in predicted_class
        fruit_type = predicted_class.split('__')[0] if '__' in predicted_class else predicted_class
        
        response = {
            "prediction": predicted_class,
            "confidence": confidence,
            "fruit_type": fruit_type,
            "is_healthy": is_healthy,
            "top_predictions": top_predictions,
            "all_predictions": {
                class_names[i]: float(predictions[0][i]) 
                for i in range(len(class_names))
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/auth/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        if not all([username, email, password]):
            return jsonify({"error": "All fields are required"}), 400
        
        if User.query.filter_by(email=email).first():
            return jsonify({"error": "Email already registered"}), 400
        
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        
        db.session.add(new_user)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": "Registration successful"
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Registration failed: {str(e)}"}), 500

@app.route('/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not all([email, password]):
            return jsonify({"error": "Email and password required"}), 400
        
        user = User.query.filter_by(email=email).first()
        
        if not user or not user.check_password(password):
            return jsonify({"error": "Invalid email or password"}), 401
        
        token = generate_token()
        user.token = token
        db.session.commit()
        
        return jsonify({
            "success": True,
            "token": token,
            "user": user.to_dict()
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Login failed: {str(e)}"}), 500

def init_database():
    with app.app_context():
        db.create_all()
        
        admin_user = User.query.filter_by(email='admin@biovigil.com').first()
        if not admin_user:
            admin_user = User(
                username='Admin',
                email='admin@biovigil.com',
                is_admin=True
            )
            admin_user.set_password('admin123')
            db.session.add(admin_user)
            db.session.commit()
            print("✅ Admin user created!")
        else:
            print("✅ Admin user already exists")

@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        auth_token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not auth_token:
            return jsonify({"error": "Authentication required"}), 401
        
        user = User.query.filter_by(token=auth_token).first()
        if not user:
            return jsonify({"error": "Invalid token"}), 401
        
        scans = ScanHistory.query.filter_by(user_id=user.id).order_by(ScanHistory.timestamp.desc()).limit(50).all()
        
        return jsonify({
            "success": True,
            "history": [scan.to_dict() for scan in scans]
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to fetch history: {str(e)}"}), 500

@app.route('/api/history/<int:scan_id>', methods=['DELETE'])
def delete_scan(scan_id):
    try:
        auth_token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not auth_token:
            return jsonify({"error": "Authentication required"}), 401
        
        user = User.query.filter_by(token=auth_token).first()
        if not user:
            return jsonify({"error": "Invalid token"}), 401
        
        scan_to_delete = ScanHistory.query.get(scan_id)
        if not scan_to_delete:
            return jsonify({"error": "Scan not found"}), 404
        
        if scan_to_delete.user_id != user.id:
            return jsonify({"error": "You can only delete your own scans"}), 403
        
        db.session.delete(scan_to_delete)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": "Scan deleted successfully"
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Failed to delete scan: {str(e)}"}), 500

# Admin Endpoints
@app.route('/api/admin/users', methods=['GET'])
def admin_get_users():
    try:
        auth_token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not auth_token:
            return jsonify({"error": "Authentication required"}), 401
        
        user = User.query.filter_by(token=auth_token).first()
        if not user or not user.is_admin:
            return jsonify({"error": "Admin access required"}), 403
        
        users = User.query.all()
        users_data = []
        for u in users:
            user_dict = u.to_dict()
            user_dict['scan_count'] = len(u.scans)
            users_data.append(user_dict)
        
        return jsonify({
            "success": True,
            "users": users_data
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to fetch users: {str(e)}"}), 500

@app.route('/api/admin/scans', methods=['GET'])
def admin_get_scans():
    try:
        auth_token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not auth_token:
            return jsonify({"error": "Authentication required"}), 401
        
        user = User.query.filter_by(token=auth_token).first()
        if not user or not user.is_admin:
            return jsonify({"error": "Admin access required"}), 403
        
        scans = ScanHistory.query.order_by(ScanHistory.timestamp.desc()).limit(100).all()
        
        return jsonify({
            "success": True,
            "scans": [scan.to_dict() for scan in scans]
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to fetch scans: {str(e)}"}), 500

@app.route('/api/admin/stats', methods=['GET'])
def admin_get_stats():
    try:
        auth_token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not auth_token:
            return jsonify({"error": "Authentication required"}), 401
        
        user = User.query.filter_by(token=auth_token).first()
        if not user or not user.is_admin:
            return jsonify({"error": "Admin access required"}), 403
        
        total_users = User.query.count()
        total_scans = ScanHistory.query.count()
        healthy_scans = ScanHistory.query.filter_by(is_healthy=True).count()
        rotten_scans = ScanHistory.query.filter_by(is_healthy=False).count()
        
        today = datetime.utcnow().date()
        today_scans = ScanHistory.query.filter(
            db.func.date(ScanHistory.timestamp) == today
        ).count()
        
        active_users = db.session.query(ScanHistory.user_id).distinct().count()
        
        return jsonify({
            "success": True,
            "totalUsers": total_users,
            "totalScans": total_scans,
            "healthyScans": healthy_scans,
            "rottenScans": rotten_scans,
            "todayScans": today_scans,
            "activeUsers": active_users
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to fetch stats: {str(e)}"}), 500

@app.route('/api/admin/users/<int:user_id>', methods=['DELETE'])
def admin_delete_user(user_id):
    try:
        auth_token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not auth_token:
            return jsonify({"error": "Authentication required"}), 401
        
        admin = User.query.filter_by(token=auth_token).first()
        if not admin or not admin.is_admin:
            return jsonify({"error": "Admin access required"}), 403
        
        if admin.id == user_id:
            return jsonify({"error": "Cannot delete your own account"}), 400
        
        user_to_delete = User.query.get(user_id)
        if not user_to_delete:
            return jsonify({"error": "User not found"}), 404
        
        db.session.delete(user_to_delete)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": "User deleted successfully"
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Failed to delete user: {str(e)}"}), 500

@app.route('/api/admin/scans/<int:scan_id>', methods=['DELETE'])
def admin_delete_scan(scan_id):
    try:
        auth_token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not auth_token:
            return jsonify({"error": "Authentication required"}), 401
        
        user = User.query.filter_by(token=auth_token).first()
        if not user or not user.is_admin:
            return jsonify({"error": "Admin access required"}), 403
        
        scan_to_delete = ScanHistory.query.get(scan_id)
        if not scan_to_delete:
            return jsonify({"error": "Scan not found"}), 404
        
        db.session.delete(scan_to_delete)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": "Scan deleted successfully"
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Failed to delete scan: {str(e)}"}), 500

# Load model when starting the server
if __name__ == '__main__':
    print("🚀 Starting Fruit Health Classification API...")
    print("🗄️  Initializing SQLite database...")
    init_database()
    
    if load_model():
        print("🔐 Authentication endpoints enabled")
        print(" Database: biovigil.db")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ Failed to load model. Please train the model first.")