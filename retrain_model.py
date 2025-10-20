import os
import cv2
import numpy as np
import joblib
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys

# -----------------------------
# Directories
# -----------------------------
BASE_DIR = os.getcwd()
FACES_DIR = os.path.join(BASE_DIR, "faces")
if not os.path.exists(FACES_DIR):
    print(f"❌ Error: 'faces' directory not found at {FACES_DIR}")
    sys.exit(1)

MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------------------
# Load FaceNet Embedder
# -----------------------------
print("🔄 Loading FaceNet...")
embedder = FaceNet()

# -----------------------------
# Prepare Data
# -----------------------------
X, y = [], []

print("🔄 Processing images in 'faces' directory...")


for filename in os.listdir(FACES_DIR):
    if filename.endswith(".jpg"):
        # Correctly extract PRN from filenames like '12345_0.jpg'
        prn = os.path.splitext(filename)[0].split('_')[0]
        img_path = os.path.join(FACES_DIR, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Convert to RGB, resize, and get embedding
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (160, 160))
        emb = embedder.embeddings([img_resized])[0]
        
        X.append(emb)
        y.append(prn)

print(f"✅ Found {len(X)} face(s) for {len(set(y))} unique person/people.")

# -----------------------------
# Check Minimum Classes
# -----------------------------
if len(set(y)) < 2:
    print("\n❌ Error: Cannot train model. Please register at least two different people.")
    sys.exit(1)

# -----------------------------
# Encode Labels & Train SVM
# -----------------------------
print("🔄 Training SVM model...")

le = LabelEncoder()
y_encoded = le.fit_transform(y)

svm_model = SVC(kernel='linear', probability=True)

# Check if there are enough samples in each class to perform a train/test split
unique_classes, counts = np.unique(y_encoded, return_counts=True)
if np.min(counts) >= 2:
    # We can safely split and evaluate
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    svm_model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"📊 Model Accuracy on test set: {accuracy * 100:.2f}%")
    
    # Retrain on the full dataset for the final deployed model
    print("🔄 Retraining on full dataset for deployment...")
    svm_model.fit(X, y_encoded)
else:
    # Not enough data to split, so just train on everything
    print("⚠️ Skipping train/test split because at least one class has fewer than 2 samples.")
    print("🔄 Training on full dataset for deployment...")
    svm_model.fit(X, y_encoded)
# -----------------------------
# Save Models
# -----------------------------
print("💾 Saving models to disk...")
joblib.dump(svm_model, os.path.join(MODELS_DIR, "svm_face_recognition.joblib"))
joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.joblib"))

print("\n✅ Model trained and saved successfully!")
