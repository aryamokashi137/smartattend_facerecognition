from flask import Flask, render_template, request, jsonify
import cv2
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from keras_facenet import FaceNet
import pytz
import os
import re # Import the regular expression module

# Define the template folder path relative to the project root
template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates')

app = Flask(__name__, template_folder=template_dir)

# ========== Define Paths ==========
# Get the absolute path of the directory where the script is located (src)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project's root directory (one level up from 'src')
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# ========== Load Models ==========
print("🔄 Loading models...")
embedder = FaceNet()
svm_model = None
label_encoder = None
face_cascade = cv2.CascadeClassifier(os.path.join(BASE_DIR, "haarcascade", "haarcascade_frontalface_default.xml"))

# ========== Attendance Setup ==========
attendance_path = os.path.join(BASE_DIR, "attendance.csv")
if not os.path.exists(attendance_path):
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_csv(attendance_path, index=False)

# ========== Student Database Setup ==========
students_path = os.path.join(BASE_DIR, "students.csv")
if not os.path.exists(students_path):
    df_students = pd.DataFrame(columns=["PRN", "FullName", "Email", "Phone", "Department", "Semester"])
    df_students.to_csv(students_path, index=False)

# A set to hold names of people who have already been marked today.
# This avoids reading the CSV file on every single check.
todays_attendees = set()

def load_recognition_models():
    """Loads or reloads the SVM and LabelEncoder models into memory."""
    global svm_model, label_encoder
    try:
        svm_model = joblib.load(os.path.join(BASE_DIR, "models", "svm_face_recognition.joblib"))
        label_encoder = joblib.load(os.path.join(BASE_DIR, "models", "label_encoder.joblib"))
        print("✅ SVM and Label Encoder models loaded successfully.")
    except FileNotFoundError:
        print("⚠️ SVM or Label Encoder model not found. Please register at least two students and retrain.")

def load_todays_attendees():
    """Loads attendees for the current day from the CSV into memory."""
    try:
        global todays_attendees
        ist = pytz.timezone('Asia/Kolkata')
        today = datetime.now(ist).strftime("%Y-%m-%d")
        df = pd.read_csv(attendance_path)
        # Ensure the 'Name' column (which holds PRNs) is treated as a string
        todays_attendees = set(df[df["Date"] == today]["Name"].astype(str).values)
        print(f"ℹ️ Loaded {len(todays_attendees)} attendees for today.")
    except FileNotFoundError:
        print("⚠️ attendance.csv not found. Starting with an empty set of attendees.")
        todays_attendees = set()

def mark_attendance(prn):
    """Marks attendance using the student's PRN to ensure uniqueness."""
    # Ensure the PRN is a string for consistent checking
    prn_str = str(prn)

    if prn_str not in todays_attendees:
        now = datetime.now(pytz.timezone('Asia/Kolkata'))
        entry = pd.DataFrame([[prn_str, now.strftime("%Y-%m-%d"), now.strftime("%I:%M:%S %p")]],
                             columns=["Name", "Date", "Time"])
        entry.to_csv(attendance_path, mode="a", header=False, index=False)
        todays_attendees.add(prn_str) # Update our in-memory set
        print(f"✅ Attendance marked for PRN: {prn_str}")
    else:
        print(f"ℹ️ PRN: {prn_str} already marked today.")



@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Extract all form data
        prn = request.form.get('prn')
        full_name = request.form.get('full_name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        department = request.form.get('department')
        semester = request.form.get('semester')
        file = request.files.get('face') # This will now come from the webcam capture

        if not all([prn, full_name, email, phone, department, semester, file]):
            return jsonify({"status": "error", "message": "Missing form data."}), 400

        # Server-side validation for the email domain
        if not email.lower().endswith('@viit.ac.in'):
            return jsonify({"status": "error", "message": "Invalid email. Please use a valid '@viit.ac.in' address."}), 400

        # Server-side validation for the phone number format (+91 and 10 digits)
        phone_pattern = re.compile(r'^\+91[1-9][0-9]{9}$')
        if not phone_pattern.match(phone):
            return jsonify({"status": "error", "message": "Invalid phone number. Format must be +91 followed by 10 digits."}), 400

        # --- Save Student Details to CSV ---
        try:
            students_df = pd.read_csv(students_path, dtype={'PRN': str})
            # Ensure the PRN from the form is a string for comparison
            prn_str = str(prn)
            # Check if PRN already exists (as a string)
            if prn_str in students_df['PRN'].values:
                return jsonify({"status": "error", "message": f"Student with PRN {prn} is already registered."}), 409 # 409 Conflict

            new_student = pd.DataFrame([[prn, full_name, email, phone, department, semester]],
                                       columns=["PRN", "FullName", "Email", "Phone", "Department", "Semester"])
            new_student.to_csv(students_path, mode='a', header=False, index=False)
        except Exception as e:
            print(f"Error saving to students.csv: {e}")
            return jsonify({"status": "error", "message": "Could not save student data."}), 500

        # Use PRN for a unique, consistent filename
        faces_dir = os.path.join(BASE_DIR, "faces")
        os.makedirs(faces_dir, exist_ok=True)
        
        filename = os.path.join(faces_dir, f"{prn}.jpg")
        # Save the new image, overwriting if it exists (for updates)
        file.save(filename) 

        # --- RETRAIN AND RELOAD THE MODEL ---
        retrain_model()  # This saves the new models to disk
        load_recognition_models()  # This reloads them into memory

        return jsonify({"status": "success", "message": f"✅ Registered {full_name}. Model has been updated."})

    return render_template('register.html')

def retrain_model():
    print(" Retraining model with new data...")
    faces_dir = os.path.join(BASE_DIR, "faces")
    X, y = [], []

    for filename in os.listdir(faces_dir):
        if filename.endswith(".jpg"):
            prn = os.path.splitext(filename)[0]
            img_path = os.path.join(faces_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (160, 160))
            emb = embedder.embeddings([img_resized])[0]
            X.append(emb)
            y.append(prn)

    # SVM requires at least 2 different classes (people) to train
    if len(set(y)) < 2:
        print(f"⚠️ Found {len(X)} face(s) for {len(set(y))} unique person/people. Need at least 2 for training. Skipping.")
        return

    from sklearn.svm import SVC # Import here as it's only used for retraining
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X, y_encoded)

    joblib.dump(svm_model, os.path.join(BASE_DIR, "models", "svm_face_recognition.joblib"))
    joblib.dump(le, os.path.join(BASE_DIR, "models", "label_encoder.joblib"))
    print("✅ Model retrained and saved successfully.")


@app.route('/scan', methods=['GET', 'POST'])
def scan():
    if request.method == 'POST':
        file = request.files['frame']
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if svm_model is None or label_encoder is None:
            # Return a 503 Service Unavailable error if the model isn't ready
            return jsonify({"status": "error", "message": "Model not trained yet. Please register at least two students."}), 503

        # ✅ Remove forced rotation — causes wrong orientation in many webcams
        # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return jsonify({"status": "error", "message": "No face detected in the image."}), 400

        recognized_faces = []

        for (x, y, w, h) in faces:
            face = img[y:y + h, x:x + w]

            # ✅ Convert to RGB before embedding
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (160, 160))

            # ✅ Get embedding from FaceNet
            emb = embedder.embeddings([face_resized])[0]

            # ✅ Predict label
            pred = svm_model.predict([emb])[0]
            person_prn = label_encoder.inverse_transform([pred])[0]

            # ✅ Fetch full name from students.csv
            # Ensure PRN is read as a string to match the label's data type
            students_df = pd.read_csv(students_path)
            students_df['PRN'] = students_df['PRN'].astype(str)
            student_record = students_df[students_df['PRN'] == person_prn]

            if not student_record.empty:
                person_name = student_record.iloc[0]['FullName']
            else:
                person_name = "Unknown"

            # ✅ Mark attendance only if known
            if person_name != "Unknown":
                mark_attendance(person_prn)

            recognized_faces.append({
                "name": person_name,
                "box": [int(x), int(y), int(w), int(h)]
            })

        return jsonify({"status": "success", "faces": recognized_faces})
    
    return render_template('scan.html')



@app.route('/detect_face', methods=['POST'])
def detect_face():
    if 'frame' not in request.files:
        return jsonify({"status": "error", "message": "No frame sent."}), 400

    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # ✅ Don’t rotate image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        return jsonify({"status": "success", "box": [int(x), int(y), int(w), int(h)]})
    else:
        return jsonify({"status": "error", "message": "No face detected."})

 #========== ROUTES ==========

@app.route('/')
def index():
    try:
        # Load both attendance and student data, ensuring PRN columns are strings for merging
        attendance_df = pd.read_csv(attendance_path, dtype={'Name': str})
        students_df = pd.read_csv(students_path, dtype={'PRN': str})

        # Rename the 'Name' column in attendance_df to 'PRN' for a clean merge
        attendance_df.rename(columns={'Name': 'PRN'}, inplace=True)

        # Merge the two dataframes to get student details for each attendance record
        # A 'left' merge ensures all attendance records are kept, even if a student is later deleted
        merged_df = pd.merge(attendance_df, students_df, on='PRN', how='left')

        # If a student was deleted, their name will be NaN. Fill it with a placeholder.
        merged_df['FullName'].fillna('Student Deleted', inplace=True)

        today_str = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d")
        todays_df = merged_df[merged_df["Date"] == today_str]
        
        stats = {
            "total_records": len(merged_df),
            "todays_attendance": len(todays_df),
            "present_count": len(todays_df["PRN"].unique())
        }

        # Sort by date and time descending to show the latest records first
        merged_df = merged_df.sort_values(by=["Date", "Time"], ascending=[False, False])
        records = merged_df.to_dict(orient='records')

        return render_template('index.html', stats=stats, records=records)
    except FileNotFoundError:
        # Handle case where one of the CSVs might not exist yet
        return render_template('index.html', stats={"total_records": 0, "todays_attendance": 0, "present_count": 0}, records=[])

if __name__ == "__main__":
    load_todays_attendees() # Load attendees when the app starts
    load_recognition_models() # Load recognition models on startup
    app.run(debug=True)
