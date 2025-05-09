import os
import cv2
import dlib
import numpy as np
import psycopg2
from flask import Flask, request, jsonify
from flask_cors import CORS
import datetime
import cloudinary
import cloudinary.uploader
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

# Configuration
DATABASE_URL = os.environ.get('DATABASE_URL')
CLOUDINARY_CLOUD_NAME = os.environ.get('CLOUDINARY_CLOUD_NAME')
CLOUDINARY_API_KEY = os.environ.get('CLOUDINARY_API_KEY')
CLOUDINARY_API_SECRET = os.environ.get('CLOUDINARY_API_SECRET')

# Initialize face detection
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Configure Cloudinary
cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET
)

# Database connection helper
def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    return conn

# Helper function to get face encodings
def get_face_encoding(image):
    try:
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = detector(image)
        if not faces:
            return None
            
        # Get first face
        face = faces[0]
        
        # Get facial landmarks
        shape = shape_predictor(image, face)
        
        # Get face encoding
        encoding = face_recognizer.compute_face_descriptor(image, shape)
        
        return np.array(encoding)
    except Exception as e:
        print(f"Error in face encoding: {str(e)}")
        return None

# API Endpoints

@app.route('/api/register', methods=['POST'])
def register_student():
    try:
        # Get data from request
        student_id = request.form.get('student_id')
        name = request.form.get('name')
        image = request.files.get('image')
        
        if not student_id or not name or not image:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Read image
        img_bytes = image.read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # Get face encoding
        encoding = get_face_encoding(img)
        if encoding is None:
            return jsonify({'error': 'No face detected in image'}), 400
        
        # Upload image to Cloudinary
        upload_result = cloudinary.uploader.upload(
            io.BytesIO(img_bytes),
            folder="student_faces",
            public_id=f"{student_id}_{name}"
        )
        
        # Save to database
        conn = get_db_connection()
        cur = conn.cursor()
        
        try:
            # Insert student
            cur.execute(
                "INSERT INTO students (student_id, name) VALUES (%s, %s) RETURNING id",
                (student_id, name)
            )
            student_db_id = cur.fetchone()[0]
            
            # Insert face encoding
            cur.execute(
                "INSERT INTO face_encodings (student_id, encoding) VALUES (%s, %s)",
                (student_id, encoding.tobytes())
            )
            
            conn.commit()
            
            return jsonify({
                'success': True,
                'student_id': student_id,
                'name': name,
                'image_url': upload_result['secure_url']
            }), 201
            
        except psycopg2.IntegrityError:
            conn.rollback()
            return jsonify({'error': 'Student ID already exists'}), 400
        finally:
            cur.close()
            conn.close()
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    try:
        # Get image from request
        image = request.files.get('image')
        if not image:
            return jsonify({'error': 'No image provided'}), 400
            
        # Read image
        img_bytes = image.read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # Get face encoding from uploaded image
        encoding = get_face_encoding(img)
        if encoding is None:
            return jsonify({'error': 'No face detected in image'}), 400
        
        # Get all face encodings from database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT s.student_id, s.name, f.encoding 
            FROM face_encodings f
            JOIN students s ON f.student_id = s.student_id
        """)
        
        # Find best match
        best_match = None
        min_distance = float('inf')
        threshold = 0.6  # Adjust as needed
        
        for row in cur.fetchall():
            db_student_id, db_name, db_encoding_bytes = row
            db_encoding = np.frombuffer(db_encoding_bytes, dtype=np.float64)
            
            # Calculate distance
            distance = np.linalg.norm(encoding - db_encoding)
            
            if distance < min_distance and distance < threshold:
                min_distance = distance
                best_match = {
                    'student_id': db_student_id,
                    'name': db_name,
                    'confidence': 1 - distance
                }
        
        cur.close()
        conn.close()
        
        if best_match:
            # Mark attendance
            mark_attendance(best_match['student_id'])
            
            return jsonify({
                'success': True,
                'match': best_match
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No matching student found'
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def mark_attendance(student_id):
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        now = datetime.datetime.now()
        date = now.date()
        time = now.time()
        
        # Check if already marked today
        cur.execute("""
            SELECT id FROM attendance 
            WHERE student_id = %s AND date = %s
        """, (student_id, date))
        
        if cur.fetchone() is None:
            # Mark attendance
            cur.execute("""
                INSERT INTO attendance (student_id, date, time)
                VALUES (%s, %s, %s)
            """, (student_id, date, time))
            
            conn.commit()
            return True
        return False
    finally:
        cur.close()
        conn.close()

@app.route('/api/attendance', methods=['GET'])
def get_attendance():
    try:
        date = request.args.get('date')
        if not date:
            date = datetime.datetime.now().date()
        else:
            date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT a.id, s.student_id, s.name, a.date, a.time
            FROM attendance a
            JOIN students s ON a.student_id = s.student_id
            WHERE a.date = %s
            ORDER BY a.time DESC
        """, (date,))
        
        attendance = []
        for row in cur.fetchall():
            attendance.append({
                'id': row[0],
                'student_id': row[1],
                'name': row[2],
                'date': row[3].strftime('%Y-%m-%d'),
                'time': str(row[4])
            })
        
        cur.close()
        conn.close()
        
        return jsonify({'attendance': attendance})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/students', methods=['GET'])
def get_students():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT id, student_id, name, registration_date
            FROM students
            ORDER BY name
        """)
        
        students = []
        for row in cur.fetchall():
            students.append({
                'id': row[0],
                'student_id': row[1],
                'name': row[2],
                'registration_date': row[3].strftime('%Y-%m-%d')
            })
        
        cur.close()
        conn.close()
        
        return jsonify({'students': students})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)