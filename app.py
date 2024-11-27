from flask import Flask, request, jsonify, render_template, send_file
import cv2
import torch
import numpy as np
import os
from datetime import datetime
import logging
from werkzeug.utils import secure_filename
from ultralytics import YOLO  # Import YOLO from the ultralytics package

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.6

# Create necessary directories
def create_required_directories():
    """Create upload and results directories if they don't exist"""
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize directories
create_required_directories()

# Load YOLOv7 model
try:
    model = YOLO('yolov8s.pt')  # Load the YOLOv7 model
    logger.info("YOLOv8 model loaded successfully")
except Exception as e:
    logger.error(f"Error loading YOLOv8 model: {str(e)}")
    model = None

def allowed_file(filename):
    """Check if the file is of an allowed type."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def safe_file_path(directory, filename):
    """Create a safe file path and ensure directory exists."""
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, secure_filename(filename))

def process_image(image):
    """Process a single image and return detections with visualization."""
    try:
        # Run inference with YOLOv7
        results = model(image)
        result = results[0]
        
        # Filter detections based on confidence threshold
        mask = result.boxes.conf >= CONFIDENCE_THRESHOLD
        
        # Create a copy of the results with only high-confidence detections
        detections = []
        annotated_img = result.orig_img.copy()
        
        for box, conf, cls in zip(result.boxes.xyxy[mask], result.boxes.conf[mask], result.boxes.cls[mask]):
            x1, y1, x2, y2 = box.tolist()
            confidence = conf.item()
            class_name = model.names[int(cls)]
            
            # Draw bounding box
            cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f'{class_name} {confidence:.2f}'
            cv2.putText(annotated_img, label, (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Store detection information
            detection = {
                'class': class_name,
                'confidence': confidence,
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            }
            detections.append(detection)
        
        return detections, annotated_img
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_ext = os.path.splitext(file.filename)[1]
        safe_filename = f"upload_{timestamp}{original_ext}"
        filepath = safe_file_path(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(filepath)
        
        if file.content_type.startswith('image/'):
            img = cv2.imread(filepath)
            detections, annotated_img = process_image(img)
            
            if detections is None:
                raise Exception('Error processing image')
            
            output_filename = f"detected_{timestamp}.jpg"
            output_path = safe_file_path(app.config['RESULTS_FOLDER'], output_filename)
            cv2.imwrite(output_path, annotated_img)
            
            response = {
                'detections': detections,
                'annotated_image': output_filename,
                'statistics': {
                    'total_objects': len(detections),
                    'classes_found': list(set(d['class'] for d in detections))
                }
            }
        
        elif file.content_type.startswith('video/'):
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                raise Exception("Error opening video file")
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'X264')
            output_filename = f"detected_{timestamp}.mp4"
            output_path = safe_file_path(app.config['RESULTS_FOLDER'], output_filename)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_detections = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                detections, annotated_frame = process_image(frame)
                if detections is None:
                    continue
                
                out.write(annotated_frame)
                
                if detections:
                    frame_detections.append({
                        'frame': frame_count,
                        'detections': detections
                    })
                frame_count += 1
            
            cap.release()
            out.release()
            
            response = {
                'video_analysis': {
                    'output_video': output_filename,
                    'total_frames': frame_count,
                    'frames_with_detections': len(frame_detections),
                    'detections': frame_detections
                }
            }
        
        os.remove(filepath)
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in detect_objects: {str(e)}")
        if 'filepath' in locals():
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/results/<filename>')
def get_result(filename):
    try:
        response = send_file(
            os.path.join(app.config['RESULTS_FOLDER'], filename),
            as_attachment=False,
            mimetype='video/mp4' if filename.endswith('.mp4') else 'image/jpeg'
        )
        response.headers['Cache-Control'] = 'no-store'
        return response
    except Exception as e:
        logger.error(f"Error serving result file {filename}: {str(e)}")
        return jsonify({'error': 'File not found'}), 404

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'upload_dir_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
        'results_dir_exists': os.path.exists(app.config['RESULTS_FOLDER']),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    create_required_directories()
    app.run(debug=True)