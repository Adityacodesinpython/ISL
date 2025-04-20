from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import tempfile
from werkzeug.utils import secure_filename
from content_analysis_fixed import predict_sign
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import pickle

app = Flask(__name__)
CORS(app)

# Load model and label encoder once
MODEL_PATH = 'best_model.keras'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

# Load the model and label encoder
print("Loading model...")
model = load_model(MODEL_PATH, compile=False)
model.compile(optimizer=Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
print("Model loaded successfully")

print("Loading label encoder...")
with open(LABEL_ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)

def create_video_from_frames(frames, output_path):
    """Create MP4 video from frames"""
    if not frames:
        return None
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    
    for frame in frames:
        out.write(frame)
    out.release()
    return output_path

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
        
    video_file = request.files['video']
    print(f"Received video file: {video_file.filename}")
    
    # Save the uploaded WebM video
    temp_webm = tempfile.NamedTemporaryFile(suffix='.webm', delete=False)
    video_file.save(temp_webm.name)
    print(f"Saved video to temporary file: {temp_webm.name}")
    
    try:
        # Extract frames from WebM
        cap = cv2.VideoCapture(temp_webm.name)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open video file'}), 400

        # Read all frames first
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        
        cap.release()
        print(f"Total frames read: {len(all_frames)}")
        
        if len(all_frames) == 0:
            return jsonify({'error': 'No frames could be extracted from video'}), 400
            
        # Sample 16 evenly spaced frames
        indices = np.linspace(0, len(all_frames)-1, 16, dtype=int)
        print(f"Sampling frames at indices: {indices}")
        frames = [all_frames[i] for i in indices]
            
        cap.release()
        temp_webm.close()
        
        if len(frames) != 16:
            return jsonify({'error': 'Could not extract 16 frames from video'}), 400

        # Create temporary MP4
        temp_mp4 = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_mp4.close()
        
        video_path = create_video_from_frames(frames, temp_mp4.name)
        print(f"Created MP4 video at: {video_path}")
        
        if video_path:
            try:
                print("Starting sign prediction...")
                sign, confidence = predict_sign(video_path, model, le)
                print(f"Prediction complete: {sign} with confidence {confidence}")
                
                result = {
                    'sign': sign,
                    'confidence': float(confidence)
                }
                print(f"Returning result: {result}")
                return jsonify(result)
                
            except Exception as e:
                print(f"Error predicting sign: {str(e)}")
                import traceback
                print(f"Full error: {traceback.format_exc()}")
                return jsonify({'error': str(e)}), 500
            finally:
                try:
                    if os.path.exists(temp_mp4.name):
                        os.unlink(temp_mp4.name)
                        print(f"Cleaned up MP4: {temp_mp4.name}")
                except Exception as e:
                    print(f"Error cleaning up MP4: {e}")
        
        if not predictions:
            return jsonify({'error': 'No valid predictions generated'}), 400
            
        return jsonify(predictions)
        
    except Exception as e:
        print(f"Error in predict endpoint: {e}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        try:
            if os.path.exists(temp_webm.name):
                os.unlink(temp_webm.name)
        except Exception as e:
            print(f"Error cleaning up WebM: {e}")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
