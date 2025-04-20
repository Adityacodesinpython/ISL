import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import pickle
from content_analysis_fixed import predict_sign

def main():
    # Paths
    MODEL_PATH = 'best_model.keras'
    LABEL_ENCODER_PATH = 'label_encoder.pkl'
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run content_analysis_fixed.py first to train the model.")
        return
        
    # Load the model
    print("Loading model...")
    model = load_model(MODEL_PATH, compile=False)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    # Load label encoder
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
    
    while True:
        # Get video path from user
        print("\nEnter the path to your video file (or 'q' to quit):")
        video_path = input().strip()
        
        if video_path.lower() == 'q':
            break
            
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            continue
            
        # Predict the sign
        print("\nAnalyzing video...")
        try:
            sign, confidence = predict_sign(video_path, model, le)
            
            # Clean the predicted sign by removing numbers and dots
            clean_sign = ' '.join(word for word in sign.split() if not any(c.isdigit() for c in word))
            clean_sign = clean_sign.replace('.', '').strip()
            
            print("\n=== Recognition Result ===")
            print(f"Predicted Sign: {clean_sign}")
            print(f"Confidence: {confidence:.2%}")
            
            if confidence < 0.7:
                print("\nNote: Low confidence prediction. The model is not very sure about this result.")
                
        except Exception as e:
            print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
