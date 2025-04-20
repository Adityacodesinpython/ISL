import os
import cv2
import numpy as np
import mediapipe as mp
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, LayerNormalization, Embedding, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

# ====================== 1. Pose Landmark Extraction ======================
def extract_pose_landmarks(frame, pose):
    """Extracts pose landmarks (33 points, x,y,z) from a frame."""
    if frame is None:
        return None
        
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = np.array([[landmark.x, landmark.y, landmark.z] 
                                for landmark in results.pose_landmarks.landmark])
            return landmarks.flatten()
        return None
    except Exception as e:
        print(f"Landmark extraction error: {e}")
        return None

# ==================== 2. Temporal Dataset Creation ====================
def create_temporal_dataset(database, seq_length=16):
    """Creates sequences of pose landmarks"""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    X, y = [], []
    for category_folder in os.listdir(database):
        if category_folder.endswith(('.py', '.md', '.sh')):
            continue
            
        category_path = os.path.join(database, category_folder)
        if not os.path.isdir(category_path):
            continue
            
        # Extract the base category name (e.g., 'Adjectives' from 'Adjectives_1of8')
        base_category = category_folder.split('_')[0]
        
        # Path to the sign class folder
        sign_class_path = os.path.join(category_path, base_category)
        if not os.path.isdir(sign_class_path):
            os.makedirs(sign_class_path)
            continue
            
        for sign_class in os.listdir(sign_class_path):
            class_path = os.path.join(sign_class_path, sign_class)
            if not os.path.isdir(class_path):
                continue
                
            print(f"\nProcessing class: {sign_class} from {category_folder}")

            for video_file in os.listdir(class_path):
                if not video_file.lower().endswith(('.mov', '.mp4', '.avi')):
                    continue
                    
                video_path = os.path.join(class_path, video_file)
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_indices = np.linspace(0, total_frames-1, seq_length, dtype=int)
                sequence = []
                
                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    landmarks = extract_pose_landmarks(frame, pose) if ret else np.zeros(99)
                    sequence.append(landmarks if landmarks is not None else np.zeros(99))
                
                cap.release()
                X.append(np.array(sequence))
                y.append(sign_class)
        
    pose.close()
    return np.array(X), np.array(y)

# ========================= 3. TimeSformer Model =========================
def build_timesformer_model(input_shape, num_classes):
    """Builds a TimeSformer model"""
    inputs = Input(shape=input_shape)
    
    # Positional Encoding
    positions = tf.range(input_shape[0])
    position_embed = Embedding(input_dim=input_shape[0], output_dim=input_shape[1])(positions)
    
    # Add positional encoding
    x = inputs + position_embed
    
    # Transformer Blocks
    for _ in range(4):
        # Self-Attention
        attn_output = MultiHeadAttention(num_heads=4, key_dim=input_shape[1]//4)(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # Feed Forward
        ffn = tf.keras.Sequential([
            Dense(input_shape[1]*4, activation='gelu'),
            Dense(input_shape[1])
        ])
        x = LayerNormalization(epsilon=1e-6)(x + ffn(x))
    
    # Classification Head
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

# ========================= 4. Training =========================
def train_model(X, y):
    """Trains the TimeSformer model"""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42)
    
    model = build_timesformer_model(X.shape[1:], len(le.classes_))
    
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint('best_model.keras', save_best_only=True)  # Changed to .keras format
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    return model, le

# ========================= 5. Prediction =========================
def predict_sign(video_path, model, le):
    """Predicts sign from video"""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames-1, 16, dtype=int)
    sequence = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        landmarks = extract_pose_landmarks(frame, pose) if ret else np.zeros(99)
        sequence.append(landmarks if landmarks is not None else np.zeros(99))
    
    cap.release()
    pose.close()
    
    sequence = np.array(sequence)[np.newaxis, ...]  # Add batch dimension
    
    pred = model.predict(sequence)
    confidence = np.max(pred[0])
    predicted_class = le.inverse_transform([np.argmax(pred[0])])[0]
    
    return predicted_class, confidence

# ========================= 6. Main Execution =========================
if __name__ == "__main__":
    try:
        DATASET_PATH = r"D:\Aditya_Work\ISL_dataset"
        TEST_VIDEO = r"D:\Aditya_Work\ISL_dataset\Adjectives_1of8\Adjectives\1. loud\MVI_5177.MOV"
        
        # Create or load dataset
        if not os.path.exists('X.npy'):
            print("Creating temporal dataset...")
            X, y = create_temporal_dataset(DATASET_PATH)
            np.save('X.npy', X)
            np.save('y.npy', y)
        else:
            X = np.load('X.npy')
            y = np.load('y.npy')
        
        # Train or load model
        if not os.path.exists('best_model.keras'):
            print("\nTraining TimeSformer model...")
            model, le = train_model(X, y)
        else:
            print("\nLoading pre-trained model...")
            model = load_model('best_model.keras', compile=False)
            model.compile(optimizer=Adam(learning_rate=1e-4),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
            with open('label_encoder.pkl', 'rb') as f:
                le = pickle.load(f)
        
        # Predict
        if os.path.exists(TEST_VIDEO):
            print("\nPredicting sign...")
            sign, confidence = predict_sign(TEST_VIDEO, model, le)
            
            if confidence > 0.7:
                print(f"\n--- Recognition Result ---")
                print(f"Predicted Sign: {sign}")
                print(f"Confidence: {confidence:.2%}")
            else:
                print(f"\n--- Low Confidence Prediction ---")
                print(f"Predicted: {sign}")
                print(f"Confidence: {confidence:.2%}")
                print("Consider adding more training samples for this sign")
        else:
            print(f"\nError: Test video not found at {TEST_VIDEO}")
    
    except Exception as e:
        print(f"\nError occurred: {str(e)}")