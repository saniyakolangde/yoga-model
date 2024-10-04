from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf
import mediapipe as mp
import numpy as np
import pandas as pd
import base64
from io import BytesIO

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('landmarks_model.h5')
class_names = ['downdog', 'goddess', 'plank', 'tree', 'warrior2']
confidence_threshold = 0.6 
visibility_threshold = 0.5
distance_threshold = 0.1
feedback_messages_dict = {
    11: "Your shoulders are not correctly positioned",
    12: "Your shoulders are not correctly positioned",
    23: "Your hips are not correctly aligned.",
    24: "Your hips are not correctly aligned.",
    25: "Your left knee is not positioned correctly.",
    26: "Your right knee is not positioned correctly.",
    27: "Your left foot is not positioned correctly.",
    28: "Your right foot is not positioned correctly."
}

critical_landmark_indices = [11, 12, 23, 24, 25, 26, 27, 28]
cumulative_deviation_threshold = 2.3  # Adjust based on requirements

# Load reference landmarks from CSV
reference_landmarks = pd.read_csv('pose_landmarks.csv')

def extract_landmarks(image):
    image_rgb = np.array(image.convert('RGB'))  # Convert PIL image to RGB NumPy array
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        landmarks = []
        visibility_scores = []
        
        for lm in results.pose_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
            visibility_scores.append(lm.visibility)

        landmarks = np.array(landmarks)
        visibility_scores = np.array(visibility_scores)

        visible_landmarks_count = np.sum(visibility_scores[critical_landmark_indices] > visibility_threshold)

        if visible_landmarks_count >= 4:
            return landmarks  # Return the landmarks as a 2D array
    return None

def calculate_distances_v2(user_landmarks, reference_landmarks):
    user_landmarks = np.asarray(user_landmarks, dtype=np.float64)
    reference_landmarks = np.asarray(reference_landmarks, dtype=np.float64)

    if np.any(np.isnan(user_landmarks)) or np.any(np.isnan(reference_landmarks)):
        raise ValueError("Landmarks contain NaN values, unable to calculate distances.")

    squared_differences = (user_landmarks - reference_landmarks) ** 2
    sum_of_squares = np.sum(squared_differences, axis=1)

    if np.any(sum_of_squares < 0):
        raise ValueError("Sum of squares contains negative values, cannot take square root.")

    distances = np.sqrt(sum_of_squares)

    return distances

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_pose():
    try:
        data = request.json
        image_data = base64.b64decode(data['image'])
        image = Image.open(BytesIO(image_data))  # Decode the image using PIL
        
        lm_data = extract_landmarks(image)
        if lm_data is not None:
            lm_data_reshaped = lm_data.reshape(1, -1)
            prediction = model.predict(lm_data_reshaped)
            class_idx = np.argmax(prediction)
            confidence = prediction[0][class_idx]

            if confidence < confidence_threshold:
                predicted_class_name = "Unknown"
            else:
                predicted_class_name = class_names[class_idx]

            response = {
                'pose': predicted_class_name,
                'confidence': float(confidence),
                'feedback': ''
            }

            reference_pose = reference_landmarks[reference_landmarks['label'] == predicted_class_name].values[:, :-1]
            if len(reference_pose) > 0:
                reference_landmarks_arr = np.array(reference_pose).reshape(-1, 3)
                critical_user_landmarks = lm_data[critical_landmark_indices]
                critical_reference_landmarks = reference_landmarks_arr[critical_landmark_indices]
                distances = calculate_distances_v2(lm_data, reference_landmarks_arr)
                critical_distances = distances[critical_landmark_indices]
                cumulative_deviation = np.sum(critical_distances)
                max_deviation_index = np.argmax(critical_distances)
                max_deviation_value = critical_distances[max_deviation_index]
                landmark_idx = critical_landmark_indices[max_deviation_index]

                if cumulative_deviation >= cumulative_deviation_threshold:
                    if max_deviation_value >= distance_threshold:
                        feedback_message = feedback_messages_dict.get(landmark_idx, f'Landmark {landmark_idx} is not correct.')
                        response['feedback'] = feedback_message
                    else:
                        response['feedback'] = "Amazing job! Your pose is perfect!"
                else:
                    response['feedback'] = "Great job! Your pose looks good!"
        else:
            response = {'pose': 'No pose detected', 'confidence': 0, 'feedback': 'No pose detected.'}
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
