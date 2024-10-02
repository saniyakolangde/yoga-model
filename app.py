from flask import Flask, request, jsonify
import cv2
import tensorflow as tf
import mediapipe as mp
import numpy as np
import pandas as pd
import base64
import os
from flask_cors import CORS

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

app = Flask(__name__)
# CORS(app)
CORS(app, resources={r"/*": {"origins": "https://localhost:44352"}})


# Load the trained model
model_path = os.environ.get('MODEL_PATH', 'landmarks_model.h5')
model = tf.keras.models.load_model(model_path)
#model = tf.keras.models.load_model('landmarks_model.h5')
class_names = ['downdog', 'goddess', 'plank', 'tree', 'warrior2']
confidence_threshold = 0.6 
visibility_threshold = 0.5
distance_threshold = 0.1
# Define customized feedback messages for specific landmarks
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

# Critical landmarks: hips, legs, arms
critical_landmark_indices = [11, 12, 23, 24, 25, 26, 27, 28]

# Define a new cumulative threshold for the overall deviation
cumulative_deviation_threshold = 2.3  # Adjust this based on your requirements

# Load reference landmarks from CSV
reference_landmarks = pd.read_csv('pose_landmarks.csv')

def extract_landmarks(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    if results.pose_landmarks:
        landmarks = []
        visibility_scores = []
        
        for lm in results.pose_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
            visibility_scores.append(lm.visibility)

        landmarks = np.array(landmarks)
        visibility_scores = np.array(visibility_scores)

        critical_landmark_indices = [11, 12, 23, 24, 25, 26, 27, 28]
        visible_landmarks_count = np.sum(visibility_scores[critical_landmark_indices] > visibility_threshold)

        if visible_landmarks_count >= 4:
            return landmarks  # Return the landmarks as a 2D array
    return None

def calculate_distances_v2(user_landmarks, reference_landmarks):
    """
    Calculate the Euclidean distance between the user's landmarks and the reference landmarks.

    Arguments:
    user_landmarks -- np.array of shape (33, 3) containing the user's detected landmarks.
    reference_landmarks -- np.array of shape (33, 3) containing the reference landmarks.

    Returns:
    distances -- np.array of shape (33,) containing the Euclidean distances for each landmark.
    """
    # Ensure both arrays are numpy arrays and of the correct shape
    user_landmarks = np.asarray(user_landmarks, dtype=np.float64)
    reference_landmarks = np.asarray(reference_landmarks, dtype=np.float64)

    # Check for any NaN or invalid values in the landmarks
    if np.any(np.isnan(user_landmarks)) or np.any(np.isnan(reference_landmarks)):
        raise ValueError("Landmarks contain NaN values, unable to calculate distances.")

    # Debugging print statements
    print(f"User landmarks shape: {user_landmarks.shape}")
    print(f"Reference landmarks shape: {reference_landmarks.shape}")

    if user_landmarks.shape != reference_landmarks.shape:
        raise ValueError(f"Shape mismatch: User landmarks shape {user_landmarks.shape} does not match reference landmarks shape {reference_landmarks.shape}")

    # Vectorized calculation of Euclidean distance for each landmark
    squared_differences = (user_landmarks - reference_landmarks) ** 2
    sum_of_squares = np.sum(squared_differences, axis=1)

    # Check for any invalid values before taking the square root
    if np.any(sum_of_squares < 0):
        raise ValueError("Sum of squares contains negative values, cannot take square root.")

    distances = np.sqrt(sum_of_squares)

    return distances


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'https://localhost:44352')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict_pose():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
    elif request.method == "POST":
        try:
            data = request.json
            # Decode the base64-encoded image
            image_data = base64.b64decode(data['image'])
            np_image = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

            lm_data = extract_landmarks(frame)
            if lm_data is not None:
                lm_data_reshaped = lm_data.reshape(1, -1)
                prediction = model.predict(lm_data_reshaped)
                class_idx = np.argmax(prediction)
                confidence = prediction[0][class_idx]

                if confidence < confidence_threshold:
                    predicted_class_name = "Unknown"
                else:
                    predicted_class_name = class_names[class_idx]

                # Return the response with prediction and feedback
                response = {
                    'pose': predicted_class_name,
                    'confidence': float(confidence),
                    'feedback': ''
                }

                # Calculate distances and provide feedback if needed
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
                        incorrect_landmarks = np.where(critical_distances >= distance_threshold)[0]
                        feedback_texts = []
                        # for i in incorrect_landmarks:
                        #     landmark_idx = critical_landmark_indices[i]
                        #     feedback_message = feedback_messages_dict.get(landmark_idx, f'Landmark {landmark_idx} is not correct.')
                        #     feedback_texts.append(feedback_message)
                        # response['feedback'] = " | ".join(feedback_texts)
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

def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "https://localhost:44352")
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

if __name__ == '__main__':
    app.run(debug=True)
