from flask import Flask, request, jsonify, render_template
# import cv2
# import tensorflow as tf
import mediapipe as mp
import numpy as np
# import pandas as pd
import base64
import tensorflow.lite as tflite
from PIL import Image


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

app = Flask(__name__)

import os
from google.cloud import storage

# Set the environment variable to the path of your JSON key file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'key2.json'

# Initialize a Cloud Storage client
client = storage.Client()

# Define your bucket name
bucket_name = 'yoga-model'
bucket = client.get_bucket(bucket_name)

# Function to download the model file
def download_model(file_name):
    blob = bucket.blob(file_name)
    blob.download_to_filename(file_name)
    return file_name

# Use the function to download the model
model_file = download_model('landmarks_model.h5')


# Load the trained model
# model = tf.keras.models.load_model(model_file)
# Load the TensorFlow Lite model
interpreter = tflite.Interpreter(model_path=model_file)
interpreter.allocate_tensors()

# Get input and output details of the TFLite model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
# reference_landmarks = pd.read_csv('pose_landmarks.csv')
import csv

def load_reference_landmarks(file_path):
    reference_landmarks = []
    labels = []
    
    with open(file_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            label = row[-1]  # Assuming the last column is the label
            labels.append(label)
            landmarks = [float(value) for value in row[:-1]]  # Convert landmarks to floats
            reference_landmarks.append(landmarks)

    reference_landmarks = np.array(reference_landmarks).reshape(-1, 33, 3)  # Assuming 33 landmarks
    return reference_landmarks, labels

# Loading the landmarks
reference_landmarks, labels = load_reference_landmarks('pose_landmarks.csv')

def extract_landmarks(frame):
    # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame)
    image = pil_image.convert("RGB")  # Convert to RGB
    image = np.array(image)  # Convert back to numpy array if needed
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
import io
@app.route('/predict', methods=['POST'])
def predict_pose():
    try:
        data = request.json
        image_data = base64.b64decode(data['image'])
        # np_image = np.frombuffer(image_data, np.uint8)

        # Use PIL to open the image from bytes
        image = Image.open(io.BytesIO(image_data))
        image = image.convert("RGB")  # Convert to RGB if needed

        # Convert PIL image to numpy array
        frame = np.array(image)

        # lm_data = extract_landmarks(frame)
        # frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        lm_data = extract_landmarks(frame)
        if lm_data is not None:
            # lm_data_reshaped = lm_data.reshape(1, -1)
            # Set input tensor
            lm_data_reshaped = lm_data.reshape(1, -1).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], lm_data_reshaped)

            # Invoke the interpreter to make predictions
            interpreter.invoke()

            # Get output tensor
            output_data = interpreter.get_tensor(output_details[0]['index'])
            prediction = output_data[0]
            # prediction = model.predict(lm_data_reshaped)
            class_idx = np.argmax(prediction)
            confidence = prediction[class_idx]
            # confidence = prediction[0][class_idx]

            if confidence < confidence_threshold:
                predicted_class_name = "Unknown"
            else:
                predicted_class_name = class_names[class_idx]

            response = {
                'pose': predicted_class_name,
                'confidence': float(confidence),
                'feedback': ''
            }

            # reference_pose = reference_landmarks[reference_landmarks['label'] == predicted_class_name].values[:, :-1]
            # Find the corresponding reference pose by label
            try:
                reference_index = labels.index(predicted_class_name)
                reference_pose = reference_landmarks[reference_index]
            except ValueError:
                reference_pose = None

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
