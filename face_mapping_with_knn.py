import cv2
import os
import time
import numpy as np
import face_recognition
from ultralytics import YOLO
from face_recognition.face_recognition_cli import image_files_in_folder

pose_model = YOLO("yolov8s-pose.pt")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}

def load_reference_images(reference_dir):
    known_encodings = []
    known_names = []
    for class_dir in os.listdir(reference_dir):
        if not os.path.isdir(os.path.join(reference_dir, class_dir)):
            continue
        for img_path in image_files_in_folder(os.path.join(reference_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_encodings.append(encoding[0])
                known_names.append(class_dir)
    return known_encodings, known_names

def predict_direct(X_frame, known_encodings, known_names, n_neighbors=4, distance_threshold=0.5):
    X_face_locations = face_recognition.face_locations(X_frame)
    if len(X_face_locations) == 0:
        return []

    faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)
    predictions = []

    for face_encoding, face_location in zip(faces_encodings, X_face_locations):
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        sorted_indices = np.argsort(distances)[:n_neighbors]   
        sorted_distances = distances[sorted_indices]
        closest_names = [known_names[i] for i in sorted_indices]

        votes = {}
        scores = {}
        for i, name in enumerate(closest_names):
            if sorted_distances[i] <= distance_threshold:
                votes[name] = votes.get(name, 0) + 1
                if name not in scores or sorted_distances[i] < scores[name]:
                    scores[name] = sorted_distances[i]

        if votes:
            best_guess = max(votes, key=votes.get)
            score = scores[best_guess]
            predictions.append((f"{best_guess}", face_location))

    return predictions

print("Loading reference images...")
known_encodings, known_names = load_reference_images("knn_examples/train")
print("Reference images loaded!")

foot_keypoints_indices = [0]
track_id_colors = {}
track_id_labels = {}

video_path = "Mobile.mp4"
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('Mobile_output_test.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    annotated_frame = frame.copy()

    img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    predictions = predict_direct(img, known_encodings, known_names, n_neighbors=4)

    face_boxes = []
    face_labels = []
    for name, (top, right, bottom, left) in predictions:
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        face_boxes.append((left, top, right, bottom))
        face_labels.append(name)
        if name is not None:
            color = (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            label = name
            font_scale = 0.6
            font_thickness = 1
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_w, text_h = text_size
            cv2.rectangle(frame, (left, bottom - text_h - 10), (left + text_w + 10, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 5, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    pose_results = pose_model.track(frame, conf=0.5, verbose=False, persist=True)

    for result in pose_results:
        if result.boxes is not None:
            track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else [-1] * len(result.boxes)
            keypoints_data = result.keypoints.data.cpu().numpy() if result.keypoints is not None else []

            for box, keypoints, track_id in zip(result.boxes, keypoints_data, track_ids):
                if track_id == -1:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())

                for i in foot_keypoints_indices:
                    x, y, conf = keypoints[i]
                    if conf > 0.5:
                        # cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)
                        extended_line_start = (int(x), int(y))
                        extended_line_end = (int(x), int(y + 20))
                        # cv2.line(frame, extended_line_start, extended_line_end, (255, 255, 0), 2)

                        for (left, top, right, bottom), face_label in zip(face_boxes, face_labels):
                            if left <= x <= right and top <= y <= bottom:
                                track_id_labels[track_id] = face_label if face_label else track_id_labels.get(track_id)

                box_color = track_id_colors.get(track_id, (0, 255, 0))
                label = track_id_labels.get(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("Pose Estimation", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
