import cv2
import os
import time
import numpy as np
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

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

def knn_predict_direct(X_frame, known_encodings, known_names, n_neighbors=2, distance_threshold=0.5):
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
        

        # Count votes for each class
        votes = {}
        for i, name in enumerate(closest_names):
            if sorted_distances[i] <= distance_threshold:
                votes[name] = votes.get(name, 0) + 1

        if votes:
            
            best_guess = max(votes, key=votes.get)
            print(best_guess)
            predictions.append((best_guess, face_location))
            
        else:
            predictions.append(("unknown", face_location))

    return predictions

def show_prediction_labels_on_image(frame, predictions):
    for name, (top, right, bottom, left) in predictions:
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        color = (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        label = name if isinstance(name, str) else name.decode("utf-8")
        font_scale = 0.6
        font_thickness = 1
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_w, text_h = text_size

        cv2.rectangle(frame, (left, bottom - text_h - 10), (left + text_w + 10, bottom), color, cv2.FILLED)
        cv2.putText(frame, label, (left + 5, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    return frame

if __name__ == "__main__":
    print("Loading reference images...")
    known_encodings, known_names = load_reference_images("knn_examples/train")
    print("Reference images loaded!")

    cap = cv2.VideoCapture("CCTV_video_cut.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('m2i.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    total_time_cpu = 0
    frame_count_cpu = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count_cpu += 1
        start_time_cpu = time.time()
        img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        predictions = knn_predict_direct(img, known_encodings, known_names, n_neighbors=2)

        frame = show_prediction_labels_on_image(frame, predictions)
        end_time_cpu = time.time()
        total_time_cpu += end_time_cpu - start_time_cpu
        out.write(frame)

        cv2.imshow('camera', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Total inference time for {frame_count_cpu} frames: {total_time_cpu:.2f} seconds")
    print(f"Average inference speed: {total_time_cpu / frame_count_cpu:.4f} seconds per frame")
