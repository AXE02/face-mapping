import cv2
import numpy as np
import face_recognition
import os
from sklearn.cluster import DBSCAN
from collections import defaultdict
from zipfile import ZipFile

def load_images_from_folder(folder):
    images = []
    images_path = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if img_path.endswith(('.png', '.JPG', '.jpeg')):  # check for valid file types
            img = face_recognition.load_image_file(img_path)
            images.append(img)
            images_path.append(img_path)
    return images, images_path

def extract_faces(images):
    face_encodings = []
    face_locations = []
    for img in images:
        loc = face_recognition.face_locations(img)
        if loc:
            encoding = face_recognition.face_encodings(img, known_face_locations=loc)[0]
            face_encodings.append(encoding)
            face_locations.append(loc[0])  # take the first face found
    return face_encodings, face_locations

def cluster_faces(face_encodings):
    clustering_model = DBSCAN(eps=0.5, min_samples=3, metric='euclidean')
    clustering_model.fit(face_encodings)
    labels = clustering_model.labels_
    return labels

def create_zip_file(images_path, face_id):
    zip_file_name = f'face_{face_id}.zip'
    with ZipFile(zip_file_name, 'w') as zipf:
        for img_path in images_path:
            zipf.write(img_path, os.path.basename(img_path))
    return zip_file_name

def main(google_drive_link):
    images, images_path = download_images_from_drive(google_drive_link)
    face_encodings, face_locations = extract_faces(images)
    
    if not face_encodings:
        print("No faces found in the folder")
        return None
    
    labels = cluster_faces(face_encodings)
    
    # Group images by detected face
    images_by_face = defaultdict(list)
    for label, img_path in zip(labels, images_path):
        images_by_face[label].append(img_path)

    # Function to download images for a given face ID
    def download_images_for_face(face_id):
        if face_id in images_by_face:
            paths = images_by_face[face_id]
            zip_file_name = create_zip_file(paths, face_id)
            print(f"Images downloaded for face cluster ID {face_id}. Zip file created: {zip_file_name}")
        else:
            print(f"No images found for face cluster ID {face_id}")

    return download_images_for_face

def download_images_from_drive(google_drive_link):
    folder_path = r' ' #Idhar user ka search photo
    # Download images from the provided Google Drive link
    # Add your logic to download the images from the Google Drive link (e.g., using gdown)
    # For simplicity, you can use the existing load_images_from_folder function
    images, images_path = load_images_from_folder(folder_path)
    return images, images_path

# Replace 'your_google_drive_link' with the actual Google Drive link provided by the user
google_drive_link = '' #Idhar drive
download_images_for_face = main(google_drive_link)

# To download images for a given face ID (cluster), call the function returned from main
# For example, to download images for the first face cluster:
download_images_for_face(0)
