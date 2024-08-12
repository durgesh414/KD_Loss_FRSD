import os
import cv2
import dlib
import numpy as np

# Initialize dlib's face detector (HOG-based) and create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Update this path if needed

def align_face(image, gray, rect):
    shape = predictor(gray, rect)
    shape = np.array([[p.x, p.y] for p in shape.parts()])
    
    left_eye = shape[36:42]
    right_eye = shape[42:48]
    
    left_eye_center = left_eye.mean(axis=0).astype(int)
    right_eye_center = right_eye.mean(axis=0).astype(int)
    
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    eyes_center = (int((left_eye_center[0] + right_eye_center[0]) // 2), 
                   int((left_eye_center[1] + right_eye_center[1]) // 2))
    
    M = cv2.getRotationMatrix2D((eyes_center[0], eyes_center[1]), angle, 1)
    output = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)
    
    return output

def process_image(image_path, dest_dir):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    save_path = os.path.join(dest_dir, os.path.relpath(image_path, start=dataset_path))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if len(rects) == 1:
        aligned_image = align_face(image, gray, rects[0])
        cv2.imwrite(save_path, aligned_image)
    else:
        cv2.imwrite(save_path, image)  # Save the original image if no or multiple faces detected

    return 1 if len(rects) == 1 else 0

def process_dataset(dataset_path, aligned_dataset_path):
    total_processed = 0
    total_images = 0

    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                total_images += 1
                image_paths.append(os.path.join(root, file))

    for path in image_paths:
        total_processed += process_image(path, aligned_dataset_path)

    print(f"Total Images: {total_images}")
    print(f"Successfully Processed: {total_processed}")
    print(f"Failed to Process: {total_images - total_processed}")

dataset_path = '/home/kaushik/Durgesh/repo/SphereFace2/scripts/data/train/vggface2_train_112x112'
aligned_dataset_path = '/home/kaushik/Durgesh/repo/SphereFace2/scripts/data/train/vggface2_train_aligned_112x112'
process_dataset(dataset_path, aligned_dataset_path)