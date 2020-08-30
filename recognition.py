import face_recognition
import cv2
import numpy as np

# # video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("assets/img/users/sang.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("assets/img/users/kiet.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

long_image = face_recognition.load_image_file("assets/img/users/long.jpg")
long_face_encoding = face_recognition.face_encodings(long_image)[0]

hieu_image = face_recognition.load_image_file("assets/img/users/hieu.jpg")
hieu_face_encoding = face_recognition.face_encodings(hieu_image)[0]

thuan_image = face_recognition.load_image_file("assets/img/users/thuan.jpg")
thuan_face_encoding = face_recognition.face_encodings(thuan_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    long_face_encoding,
    hieu_face_encoding,
    thuan_face_encoding
]
known_face_names = [
    "Sang",
    "Kiet",
    "Pham",
    "Hieu",
    "Thuan"
]

# known_image = face_recognition.load_image_file("assets/img/users/biden2.jpg")
unknown_image = face_recognition.load_image_file("assets/img/users/sang.jpg")

# biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces(known_face_encodings, unknown_encoding)

print('results: ', results)

face_distances = face_recognition.face_distance(known_face_encodings, unknown_encoding)
best_match_index = np.argmin(face_distances)
print('best: ', best_match_index)
