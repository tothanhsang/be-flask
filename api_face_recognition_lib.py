 from flask import Flask, request, jsonify 
from flask_cors import CORS, cross_origin 
import os 
import psycopg2 
import cv2 
import numpy as np 
import re 
import base64
import face_recognition
import numpy as np

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
app = Flask(__name__)
CORS(app, support_credentials=True)

# ------------------------------------------------
# Test Database postgre
# @app.route('/test', methods=['GET'])
# def test():
#     if request.method == 'GET':
#         json_data = request.get_json()
#         print('json data: ', json_data)
#         try:
#             connection = psycopg2.connect(
#                 user='postgres',
#                 password='123123',
#                 host='localhost',
#                 database='test_flask'
#             )
#             cursor = connection.cursor()
#             print('cursor: ', cursor)
#             connection.commit() 
#         except (Exception, psycopg2.DatabaseError) as error:
#             print('ERROR DB: ', error)
#         # finally:
#         #     connection.commit()
#         #     if connection:
#         #         cursor.close()
#         #         connection.close()
#         #         print('PostgreSQL connection is closed')
#         return jsonify(json_data)

# ---------------------------------------------
@app.route('/recognition', methods=['GET', 'POST'])
def get_recognition():
    if request.method == 'POST':
        json_data = request.get_json()

        input_file = open('input.txt', "w")
        input_file.write(json_data['img'])
        input_file = open('input.txt', 'r')
        coded_string = input_file.read()
        image_64_decode = base64.b64decode(coded_string)
        image_result = open('decode.jpg', 'wb') 
        image_result.write(image_64_decode)
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
            # "Pham",
            # "Hieu",
            # "Thuan"
        ]

        unknown_image = face_recognition.load_image_file("assets/img/users/sang.jpg")
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
        # if len(unknown_encoding) > 0:
        #     unknown_encoding = unknown_encoding[0]
        # else:
        #     return jsonify("No faces found in the image!")
        results = face_recognition.compare_faces(known_face_encodings, unknown_encoding)
        print('results: ', results)
        face_distances = face_recognition.face_distance(known_face_encodings, unknown_encoding)
        best_match_index = np.argmin(face_distances)
        print('best: ', known_face_names[best_match_index])

        return jsonify(known_face_names[best_match_index])

if __name__ == '__main__':
    app.run(host='192.168.0.118', port=5000, debug=True)