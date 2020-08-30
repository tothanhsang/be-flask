from flask import Flask, request, jsonify 
from flask_cors import CORS, cross_origin 
import cv2
import os
import numpy as np

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
app = Flask(__name__)
CORS(app, support_credentials=True)

def face_detection(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haar_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face = haar_classifier.detectMultiScale(image_gray, scaleFactor=1.3, minNeighbors=7)
    (x,y,w,h) = face[0]
    return image_gray[y:y+w, x:x+h], face[0]

def prepare_data(data_path):
    folders = os.listdir(data_path)
    labels = []
    faces = []
    for folder in folders:
        label = int(folder)
        training_images_path = data_path + "/" + folder
        for image in os.listdir(training_images_path):
            image_path = training_images_path + '/' + image 
            training_image = cv2.imread(image_path)
            print("image path: ", image_path)
            try: 
                face, bounding_box = face_detection(training_image)
                faces.append(face)
                labels.append(label)
            except:
                os.remove(image_path)
                print('image err')
    return faces, labels


# ---------------------------------------------
@app.route('/training', methods=['GET', 'POST'])
def training():
    if request.method == 'POST':
        json_data = request.get_json()
        print("training data", json_data['listStudent'])

        folders = os.listdir('training')
        labels = []
        faces = []
        for folder in folders:
            label = int(folder)
            if folder in json_data['listStudent']:
                training_images_path = 'training' + "/" + folder
                for image in os.listdir(training_images_path):
                    image_path = training_images_path + '/' + image 
                    print('image: ', image_path)
                    training_image = cv2.imread(image_path)
                    try: 
                        face, bounding_box = face_detection(training_image)
                        faces.append(face)
                        labels.append(label)
                    except:
                        os.remove(image_path)

        print('Total faces = ', faces)
        print('Total labels = ', labels)

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(labels))
        recognizer.write('trainer.yml') 

        return jsonify("Training Data")
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)