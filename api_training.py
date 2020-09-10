from flask import Flask, request, jsonify 
from flask_cors import CORS, cross_origin 
import cv2
import os
import numpy as np

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
app = Flask(__name__)
CORS(app, support_credentials=True)

lbp_face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml') 
haar_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# ---------------------------------------------
def face_detection(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = haar_classifier.detectMultiScale(image_gray)
    if (len(faces) > 1) | (len(faces) ==0):
        return "not face"
    else:
        return image_gray

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
            face = face_detection(training_image)
            if (face == "not face"):
                os.remove(image_path)
                print("not face")
            else:
                faces.append(face)
                labels.append(label)
            # try: 
            #     face, bounding_box = face_detection(training_image)
            #     faces.append(face)
            #     labels.append(label)
            # except:
            #     # os.remove(image_path)
            #     print('image err')
    return faces, labels

def transfer_datas(data_path):
    folders = os.listdir(data_path)
    labels = []
    faces = []
    for folder in folders:
        label = int(folder)
        training_images_path = data_path + "/" + folder
        for image in os.listdir(training_images_path):
            image_path = training_images_path + '/' + image 
            print("path: ", image_path)
          
            img = cv2.imread(image_path)
            height, width, channels = img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            try:
                faces = lbp_face_cascade.detectMultiScale(gray)
            except:
                # os.remove(image_path)
                print('image err')

            print("len faces: ", len(faces))
            if (len(faces) > 1) | (len(faces) ==0):
                print("multi face")
                os.remove(image_path)
                continue

            for i, (x,y,w,h) in enumerate(faces):
                # if(i > 0):       
                #     image_path_multi = image_path.split('.')[0] + str(i) + "." + image_path.split('.')[1]
                #     cv2.imwrite(image_path_multi, gray)
                #     img_multi = cv2.imread(image_path_multi)
                #     height, width, channels = img_multi.shape
                #     gray_multi = cv2.cvtColor(img_multi, cv2.COLOR_BGR2GRAY)
                #     faces_multi = haar_classifier.detectMultiScale(gray_multi)
                #     if (len(faces_multi) > 1) | (len(faces_multi) ==0):
                #         print("multi face")
                #         os.remove(image_path_multi)
                #     continue

                print("image path: ", image_path)
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                if (height == 2592) & (width == 1944):
                    cv2.imwrite(image_path, gray[y:y+h,x:x+w])
                else:
                    cv2.imwrite(image_path, gray)
                
    return faces, labels
# ---------------------------------------------
@app.route('/training_from', methods=['GET', 'POST'])
def training_from():
    if request.method == 'POST':
        print("training data")
        faces, labels = prepare_data('training')
        print('Total faces = ', faces)
        print('Total labels = ', labels)

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(labels))
        recognizer.write('trainer.yml') 

        return jsonify("Training from student")
# ---------------------------------------------
@app.route('/training_froms', methods=['GET', 'POST'])
def training_froms():
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
                    # try: 
                    #     face, bounding_box = face_detection(training_image)
                    #     faces.append(face)
                    #     labels.append(label)
                    # except:
                    #     os.remove(image_path)
                    #     print('image err')
                    face = face_detection(training_image)
                    if (face == "not face"):
                        os.remove(image_path)
                        print("not face")
                    else:
                        faces.append(face)
                        labels.append(label)

        print('Total faces = ', faces)
        print('Total labels = ', labels)

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(labels))
        recognizer.save('trainer.yml') 

        return jsonify("Training from student")
# ---------------------------------------------
@app.route('/training', methods=['GET', 'POST'])
def training():
    if request.method == 'GET':
        print("training data")
        faces, labels = prepare_data('training')
        print('Total faces = ', faces)
        print('Total labels = ', labels)

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(labels))
        recognizer.write('trainer.yml') 

        return jsonify("Training all student")

# ---------------------------------------------
@app.route('/transfer_data', methods=['GET', 'POST'])
def transfer_data():
    if request.method == 'GET':
        faces, labels = transfer_datas('training')
        return jsonify("Transfer Data")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)