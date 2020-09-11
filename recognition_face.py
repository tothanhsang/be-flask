import cv2
import os
import numpy as np
import time

lbp_face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml') 
haar_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_faces(f_cascade, colored_img, scaleFactor = 1.2):
    gray = cv2.cvtColor(colored_img, cv2.COLOR_BGR2GRAY) 
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);   
    print("faces: ", faces)
    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return gray[y:y + h, x:x + w]

def face_detection(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = haar_classifier.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=5)
    (x,y,w,h) = face[0]
    image_gray= cv2.resize(image_gray[y:y+w, x:x+h], (200, 200), interpolation = cv2.INTER_AREA)
    return image_gray, face[0]


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
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = haar_classifier.detectMultiScale(gray, 1.2, 5)
            if (len(faces) > 1):
                os.remove(image_path)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.imwrite(image_path, gray[y:y+h,x:x+w])
    return faces, labels
# transfer_datas("image_emotion")


# faces, labels = prepare_data('image_emotion')
# print('Total faces = ', faces)
# print('Total labels = ', labels)

# # EigenFaces
# #face_recognizer = cv2.face.createEigenFaceRecognizer()
# face_recognizer_eigen = cv2.face.EigenFaceRecognizer_create()
# face_recognizer_eigen.train(faces, np.array(labels))
# face_recognizer_eigen.write('face_recognizer_eigen.yml') 

# # EigenFaces
# #face_recognizer = cv2.face.createFisherFaceRecognizer()
# face_recognizer_fisher = cv2.face.FisherFaceRecognizer_create()
# face_recognizer_fisher.train(faces, np.array(labels))
# face_recognizer_fisher.write('face_recognizer_fisher.yml') 

# # LBPH
# # face_recognizer_lbph = cv2.face.createLBPHFaceRecognizer()
# face_recognizer_lbph = cv2.face.LBPHFaceRecognizer_create()
# face_recognizer_lbph.train(faces, np.array(labels))
# face_recognizer_lbph.write('training_emotion.yml') 

recognizer = cv2.face.LBPHFaceRecognizer_create()
# # recognizer = cv2.face.EigenFaceRecognizer_create()
# recognizer = cv2.face.FisherFaceRecognizer_create()
recognizer.read('trainer.yml')

test =cv2.imread("./FaceRecognition/15.jpg")
# lbp_detected_img = detect_faces(lbp_face_cascade, test)
# predict_result = recognizer.predict(lbp_detected_img)
# print(predict_result)
gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
faces = lbp_face_cascade.detectMultiScale(gray)
if(len(faces) == 0):
    print("not exist face") 
for (x, y, w, h) in faces:
    cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.namedWindow("window", cv2.WINDOW_AUTOSIZE) 
    cv2.resizeWindow("window",1366,768)
    cv2.imshow('window', gray[y:y + h, x:x + w])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # gray= cv2.resize(gray[y:y+w, x:x+h], (200, 200), interpolation = cv2.INTER_AREA)
    # predict_result = recognizer.predict(gray)
    predict_result = recognizer.predict(gray[y:y + h, x:x + w])
    print(predict_result)