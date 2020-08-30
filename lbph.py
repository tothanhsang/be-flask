import numpy as np
import cv2 
import os 

# creating database
database = ["Sang", "Kiet"]
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
            face, bounding_box = face_detection(training_image)
            faces.append(face)
            labels.append(label)
    return faces, labels

faces, labels = prepare_data('training')
print('Total faces = ', faces)
print('Total labels = ', labels)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.write('trainer.yml') 

# test1 =cv2.imread("training/16130111/1.jpg")
# face_test, bounding_box = face_detection(test1)
# print(recognizer.predict(face_test))

# def predict_image(test_image):
#     print('tst image: ', test_image)
#     img = test_image.copy()
#     face, bounding_box = face_detection(img)
#     label = recognizer.predict(face)
#     label_text = database[label-1]
#     print(label)
#     print(label_text)
#     (x,y,w,h) = bounding_box
#     cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
#     cv2.putText(img, label_text, (x,y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
#     return img 

# test1 =cv2.imread("training/1/hieu.jpg")
# predict1 = predict_image(test1)
# cv2.imshow('Face Recognition', predict1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()