import cv2
import time
from PIL import Image

# image = Image.open('./FaceDetect/14.jpg')
# # image.show()
# new_image = image.resize((700, 933))
# image.show()

def convertToRGB(img): 
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detect_faces(f_cascade, colored_img, scaleFactor = 1.2):
    gray = cv2.cvtColor(colored_img, cv2.COLOR_BGR2GRAY) 
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);   
    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return gray

# # # Haar Classifier
# haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
# # lbp_face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')  
# test1 = cv2.imread('./FaceDetect/1.jpg')
# test1 = cv2.resize(test1, (700, 933), interpolation = cv2.INTER_AREA)
# faces_detected_img = detect_faces(haar_face_cascade, test1)
# cv2.namedWindow("window", cv2.WINDOW_AUTOSIZE) 
# cv2.resizeWindow("window",700,950)
# # cv2.imshow('window', convertToRGB(faces_detected_img))
# cv2.imshow('window', faces_detected_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # LBP Classifier
# lbp_face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')  
# test2 = cv2.imread('./FaceDetect/1.jpg')
# test2 = cv2.resize(test2, (700, 933), interpolation = cv2.INTER_AREA)
# faces_detected_img = detect_faces(lbp_face_cascade, test2)
# cv2.namedWindow("window", cv2.WINDOW_AUTOSIZE) 
# cv2.resizeWindow("window",1366,768)
# cv2.imshow('window', faces_detected_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# #------------HAAR-----------
# t1 = time.time()
# haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
# test1 = cv2.imread('./FaceDetect/6.jpg')
# test1 = cv2.resize(test1, (700, 933), interpolation = cv2.INTER_AREA)
# haar_detected_img = detect_faces(haar_face_cascade, test1)
# t2 = time.time()
# dt1 = t2 - t1
# print("dt1: ", dt1)
# cv2.namedWindow("window", cv2.WINDOW_AUTOSIZE) 
# # cv2.resizeWindow("window",1366,768)
# cv2.resizeWindow("window",700,950)
# # cv2.imshow('window', convertToRGB(faces_detected_img))
# cv2.imshow('window', haar_detected_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#------------LBP-----------
t1 = time.time()
lbp_face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')  
test2 = cv2.imread('./FaceDetect/6.jpg')
test2 = cv2.resize(test2, (700, 933), interpolation = cv2.INTER_AREA)
lbp_detected_img = detect_faces(lbp_face_cascade, test2)
t2 = time.time()
dt2 = t2 - t1
print("dt2: ", dt2)
cv2.namedWindow("window", cv2.WINDOW_AUTOSIZE) 
cv2.resizeWindow("window",1366,768)
cv2.imshow('window', lbp_detected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()