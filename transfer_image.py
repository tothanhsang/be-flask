import numpy as np
import cv2
from datetime import datetime
import os 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread('./training/16130334/1025190.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    print('x: ', x)
    print('y: ', y)
    print('w: ', w)
    print('h: ', h)
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    # cv2.imwrite('image.jpg', gray[y:y+h,x:x+w])

    filename = str(datetime.now()).split(" ")[1]
    name = filename.split(".")[0]
    namefile = name.split(":")[0] + name.split(":")[1] + name.split(":")[2] + name[1] + '.jpg'

    cwd = os.getcwd()
    print(cwd)
    dir = os.path.join(cwd + "\\trainings\\" , "16130546")
    if not os.path.exists(dir):
        os.mkdir(dir)

    list = os.listdir(dir) 
    number_files = len(list)

    if number_files >= 50:
        # return jsonify("full_image")
        print("number_files: ", number_files) 

    image_result = 'trainings/'+ "16130546" +"/" + namefile
    print("image result: ", image_result)
    cv2.imwrite(image_result, gray[y:y+h,x:x+w])

# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()