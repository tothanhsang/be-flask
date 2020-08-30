import cv2
import os
from datetime import datetime

cam = cv2.VideoCapture("http://192.168.20.107:8080/video")
face_detector = cv2.CascadeClassifier('../Data/haarcascade_frontalface_default.xml')

face_id = input('\nEnter user id  ')
path = '../Data/Image'
def existIdentifier(face_ids) :
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    id_map = {}
    for imagePath in imagePaths:
        ids = int(os.path.split(imagePath)[-1].split(".")[1])
        id_map[str(ids)] = ids
    return face_ids in id_map


count = 0
if  not existIdentifier(face_id):
    while(True):
        ret, img = cam.read(2)
        # Convert image color to Gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            count += 1

            img_name = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
            img_name = os.path.join(os.getcwd(), 'images', ".temp", img_name + '.jpg')
            
            if count<=30 :
                cv2.imwrite("../Data/Image/User." + str(face_id) + '.' + str(count)+ '.jpg', gray[y:y+h,x:x+w] )
                # cv2.imwrite(img_name, img)
        # if (count > 30):
        #     break

        cv2.imshow('image', img)

        if cv2.waitKey(100) & 0XFF == ord('q'):
            break
else: 
    print('User enter is existed')


print(existIdentifier(face_id))


print("\n [INFO] Exiting Program")
cam.release()
cv2.destroyAllWindows()