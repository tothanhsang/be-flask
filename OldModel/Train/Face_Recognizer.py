import cv2


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('../Data/trainer.yml')
detector = cv2.CascadeClassifier('../Data/haarcascade_frontalface_default.xml')
font_cv2 = cv2.FONT_HERSHEY_SIMPLEX

ids = 0

names = ['None', 'hieu', 'thuan', 'sang', 'elsa', 'None', 'hieu', 'thuan', 'sang', 'elsa']

# Initialize and start realtime video capture
cam = cv2.VideoCapture("http://192.168.20.107:8080/video")
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ids, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        print('ids: ', ids)
        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence <= 25):
            ids = names[ids]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            ids = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))


        cv2.putText(img, str(id), (x + 5, y - 5), font_cv2, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font_cv2, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)
    if cv2.waitKey(100) & 0XFF == ord('q') :
        break

print("\n [INFO] Exiting Program")
cam.release()
cv2.destroyAllWindows()