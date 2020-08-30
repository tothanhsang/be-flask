from flask import Flask, request, jsonify 
from flask_cors import CORS, cross_origin 
import os 
import psycopg2 
import cv2 
import numpy as np 
import re 
import base64
import numpy as np
import pymongo
from datetime import datetime
from PIL import Image

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
app = Flask(__name__)
CORS(app, support_credentials=True)
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
db = myclient["college_attendance_system"]
# mycol = mydb["customers"]

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
@app.route('/flask', methods = ['GET', 'POST'])
def flask():
    if request.method == 'POST':
        json_data = request.get_json()
        mycol = db["flask"]
        mydict = { "name": "John", "address": "Highway 37" }
        mydoc = mycol.find(mydict).count()
        if mydoc == 0:
            x = mycol.insert_one(mydict)
            print(x.inserted_id)
            return jsonify("success !")
        else:
            return jsonify("Objects exists")

# ---------------------------------------------
@app.route('/attendence', methods = ["POST"])
def attendance():
    if request.method == 'POST':
        collection = db['attendence']

        attendence_check = {'course': "123123", 'teacherID': "456456",
            "times": datetime.now().strftime('%Y/%m/%d')}

        if ("542" in collection.find(attendence_check)[0]['students']):
            print('true')

        collection.update( attendence_check, {'$push': {'students': "12345"}})
        return jsonify('test')

        date_now = str(datetime.now()).split(" ")[1]
        date_now_split = date_now.split(".")[0]
        attendance_id = date_now_split.split(":")[0] + date_now_split.split(":")[1] + date_now_split.split(":")[2] + date_now_split[1]

        attendence_data = {
            'attendance_id': attendance_id, 'course': "123123", 'teacherID': "456456",
            "times": datetime.now().strftime('%Y/%m/%d'), 'students': ['123123']
        }

        collection.insert_one(attendence_data)

        return jsonify('success')

# ---------------------------------------------
@app.route('/coures', methods = ['GET'])
def course():
    if request.method == 'GET':
        collection = db['courses']
        cursor = collection.find({})

        list_course = []
        for document in cursor:
            hour = document['startTime'].split(":")[0]+"h"+document['startTime'].split(":")[1]
            weekDays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            # print(document['weekday'])
            d = {"courseID": document['courseID'], "name": document['name'] + "( " + hour + "-" + weekDays[document['weekday']-1] + " )"+"Nguyen Thi Phuong Tram"}
            list_course.append(d)
        return {"data": list_course}
# ---------------------------------------------
@app.route('/students', methods = ['GET'])
def students():
    if request.method == 'GET':
        collection = db['students']
        cursor = collection.find({})
        list_student = []
        for student in cursor:
            s = {"studentID": student['studentID'], "name": "MSSV: "+student['studentID']+"-"+student['name'], 'names': student['name']}
            list_student.append(s)
        return {'data': list_student}

# ---------------------------------------------
@app.route('/training', methods=['GET', 'POST'])
def training():
    if request.method == 'GET':
        print("training data")
        faces, labels = prepare_data('trainings')
        print('Total faces = ', faces)
        print('Total labels = ', labels)

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(labels))
        recognizer.write('trainer.yml') 

        return jsonify("Training Data")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
@app.route('/saveimagegray', methods=['POST'])
def saveimagegray():
    if request.method == 'POST':
        json_data = request.get_json()
        input_file = open('gray.txt', "w")
        input_file.write(json_data['img'])
        input_file = open('gray.txt', 'r')
        coded_string = input_file.read()
        image_64_decode = base64.b64decode(coded_string)
    
        image_save = open("gray.jpg" , 'wb') 
        image_save.write(image_64_decode)

        image = cv2.imread("gray.jpg")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,)
        for (x,y,w,h) in faces:
            try:
                # cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                filename = str(datetime.now()).split(" ")[1]
                name = filename.split(".")[0]
                namefile = name.split(":")[0] + name.split(":")[1] + name.split(":")[2] + name[1] + '.jpg'

                cwd = os.getcwd()
                print(cwd)
                dir = os.path.join(cwd + "\\traininggray\\" , json_data['id'])
                if not os.path.exists(dir):
                    os.mkdir(dir)

                list = os.listdir(dir) 
                number_files = len(list)

                if number_files >= 50:
                    return jsonify("full_image")
                print("number_files: ", number_files) 

                image_result = 'traininggray/'+ json_data['id'] +"/" + namefile
                cv2.imwrite(image_result, gray[y:y+h,x:x+w])
            except:
                # return jsonify("")
                print('no save image', json_data['id'])
                pass
        return jsonify("Take picture")
        
         
# ---------------------------------------------
@app.route('/saveimage', methods=['GET', 'POST'])
def saveimage():
    if request.method == 'POST':
        json_data = request.get_json()
        # return jsonify(json_data)
        # print('post')
        # print('save image', json_data['id'])

        input_file = open('save.txt', "w")
        input_file.write(json_data['img'])
        input_file = open('save.txt', 'r')
        coded_string = input_file.read()
        image_64_decode = base64.b64decode(coded_string)

        image_save = open("save.jpg" , 'wb') 
        image_save.write(image_64_decode)

        check_face =cv2.imread("save.jpg")
        try:
            face_test, bounding_box = face_detection(check_face)
            print('save image', json_data['id'])
       
            filename = str(datetime.now()).split(" ")[1]
            name = filename.split(".")[0]
            namefile = name.split(":")[0] + name.split(":")[1] + name.split(":")[2] + name[1] + '.jpg'

            cwd = os.getcwd()
            print(cwd)
            dir = os.path.join(cwd + "\\training\\" , json_data['id'])
            if not os.path.exists(dir):
                os.mkdir(dir)

            list = os.listdir(dir) 
            number_files = len(list)

            if number_files >= 50:
                return jsonify("full_image")
            print("number_files: ", number_files) 

            image_result = open('training/'+ json_data['id'] +"/" + namefile, 'wb') 
            image_result.write(image_64_decode)
        except:
            # return jsonify("")
            print('no save image', json_data['id'])
            pass


        # faces, labels = prepare_data('training')
        # recognizer = cv2.face.LBPHFaceRecognizer_create()
        # recognizer.train(faces, np.array(labels))
        # recognizer.write("trainer.yml") 

        # test1 =cv2.imread("decode.jpg")

        # try:
        #     face_test, bounding_box = face_detection(test1)
            
        # except:
        #     return jsonify("")

        return jsonify("Take picture")
# ---------------------------------------------
@app.route('/recognition', methods=['GET', 'POST'])
def get_recognition():
    if request.method == 'POST':
        i = 0 
        json_data = request.get_json()
        # return jsonify(json_data)
        # print('post')
        print('post', json_data['id'])

        # return jsonify({"number": "123"})
        # image = open('assets/img/users/kiet.jpg', 'rb')
        # image_read = image.read()
        # image_64_encode = base64.encodebytes(image_read)

        input_file = open('input.txt', "w")
        input_file.write(json_data['img'])
        input_file = open('input.txt', 'r')
        coded_string = input_file.read()
        image_64_decode = base64.b64decode(coded_string)

        # filename = str(datetime.now()).split(" ")[1]
        # name = filename.split(".")[0]
        # namefile = name.split(":")[0] + name.split(":")[1] + name.split(":")[2] + name[1] + '.jpg'
        # image_result = open('training/'+ "16130546/" + namefile, 'wb') 

        image_result = open("decode.jpg" , 'wb') 
        image_result.write(image_64_decode)
        
        # i = i+1
        # return jsonify({"number": i})

        # faces, labels = prepare_data('training')
        # recognizer = cv2.face.LBPHFaceRecognizer_create()
        # recognizer.train(faces, np.array(labels))
        # recognizer.write("trainer.yml") 

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer.yml')

        test1 =cv2.imread("decode.jpg")

        # face_test, bounding_box = face_detection(test1)
        try:
            gray = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(test1, (x, y), (x + w, y + h), (0, 255, 0), 2)
                predict_result = recognizer.predict(gray[y:y + h, x:x + w])

                if (predict_result[1] > 20):
                    return jsonify('------')
                else:
                    # mycol = db["attendance"]
                    # student_attendences = {'student': recognizer.predict(face_test)[0], 'time': datetime.now()}
                    # mycol.insert_one(student_attendences)

                    collection = db['attendence']
                    attendence_check = {'course': json_data['id'], "times": datetime.now().strftime('%Y/%m/%d')}
                    count_row = collection.find(attendence_check).count()

                    # collection_student = db['students']
                    # # student_recognition = collection_student.find_one({'studentID': str(predict_result[0])})
                    # student_recognition = collection_student.find_one({'studentID': "16130546"})
                    print(predict_result[1])
                    # if count_row == 0:
                    #     # get Gourse
                    #     collection_course = db['courses']
                    #     course_check = {'courseID': json_data['id']}
                    #     course = collection_course.find_one(course_check)
                    
                    #     # check date
                    #     start_date = datetime.strptime(course['startDate'], '%Y/%m/%d')
                    #     end_date = datetime.strptime(course['endDate'], '%Y/%m/%d')
                    #     if ((start_date < datetime.now()) & (end_date > datetime.now())):
                    #         return jsonify("This course is outdate")

                    #     # Check Week day and hour
                    #     start_time = datetime.now().strftime('%Y/%m/%d') + " " + course['startTime']
                    #     start_time = datetime.strptime(start_time, '%Y/%m/%d %H:%M')
                    #     end_time = datetime.now().strftime('%Y/%m/%d') + " " + course['endTime']
                    #     end_time = datetime.strptime(end_time, '%Y/%m/%d %H:%M')

                    #     week_day = datetime.now().weekday()
                    #     if (week_day==6):
                    #         week_day = 0
                    #     else:
                    #         week_day = week_day + 1

                    #     if ((start_time < datetime.now()) & (end_time > datetime.now()) & datetime.now() & (week_day == course['weekday'])):
                    #         return jsonify('This course has not taken place yet')

                    #     # Check Student exist in Course
                    #     if (predict_result[0] in course['students']) == False:
                    #         return jsonify('This student did not exist in the course')

                    #     date_now = str(datetime.now()).split(" ")[1]
                    #     date_now_split = date_now.split(".")[0]
                    #     attendance_id = date_now_split.split(":")[0] + date_now_split.split(":")[1] + date_now_split.split(":")[2] + date_now_split[1]
                    #     attendence_data = {
                    #         'attendance_id': attendance_id, 'course': json_data['id'], 'teacherID': "teacheID",
                    #         "times": datetime.now().strftime('%Y/%m/%d'), 'students': [predict_result[0]]
                    #     }
                    #     collection.insert_one(attendence_data)
                    # else:
                    #     if (predict_result[0] in collection.find(attendence_check)[0]['students']) == False:
                    #         collection.update( attendence_check, {'$push': {'students': predict_result[0]}})
        except:
            return jsonify("++++++")

 
        # collection_student = db['students']
        # # student_recognition = collection_student.find_one({'studentID': str(predict_result[0])})
        # student_recognition = collection_student.find_one({'studentID': "16130546"})

        # return jsonify("MSSV: "+student_recognition['studentID']+" - " + student_recognition['name'])
        return jsonify("have data")

# -------------------------------------------------
@app.route('/course_check', methods=["GET"])
def course_check():
    if (request.method == "GET"):
        collection_course = db['courses']
        course_check = {'courseID': '1597865496895E6ZJo'}
        course = collection_course.find_one(course_check)
       
        start_date = datetime.strptime(course['startDate'], '%Y/%m/%d')
        end_date = datetime.strptime(course['endDate'], '%Y/%m/%d')
        print ((start_date < datetime.now()) & (end_date > datetime.now()))

        start_time = datetime.now().strftime('%Y/%m/%d') + ' 1:5'
        start_time = datetime.strptime(start_time, '%Y/%m/%d %H:%M')
        end_time = datetime.now().strftime('%Y/%m/%d') + ' 1:5'
        end_time = datetime.strptime(end_time, '%Y/%m/%d %H:%M')
        print ((start_time < datetime.now()) & (end_time > datetime.now()))

        print("16130546" in course['students'])

        date_time_str = '22/08/20 01:55:19'
        date_time_obj = datetime.strptime(date_time_str, '%d/%m/%y %H:%M:%S')
        print ("The date is", date_time_obj.weekday())
      
    return jsonify("course check")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)