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
face_detector = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

def face_detection(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haar_classifier = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
    face = haar_classifier.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=5)
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
@app.route('/attendences', methods = ["POST"])
def attendances():
    if request.method == 'POST':
        collection = db['attendances']

        # attendence_check = {'course': "123123", 'teacherID': "456456",
        #     "times": datetime.now().strftime('%Y/%m/%d')}

        attendence_data = {
            # '_id': "12345678sdfg",
            'attendanceID': "1234", 
            'course': "1234",
            "times": "000", 
            'students': ["123456"]
        }

        print(attendence_data)

        # attendence_data = {'attendance_id': '1123551000', 'course': '1597865496895E6ZJo', 'times': '1599588000000'}

        # if ("542" in collection.find(attendence_check)[0]['students']):
        #     print('true')

        # collection.update( attendence_check, {'$push': {'students': "12345"}})
        # return jsonify('test')

        # date_now = str(datetime.now()).split(" ")[1]
        # date_now_split = date_now.split(".")[0]
        # attendance_id = date_now_split.split(":")[0] + date_now_split.split(":")[1] + date_now_split.split(":")[2] + date_now_split[1]

        # attendence_data = {
        #     'attendance_id': attendance_id, 'course': "123123", 'teacherID': "456456",
        #     "times": datetime.now().strftime('%Y/%m/%d'), 'students': ['123123']
        # }

        collection.insert_one(attendence_data)

        return jsonify('success')

# ---------------------------------------------
@app.route('/coures', methods = ['GET'])
def course():
    if request.method == 'GET':
        # print("course")
        collection = db['courses']
        cursor = collection.find({})
        list_course = []
        for document in cursor:
            hour = document['startTime'].split(":")[0]+"h"+document['startTime'].split(":")[1]
            weekDays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            # print("weekly: ", document)
            d = {"courseID": document['courseID'], "name": document['name'] + "( " + hour + "-" + weekDays[document['weekly']-1] + " )"+"Nguyen Thi Phuong Tram"}
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
            s = {
                "studentID": student['studentID'], 
                "name": "MSSV: "+student['studentID']+"-"+student['name'], 
                'names': student['name']
            }

            list_student.append(s)
        return {'data': list_student}

# ---------------------------------------------
@app.route('/saveimagegray', methods=['POST'])
def saveimagegray():
    if request.method == 'POST':
        json_data = request.get_json()
        input_file = open('save.txt', "w")
        input_file.write(json_data['img'])
        input_file = open('save.txt', 'r')
        coded_string = input_file.read()
        image_64_decode = base64.b64decode(coded_string)
    
        image_save = open("save.jpg" , 'wb') 
        image_save.write(image_64_decode)

        image = cv2.imread("save.jpg")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray)
        if(len(faces) == 1):
            for (x,y,w,h) in faces:
                try:
                    # cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

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

                    image_result = 'training/'+ json_data['id'] +"/" + namefile
                    # image_result = "training/%s/%s".format(json_data['id'],namefile)
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
            dir = os.path.join(cwd + "\\training_old\\" , json_data['id'])
            if not os.path.exists(dir):
                os.mkdir(dir)

            list = os.listdir(dir) 
            number_files = len(list)

            if number_files >= 50:
                return jsonify("full_image")
            print("number_files: ", number_files) 

            image_result = open('training_old/'+ json_data['id'] +"/" + namefile, 'wb') 
            image_result.write(image_64_decode)
        except:
            print('no save image', json_data['id'])
            pass

        return jsonify("Take picture")
# ---------------------------------------------
@app.route('/recognition', methods=['GET', 'POST'])
def get_recognition():
    if request.method == 'POST':
        i = 0 
        json_data = request.get_json()
        print('post', json_data['id'])

        input_file = open('input.txt', "w")
        input_file.write(json_data['img'])
        input_file = open('input.txt', 'r')
        coded_string = input_file.read()
        image_64_decode = base64.b64decode(coded_string)

        image_result = open("decode.jpg" , 'wb') 
        image_result.write(image_64_decode)

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer.yml')

        test1 =cv2.imread("decode.jpg")
        result = ""
        try:
            gray = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(test1, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imwrite("decode.jpg", gray[y:y+h,x:x+w])
                # break
                predict_result = recognizer.predict(gray[y:y + h, x:x + w])
                # predict_result = [16130546, 9]
                print(predict_result)
                if (predict_result[1] > 8):
                    return jsonify('-')
                else:
                    print("123456")
                    collection = db['attendances']
                    times = datetime.now().strftime('%Y/%m/%d') + ' 1:0'
                    times = datetime.strptime(times, '%Y/%m/%d %H:%M')
                    timestamp = str(datetime.timestamp(times)).split(".")[0] + "000"
                    attendence_check = {'course': json_data['id'], "times": timestamp}
                    count_row = collection.find(attendence_check).count()

                    collection_student = db['students']
                    student_recognition = collection_student.find_one({'studentID': str(predict_result[0])})

                    # result = result + "MSSV: "+student_recognition['studentID']+" - " + student_recognition['name']
                    print(result)

                    if count_row == 0:
                        print("Th1")
                        # get Gourse
                        collection_course = db['courses']
                        course_check = {'courseID': json_data['id']}
                        course = collection_course.find_one(course_check)
                    
                        # check date
                        start_date = datetime.strptime(course['startDate'], '%Y/%m/%d')
                        end_date = datetime.strptime(course['endDate'], '%Y/%m/%d')

                        if ((start_date < datetime.now()) & (end_date > datetime.now())) == False:
                            return jsonify("This course is outdate")

                        # Check Week day and hour
                        start_time = datetime.now().strftime('%Y/%m/%d') + " " + course['startTime']
                        start_time = datetime.strptime(start_time, '%Y/%m/%d %H:%M')
                        end_time = datetime.now().strftime('%Y/%m/%d') + " " + course['endTime']
                        end_time = datetime.strptime(end_time, '%Y/%m/%d %H:%M')

                        week_day = datetime.now().weekday()
                        if (week_day==6):
                            week_day = 0
                        else:
                            week_day = week_day + 1

                        # if ((start_time < datetime.now()) & (end_time > datetime.now()) & (week_day == course['weekday'])) == False:
                        #     return jsonify('This course has not taken place yet')

                        # Check Student exist in Course
                        print("course: ", course['students'])
                        if (str(predict_result[0]) in course['students']) == False:
                            return jsonify('This student did not exist in this course')
                        else:
                            result = result + "MSSV: "+student_recognition['studentID']+" - " + student_recognition['name']

                                
                        # filename = str(datetime.now()).split(" ")[1]
                        # name = filename.split(".")[0]
                        # namefile = name.split(":")[0] + name.split(":")[1] + name.split(":")[2] + name[1] + '.jpg'


                        date_now = str(datetime.now()).split(" ")[1]
                        date_now_split = date_now.split(".")[0]
                        attendance_id = date_now_split.split(":")[0] + date_now_split.split(":")[1] + date_now_split.split(":")[2] + date_now_split[1]

                        print("attendance_id: ", attendance_id)

                        times = datetime.now().strftime('%Y/%m/%d') + ' 1:0'
                        times = datetime.strptime(times, '%Y/%m/%d %H:%M')
                        timestamp = str(datetime.timestamp(times)).split(".")[0] + "000"

                        attendence_data = {
                            'attendanceID': attendance_id, 
                            'course': json_data['id'],
                            "times": timestamp, 
                            'students': [predict_result[0]]
                        }
                        print("attendence_data: ", attendence_data)
                        collection.insert_one(attendence_data)
                    else:
                        print("Th2")
                        # get Gourse
                        collection_course = db['courses']
                        course_check = {'courseID': json_data['id']}
                        course = collection_course.find_one(course_check)
                    
                        # check date
                        start_date = datetime.strptime(course['startDate'], '%Y/%m/%d')
                        end_date = datetime.strptime(course['endDate'], '%Y/%m/%d')
                        if ((start_date < datetime.now()) & (end_date > datetime.now())) == False:
                            return jsonify("This course is outdate")

                        # Check Week day and hour
                        start_time = datetime.now().strftime('%Y/%m/%d') + " " + course['startTime']
                        start_time = datetime.strptime(start_time, '%Y/%m/%d %H:%M')
                        end_time = datetime.now().strftime('%Y/%m/%d') + " " + course['endTime']
                        end_time = datetime.strptime(end_time, '%Y/%m/%d %H:%M')

                        week_day = datetime.now().weekday()
                        if (week_day==6):
                            week_day = 0
                        else:
                            week_day = week_day + 1

                        # if ((start_time < datetime.now()) & (end_time > datetime.now()) & (week_day == course['weekday'])) == False:
                        #     return jsonify('This course has not taken place yet')

                        if (str(predict_result[0]) in course['students']) == False:
                            return jsonify('This student did not exist in this course')
                        else:
                            result = result + "MSSV: "+student_recognition['studentID']+" - " + student_recognition['name']
                            # add student
                            if (predict_result[0] in collection.find(attendence_check)[0]['students']) == False:
                                print("This student is atendanced")
                                collection.update( attendence_check, {'$push': {'students': predict_result[0]}})
        except ValueError:
            print(ValueError)
            return jsonify("+")

 
        # collection_student = db['students']
        # # student_recognition = collection_student.find_one({'studentID': str(predict_result[0])})
        # student_recognition = collection_student.find_one({'studentID': "16130546"})

        # return jsonify("MSSV: "+student_recognition['studentID']+" - " + student_recognition['name'])
        print("return: ", result)
        return jsonify(result)

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