import numpy as np
import cv2
import face_recognition
import os
from datetime import datetime


path = 'Attendance_Images'
images = []
class_names = []
my_list = os.listdir(path)
print(my_list)


for cls in my_list:
    cur_img = cv2.imread(f'{path}/{cls}')
    images.append(cur_img)
    class_names.append(os.path.splitext(cls)[0])
print(class_names)

def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list


def mark_attendance(name):
    with open('attendance.csv', 'r+') as f:
        my_data_list = f.readlines()
        name_list = []
        for line in my_data_list:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            now = datetime.now()
            date_str = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {date_str}')





encode_list_known = find_encodings(images)
print('Encoding complete')


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_small = cv2.resize(img, (0,0), None, 0.25, 0.25)
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    
    faces_cur_frame = face_recognition.face_locations(img_small)
    encodes_cur_frame = face_recognition.face_encodings(img_small, faces_cur_frame)

    for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
        matches = face_recognition.compare_faces(encode_list_known, encode_face)
        faceDis = face_recognition.face_distance(encode_list_known, encode_face)
        match_index = np.argmin(faceDis)

        if matches[match_index]:
            name = class_names[match_index].upper()
            #print(name)
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4 
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.rectangle(img, (x1, y2-35),(x2, y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            mark_attendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)



#faceLoc = face_recognition.face_locations(imgIbra)[0]
#encodeIbra = face_recognition.face_encodings(imgIbra)[0]
#cv2.rectangle(imgIbra, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255,0,255),2)

#faceLocTest = face_recognition.face_locations(imgTest)[0]
#encodeTest = face_recognition.face_encodings(imgTest)[0]
#cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255,0,255),2)

#results = face_recognition.compare_faces([encodeIbra], encodeTest)
#faceDis = face_recognition.face_distance([encodeIbra], encodeTest)