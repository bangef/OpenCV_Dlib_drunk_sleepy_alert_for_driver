import cv2
import dlib
import imutils
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import time
from scipy.spatial import distance as dist
import playsound
from env import *

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def sound_alarm():
    # play an alarm sound
    playsound.playsound(alarm_path)

# Detektor untuk mendeteksi satu atau lebih wajah. Kami mengatur fungsi dlib.get_frontal_face_detector di dalam variabel.
face_detector = dlib.get_frontal_face_detector()

# Prediktor untuk mendeteksi titik kunci dari wajah. Kami mengatur fungsi dlib.shape_predictor di dalam variabel. Fungsi ini memerlukan lokasi model yang telah dilatih sebelumnya sebagai parameter, yang dapat Anda unduh di sini.
face_pose_predictor = dlib.shape_predictor('assets/shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# Insialisasi lib VideoStream dan mengaktifkan video
vs = VideoStream(src=0).start()
# menghentikan program selama 1 second
time.sleep(1.0)

while True:
    # mulai mencapture gambar
    frame = vs.read()
    # frame diresize agar gambar tidak terlalu high resolution
    frame = imutils.resize(frame, width=frame_width)
    # merubah format RGB ke BGRGRAY
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # mendeteksi gambar, dengan gambar yang ditangkap
    detected_faces = face_detector(gray, 0)
    
    for rect in detected_faces:
        detected_faces = face_pose_predictor(gray, rect)
        detected_faces = face_utils.shape_to_np(detected_faces)
        # menampilkan pola dari landmark point seluruh wajah
        for (x, y) in detected_faces:
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)
        # menangkap posisi kordinat mata kanan dan mata kiri
        leftEye = detected_faces[lStart:lEnd]
        rightEye = detected_faces[rStart:rEnd]
        # menghitung jarak antar point kordinat dengan euclidean distance
        leftEAR = eye_aspect_ratio(leftEye)        
        rightEAR = eye_aspect_ratio(rightEye)
        # ukuran aspect rasio mata gabungan
        ear = (leftEAR + rightEAR) / 2.0
        # kondisi apabila mata kurang dari posisi mata tertutup        
        if ear < EYE_AR_THRESH:
            # counter digunakan sebagai delay mata tertutup
            COUNTER += 1
            # kondisi apabila counter saat ini lebih dari sama dengan batas posisi lama mata tertutup
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # kondisi dimana alarm masih hidup
                if not ALARM_ON:
                    ALARM_ON = True

                    t = Thread(target=sound_alarm)
                    t.deamon = True
                    t.start()

                alertcor=(int(frame.shape[1]/24),70)
                cv2.putText(frame, "BANGUN, BANGUN, BANGUN!!!", alertcor, cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 0), 2)

        else:
            COUNTER = 0
            ALARM_ON = False
       
        EARcor=(int(frame.shape[1]/2-150),25)
        cv2.putText(frame, "Aspek Rasio Saat ini : {:.2f}".format(ear), EARcor,cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
 
    cv2.imshow("Detektor Kantuk", frame)
    # event ketika tombol "q" pada keyboard ditekan akan menghentikan perulangan
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
# menghentikan program video 
cv2.destroyAllWindows()
vs.stop()