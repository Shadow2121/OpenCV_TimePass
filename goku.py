import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

x_offset=y_offset=50

count = 1

while 1:
    ret, img = cap.read()
    goku1 = cv2.imread('imgs\goku1.png', -1)
    goku2 = cv2.imread('imgs\goku2.png', -1)
    goku3 = cv2.imread('imgs\goku3.png', -1)
    goku4 = cv2.imread('imgs\goku4.png', -1)
    goku5 = cv2.imread('imgs\goku5.png', -1)
    blure = cv2.imread('imgs\blure.png', -1)
    eye = cv2.imread('imgs\eye.png', -1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        try:
            if count % 5 == 0:
                ####### GOKU 1 #############    
                w = int(w*2)
                h = int(h*2)

                x -= (int(h//3.5))
                y -= (int(w//2.4))   #130

                t_face = cv2.resize(goku1, (w, h))

                alpha_s = t_face[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(0, 3):
                    img[y:y+h, x:x+w, c] = (alpha_s * t_face[:, :, c] + alpha_l * img[y:y+h, x:x+w, c])

            elif count % 5 == 1:
                
                ####### GOKU 2 #############    
                w = int(w*1.5)
                h = int(h*1.9)

                x -= (int(h//7))
                y -= (int(w//1.7))  

                t_face = cv2.resize(goku2, (w, h))

                alpha_s = t_face[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(0, 3):
                    img[y:y+h, x:x+w, c] = (alpha_s * t_face[:, :, c] + alpha_l * img[y:y+h, x:x+w, c])

            elif count % 5 == 2:
            
                ####### GOKU 3 #############    
                w = int(w*1.7)
                h = int(h*1.7)

                x -= (int(h//4.5))
                y -= (int(w//2.6))   #130

                t_face = cv2.resize(goku3, (w, h))

                alpha_s = t_face[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(0, 3):
                    img[y:y+h, x:x+w, c] = (alpha_s * t_face[:, :, c] + alpha_l * img[y:y+h, x:x+w, c])

            elif count % 5 == 3:
        
                ####### GOKU 4 #############    
                w = int(w*1.6)
                h = int(h*1.8)

                x -= (int(h//7))
                y -= (int(w//1.7))  

                t_face = cv2.resize(goku4, (w, h))

                alpha_s = t_face[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(0, 3):
                    img[y:y+h, x:x+w, c] = (alpha_s * t_face[:, :, c] + alpha_l * img[y:y+h, x:x+w, c])

            elif count % 5 == 4:

                ####### GOKU 5 #############    
                w = int(w*2)
                h = int(h*1.7)

                x -= (int(h//2.5))
                y -= (int(w//3))   #130

                t_face = cv2.resize(goku5, (w, h))

                alpha_s = t_face[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(0, 3):
                    img[y:y+h, x:x+w, c] = (alpha_s * t_face[:, :, c] + alpha_l * img[y:y+h, x:x+w, c])        
        
        except:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                # cv2.rectangle(roi_color,(ex,ey),(ex+ew, ey+eh),(0,255,0),2)
                # cv2.circle(roi_color, (ex + (ew//2), ey + (eh//2)), (ew), (0,0,255), 2)
                
                t_face = cv2.resize(eye, (ew, eh))
                alpha_s = t_face[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(0, 3):
                    roi_color[ey:ey+eh, ex:ex+ew, c] = (alpha_s * t_face[:, :, c] + alpha_l * roi_color[ey:ey+eh, ex:ex+ew, c])


    # print(len(faces))
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    count += 1
    # print(count)

cap.release()
cv2.destroyAllWindows()