import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
 
    for (x, y, w, h) in faces:
        new_x = int(x+(w/2))
        new_y = int(y+ (h/2))
        radius = int(h/2)
        cv2.circle(img, (new_x,new_y), radius,(255, 0, 0), 2 )
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
    
        eyes = eye_cascade.detectMultiScale(roi_gray)
        smile = smile_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            new_ex = int(ex+(ew/2))
            new_ey = int(ey+ (eh/2))
            eradius = int(eh/2)
            cv2.circle(roi_color, (new_ex,new_ey), eradius,(0,255,0), 2 )
            # cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)
        for (sx,sy,sw,sh) in smile:
            new_x = int(sx+(sw/2))
            new_y = int(sy+ (sh/2))
            radius = int(sh/2)
            cv2.circle(roi_color, (new_x,new_y), radius,(0, 0, 255), 2 )
            # cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 0, 255), 2)
    
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
 
cap.release()
cv2.destroyAllWindows()
