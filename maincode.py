import cv2


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

while True:
   
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor= 1.3 , minNeighbors=5)

    for (x, y, w, h) in faces:
     
        cv2.rectangle(frame, (x, y), (x+w, y+h), (200 , 255, 200), 2)
        cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()

