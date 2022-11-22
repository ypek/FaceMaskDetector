import cv2 as cv

camera = cv.VideoCapture(0)

cascadeFrontalFace = 'haarcascade_frontalface_default.xml'
cascadeWhitMask = 'mask_cascade.xml'

faceCascade = cv.CascadeClassifier(cascadeFrontalFace)

while True:
    ret, frame = camera.read()
    frame = cv.flip(frame, 1)
    cv.imshow('frame', frame)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    #Detec Face No Mask
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(frame, 'NoMask', (x, y), font, 1, (10, 10, 255), 2, cv.LINE_AA)

    #Detec Face With Mask
    maskCascade = cv.CascadeClassifier(cascadeWhitMask)
    mask = maskCascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in mask:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(frame, 'Mask', (x, y), font, 1, (10, 255, 10), 2, cv.LINE_AA)

    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv.destroyAllWindows()
