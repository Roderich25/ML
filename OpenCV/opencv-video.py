import cv2
import datetime
cap = cv2.VideoCapture(0) # default camera
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (640, 480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        font = cv2.FONT_HERSHEY_SIMPLEX
        dtt = str(datetime.datetime.now())
        frame = cv2.putText(frame, dtt, (10, 50), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
        out.write(frame)        
        print(cap.get(3), cap.get(4), cap.get(cv2.CAP_PROP_FPS)) # cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, fps
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('myFrame', frame)                
        if cv2.waitKey(1) & 0xFF == ord('q'): # press q key to exit and save
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()
