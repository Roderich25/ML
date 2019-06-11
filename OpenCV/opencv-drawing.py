import numpy as np
import cv2

#image = np.zeros([512, 512, 3], np.uint8) # black image
image = cv2.imread("lena.jpg", 1)

#image = cv2.line(image, (0, 0), (350, 350), (0, 255, 0), 4) # color in BGR format
#image = cv2.arrowedLine(image, (0, 0), (350, 350), (0, 255, 0), 4) # last parameter is the tickness of the shape
image = cv2.rectangle(image, (210, 240), (380, 340), (0, 0, 255), 3)
#image = cv2.rectangle(image, (210, 240), (380, 340), (0, 0, 255), -1) # -1 fills the shape
#image = cv2.circle(image, (290, 300), 75, (255, 0, 0), 2) # center, radius
image = cv2.putText(image, 'Random Text', (80, 300), cv2.FONT_ITALIC, 2, (255, 255, 255), 4)
cv2.imshow('Image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
