import cv2
import numpy as np

window_name = 'orange ball tracker'
remember_last_100= [[-1 for i in range(100)]for j in range(2)]
print remember_last_100

def track(image):

    hsv1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([0,177,112])
    upper_orange = np.array([92,255,234])

    mask = cv2.inRange(hsv1, lower_orange, upper_orange)
    bmask = cv2.GaussianBlur(mask, (5,5),0)

    cv2.imshow(window_name,bmask)

    moments = cv2.moments(bmask)
    m00 = moments['m00']
    centroid_x, centroid_y = None, None

    if m00 != 0:
        centroid_x = int(moments['m10']/m00)
        centroid_y = int(moments['m01']/m00)

    if centroid_x!=None and centroid_y!=None:

        for x in range(100):
            for y in range(2):
                



if __name__ == '__main__':

    capture = cv2.VideoCapture(0)

    while True:

        okay, image = capture.read()

        if okay:
            track(image)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break

        else:

           print('Capture failed')
           break