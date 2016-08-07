import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

window_name = 'orange ball tracker'
X_train= np.zeros(100)

X_train=X_train.reshape((100,1))

Y_train= np.zeros(100)
Y_train=Y_train.reshape((100,1))

print X_train.shape
print Y_train.shape
x=1


def track(image):

    global x

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
            
        X_train[x]=centroid_x
        Y_train[x]=centroid_y

        x=(x+1)%100

        algo = linear_model.LinearRegression(fit_intercept=True,normalize=False)
        algo.fit(X_train,Y_train)

        coefficients = algo.coef_
        #print coefficients
        print centroid_x
        print centroid_y
        intercept = algo.intercept_



                



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