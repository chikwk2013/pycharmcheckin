import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#testing
# Feature set containing (x,y) values of 25 known/training data
trainData = np.random.randint(0, 100, (25,2)).astype(np.float32)
# Label each one either Red or Blue with numbers 0 and 1
responses = np.random.randint(0, 2, (25, 1)).astype(np.float32)
# Take Red neighbours and plot them
red = trainData[responses.ravel()==0]
plt.scatter(red[:, 0], red[:, 1], 80, 'r', '^')
# Take Blue neighbours and plot them
blue = trainData[responses.ravel()==1]
plt.scatter(blue[:, 0], blue[:, 1], 80, 'b', 's')
plt.show()