import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('C:/Users/User/Desktop/pi.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray' , interpolation='bicubic'
plt.plot([50,100] , [80,100], 'c' , linewidth =5)
plt.show()
