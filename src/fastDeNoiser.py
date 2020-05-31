import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('../footage/me/me.0002.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
dst = cv.fastNlMeansDenoisingColored(img,None,5,1,7,21)
plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(dst)
plt.show()