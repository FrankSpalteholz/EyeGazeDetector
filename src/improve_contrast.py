import cv2

#-----Reading the image-----------------------------------------------------
img = cv2.imread('../footage/me/me.0002.jpg', 1)

# converting image to LAB Color model
lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# splitting the LAB image to different channels
l, a, b = cv2.split(lab)

# applying CLAHE to L-channel
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10,10))
cl = clahe.apply(l)

# merge the CLAHE enhanced L-channel with the a and b channel
limg = cv2.merge((cl,a,b))

# converting image from LAB Color model to RGB model
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


cv2.imshow('final', final)
cv2.waitKey(2000)
cv2.imwrite('../render/me/me_contrast_corrected_clahe.0003.jpg',final)

#_____END_____#