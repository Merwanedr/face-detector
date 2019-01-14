import cv2

# Load cascade classifiers
face_cascade = cv2.CascadeClassifier('face.xml')
eyes_cascade = cv2.CascadeClassifier('eyes.xml')

# Load image
initialImage = cv2.imread('paypal_mafia.jpg')
# Resize the image to half of it's size in case it's too big
img = cv2.resize(initialImage, (int(initialImage.shape[1]/2), int(initialImage.shape[0]/2)))
# Generate gray image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect scale on faces and eyes
faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
eyes = eyes_cascade.detectMultiScale(gray_img)

# Draw shape for faces
for x,y,w,h in faces:
    img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
    # Draw shape for eyes for each face
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0), 1)

# Show the image
cv2.imshow('Gray', img)

# Destroy the window by pressing any key
cv2.waitKey(0)

cv2.destroyAllWindows()