import cv2
import pytesseract

# configuring pytesseract
pytesseract.pytesseract.tesseract_cmd = 'your tesseract path'

# reading the image
img = cv2.imread('your image name')
imgray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(imgray1, 316, 483)
contours, p = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

# detecting license plate contour
for i in contours:
    area = cv2.contourArea(i)
    approx = cv2.approxPolyDP(i, 0.01*cv2.arcLength(i, True), True)
    if len(approx) == 4 and cv2.contourArea(i)>700:
        cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(i)
        img4 = imgray1[y:y+h, x:x+w]
licenseplate = pytesseract.image_to_string(img4)
cv2.putText(img, licenseplate, (x-10, y-25), cv2.FONT_ITALIC, 0.8, (0, 255, 0), 2)

# outut images
cv2.imwrite('final.jpg', img4)
cv2.imwrite('grayscale.jpg', imgray1)
cv2.imwrite('canny.jpg', canny)
cv2.imwrite('contour.jpg', img)
cv2.imwrite('licenseplate.jpg', img4)
print(licenseplate)
cv2.imshow('grayscale', imgray1)
cv2.imshow('canny', canny)
cv2.imshow('test', img)
cv2.imshow('res', img4)
cv2.waitKey(0)
cv2.destroyAllWindows()
