import cv2
import numpy as np

img = cv2.imread("./denoising_images/HR _image.png")

denoise_1 = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
denoise_2 = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
denoise_3 = cv2.fastNlMeansDenoisingColored(img, None, 15, 15, 7, 21)

cv2.imwrite('image_1.png', denoise_1)
cv2.imwrite('image_2.png', denoise_2)
cv2.imwrite('image_3.png', denoise_3)
