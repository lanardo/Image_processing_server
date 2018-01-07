from PIL import Image
from pytesseract import image_to_string

import cv2

# cv2_im = cv2.imread('../data/temp/volvo_xc60_2017_YV449MRS6H2008569_236_244864367.jpg')
cv2_im = cv2.imread("../data/temp/ram_dakota_2011_1D7RW3GK0BS521408_13687_271731641.png")
cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
pil_im = Image.fromarray(cv2_im)

text = image_to_string(pil_im, lang='eng')

print(text)

print(text.replace('\n', ' '))

# print(image_to_string(Image.open('../data/temp/volvo_xc60_2017_YV449MRS6H2008569_236_244864367.jpg'), lang='eng'))
