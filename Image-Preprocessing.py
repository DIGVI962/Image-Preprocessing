#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
import sns
from PIL import Image, ImageChops, ImageEnhance, ImageOps


# In[2]:


img = cv2.imread('C://Users//IAmTheWizard//Desktop//list.png', -1)


# In[10]:


# Resizing

print(type(img))
print(img.shape)
new_image = cv2.resize(img, (400,600))
print(new_image.shape)


# In[11]:


# Printing Original Image
plt.figure(figsize=(10,20))
plt.imshow(img)


# In[12]:


# Removing shadows

rgb_planes = cv2.split(img)

result_planes = []
result_norm_planes = []
for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))    
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(plane, bg_img)    
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)

result = cv2.merge(result_planes)
result_norm = cv2.merge(result_norm_planes)

cv2.imwrite('shadows_out.png', result)
cv2.imwrite('shadows_out_norm.png', result_norm)
plt.figure(figsize=(10,20))
plt.imshow(result_norm)

print(rgb_planes)


# In[13]:


# Enhancing Image

img = Image.fromarray(np.uint8(result_norm)).convert('RGB')
img = Image.fromarray(result_norm.astype('uint8'),'RGB')
img=ImageEnhance.Contrast(img)
img = img.enhance(7)
img.show('Image with more contrast')
print(img.mode)
img = asarray(img)
print(type(img))


# In[23]:


# Image processing

'''
img = cv2.imread('C://Users//IAmTheWizard//Desktop//OCR_training_data//8.jpeg',1)

originalimage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)'''
#img = cv2.cvtColor(result_norm, cv2.COLOR_RGB2GRAY)
#plt.imshow(img)
#plt.title('OG')


# Original
fig=plt.figure(figsize=(10,20))
plt.figure(figsize=(10,20))
plt.imshow(img)
plt.title('Original')

# Thresholding
ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
plt.figure(figsize=(10,20))
plt.imshow(thresh,cmap='gray')
plt.title('Threshold')

# Denoising
denoise = cv2.fastNlMeansDenoising(thresh,None,10,9,21)
plt.figure(figsize=(10,20))
plt.imshow(denoise,cmap='gray')
plt.title('Denoise')


# Dilating
kernel = np.ones((2,2), np.uint8)
imag_dil = cv2.dilate(thresh,kernel,iterations=3)
plt.figure(figsize=(10,20))
plt.imshow(imag_dil,cmap='gray')
plt.title('Dilate')
'''
imag_dil = cv2.cvtColor(imag_dil, cv2.COLOR_BGR2GRAY)
contour,hierarchy = cv2.findContours(imag_dil,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contour,key=cv2.contourArea,reverse=True)
largest_contour = contours[0]
minarearect = cv2.minAreaRect(largest_contour)
angle = minarearect[-1]
if angle<-45.0:
    angle=90+angle
    angle=-1.0*angle
print(angle)
pil_img = Image.fromarray(np.uint8(imag_dil)).convert('RGB')
pil_img.rotate(90-angle)
'''


# Eroding
kernel = np.ones((3,3), np.uint8)
imag_er = cv2.erode(imag_dil,kernel,iterations=1)
plt.figure(figsize=(10,20))
plt.imshow(imag_er,cmap='gray')
plt.title('Erode')


# In[8]:


# Deskewing

kernel = np.ones((40,20), np.uint8)
imag_dil = cv2.dilate(denoise,kernel,iterations=1)
plt.figure(figsize=(10,20))
plt.imshow(imag_dil,cmap='gray')
plt.title('Dilate')
imag_dil = cv2.cvtColor(imag_dil, cv2.COLOR_BGR2GRAY)
contour,hierarchy = cv2.findContours(imag_dil,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contour,key=cv2.contourArea,reverse=True)
largest_contour = contours[0]
cont = cv2.drawContours(imag_dil,largest_contour,0,(124,252,0),10)

plt.figure(figsize=(10,20))
plt.imshow(cont)
plt.title('cont')

minarearect = cv2.minAreaRect(largest_contour)
angle = minarearect[-1]
if angle<-45:
    angle=90+angle
    angle=-1.0*angle
print(angle)
pil_img = Image.fromarray(np.uint8(denoise)).convert('RGB')
pil_img = pil_img.rotate(90-angle)
pil_img.show()

deskewed = asarray(pil_img)

plt.figure(figsize=(10,20))
plt.imshow(deskewed)
plt.title('Deskewed Image')


# In[15]:


# OCR

text = pytesseract.image_to_string(deskewed,lang='eng')
print(text)


# In[ ]:




