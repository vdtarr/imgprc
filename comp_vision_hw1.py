import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import math
from skimage.measure.simple_metrics import compare_psnr
from skimage import filters

img = mpimg.imread('DnCNN-PyTorch-master/SunnyLake.bmp')

print(img.shape)

I = np.zeros([img.shape[0],img.shape[1]],np.float32)
I_1 = np.zeros([img.shape[0],img.shape[1],img.shape[2]],np.float32)
I_5 = np.zeros([img.shape[0],img.shape[1],img.shape[2]],np.float32)
I_10 = np.zeros([img.shape[0],img.shape[1],img.shape[2]],np.float32)
I_20 = np.zeros([img.shape[0],img.shape[1],img.shape[2]],np.float32)

I_1_gray = np.zeros([img.shape[0],img.shape[1]],np.float32)
I_5_gray = np.zeros([img.shape[0],img.shape[1]],np.float32)
I_10_gray = np.zeros([img.shape[0],img.shape[1]],np.float32)
I_20_gray = np.zeros([img.shape[0],img.shape[1]],np.float32)


for i in range(img.shape[0]):

    for j in range(img.shape[1]):

         I[i,j] = (img[i,j,0]/3) + (img[i,j,1]/3) + (img[i,j,2]/3)
for i in range(I.shape[0]):
    for j in range(I.shape[1]):
        if(I[i,j]<50):
            I[i,j] = 0
        else:
            I[i,j] = 255

noise_1 = np.random.normal(0,(1/255.0),[img.shape[0],img.shape[1]])
noise_5 = np.random.normal(0,(5/255.0),[img.shape[0],img.shape[1]])
noise_10 = np.random.normal(0,(10/255.0),[img.shape[0],img.shape[1]])
noise_20 = np.random.normal(0,(20/255.0),[img.shape[0],img.shape[1]])



img = img / 255.0

I_1[:, :, 0] = img[:, :, 0] + noise_1
I_1[:, :, 1] = img[:, :, 1] + noise_1
I_1[:, :, 2] = img[:, :, 2] + noise_1

I_5[:, :, 0] = img[:, :, 0] + noise_5
I_5[:, :, 1] = img[:, :, 1] + noise_5
I_5[:, :, 2] = img[:, :, 2] + noise_5

I_10[:, :, 0] = img[:, :, 0] + noise_10
I_10[:, :, 1] = img[:, :, 1] + noise_10
I_10[:, :, 2] = img[:, :, 2] + noise_10

I_20[:, :, 0] = img[:, :, 0] + noise_20
I_20[:, :, 1] = img[:, :, 1] + noise_20
I_20[:, :, 2] = img[:, :, 2] + noise_20

for i in range(img.shape[0]):

    for j in range(img.shape[1]):

         I_1_gray[i,j] = (I_1[i,j,0]/3) + (I_1[i,j,1]/3) + (I_1[i,j,2]/3)
for i in range(img.shape[0]):

    for j in range(img.shape[1]):

         I_5_gray[i,j] = (I_5[i,j,0]/3) + (I_5[i,j,1]/3) + (I_5[i,j,2]/3)


for i in range(img.shape[0]):

    for j in range(img.shape[1]):

         I_10_gray[i,j] = (I_10[i,j,0]/3) + (I_10[i,j,1]/3) + (I_10[i,j,2]/3)
for i in range(img.shape[0]):

    for j in range(img.shape[1]):

         I_20_gray[i,j] = (I_20[i,j,0]/3) + (I_20[i,j,1]/3) + (I_20[i,j,2]/3)
kernel_lp1 = np.ones((3,3),np.float32)/9
kernel_lp2 = np.array([[1., 2., 1.],[2. ,4. ,2.],[1., 2., 1.]])/16
#kernel_lp2 = kernel_lp2.reshape(3,3)

kernel_hp1 = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
#kernel_hp1 = kernel_hp1.reshape(3,3)
kernel_hp2 = np.array([[0.17,0.67,0.17],[0.67,-3.33,0.67],[0.17,0.67,0.17]])
#kernel_hp2 = kernel_hp2.reshape(3,3)
I_20_gray_denoised_lp1 = cv2.filter2D(I_20_gray,-1,kernel_lp1)
I_10_gray_denoised_lp1 = cv2.filter2D(I_10_gray,-1,kernel_lp1)
I_5_gray_denoised_lp1 = cv2.filter2D(I_5_gray,-1,kernel_lp1)
I_1_gray_denoised_lp1 = cv2.filter2D(I_1_gray,-1,kernel_lp1)

I_20_gray_denoised_lp2 = cv2.filter2D(I_20_gray,-1,kernel_lp2)
I_10_gray_denoised_lp2 = cv2.filter2D(I_10_gray,-1,kernel_lp2)
I_5_gray_denoised_lp2 = cv2.filter2D(I_5_gray,-1,kernel_lp2)
I_1_gray_denoised_lp2 = cv2.filter2D(I_1_gray,-1,kernel_lp2)

I_20_gray_denoised_hp1 = cv2.filter2D(I_20_gray,-1,kernel_hp1)
I_10_gray_denoised_hp1 = cv2.filter2D(I_10_gray,-1,kernel_hp1)
I_5_gray_denoised_hp1 = cv2.filter2D(I_5_gray,-1,kernel_hp1)
I_1_gray_denoised_hp1 = cv2.filter2D(I_1_gray,-1,kernel_hp1)

I_20_gray_denoised_hp2 = cv2.filter2D(I_20_gray,-1,kernel_hp2)
I_10_gray_denoised_hp2 = cv2.filter2D(I_10_gray,-1,kernel_hp2)
I_5_gray_denoised_hp2 = cv2.filter2D(I_5_gray,-1,kernel_hp2)
I_1_gray_denoised_hp2 = cv2.filter2D(I_1_gray,-1,kernel_hp2)

print(compare_psnr(I_20_gray_denoised_lp1,I_20_gray,1.))
print(compare_psnr(I_20_gray_denoised_lp2,I_20_gray,1.))

print(compare_psnr(I_10_gray_denoised_lp1,I_10_gray,1.))
print(compare_psnr(I_10_gray_denoised_lp2,I_10_gray,1.))

print(compare_psnr(I_5_gray_denoised_lp1,I_5_gray,1.))
print(compare_psnr(I_5_gray_denoised_lp2,I_5_gray,1.))

print(compare_psnr(I_1_gray_denoised_lp1,I_1_gray,1.))
print(compare_psnr(I_1_gray_denoised_lp2,I_1_gray,1.))






plt.imsave('I_20_gray_denoised_hp1.png',I_20_gray_denoised_hp1)
plt.imsave('I_20_gray_denoised_hp2.png',I_20_gray_denoised_hp2)

plt.imsave('I_10_gray_denoised_hp1.png',I_10_gray_denoised_hp1)
plt.imsave('I_10_gray_denoised_hp2.png',I_10_gray_denoised_hp2)


plt.imsave('I_5_gray_denoised_hp1.png',I_5_gray_denoised_hp1)
plt.imsave('I_5_gray_denoised_hp2.png',I_5_gray_denoised_hp2)



plt.imsave('I_1_gray_denoised_hp1.png',I_1_gray_denoised_hp1)
plt.imsave('I_1_gray_denoised_hp2.png',I_1_gray_denoised_hp2)


plt.imsave('I_20_gray_denoised_lp1.png',I_20_gray_denoised_lp1)
plt.imsave('I_20_gray_denoised_lp2.png',I_20_gray_denoised_lp2)

plt.imsave('I_10_gray_denoised_lp1.png',I_10_gray_denoised_lp1)
plt.imsave('I_10_gray_denoised_lp2.png',I_10_gray_denoised_lp2)

plt.imsave('I_5_gray_denoised_lp1.png',I_5_gray_denoised_lp1)
plt.imsave('I_5_gray_denoised_lp2.png',I_5_gray_denoised_lp2)


plt.imsave('I_1_gray_denoised_lp1.png',I_1_gray_denoised_lp1)
plt.imsave('I_1_gray_denoised_lp2.png',I_1_gray_denoised_lp2)

"""
plt.figure()
plt.subplot(211)
plt.imshow(I_20_gray_denoised_lp1)
plt.subplot(212)
plt.imshow(I_20_gray_denoised_lp2)


plt.figure()
plt.subplot(211)
plt.imshow(I_10_gray_denoised_lp1)
plt.subplot(212)
plt.imshow(I_10_gray_denoised_lp2)

plt.figure()
plt.subplot(211)
plt.imshow(I_5_gray_denoised_lp1)
plt.subplot(212)
plt.imshow(I_5_gray_denoised_lp2)

plt.figure()
plt.subplot(211)
plt.imshow(I_1_gray_denoised_lp1)
plt.subplot(212)
plt.imshow(I_1_gray_denoised_lp2)
plt.figure()"""

img_noisy = cv2.imread('DnCNN-PyTorch-master/Figure_1.png')
median = cv2.medianBlur(img_noisy,5)
compare = np.concatenate((img_noisy, median), axis=1)
kernel2 = np.ones((5,5),np.uint8)
#opening = cv2.morphologyEx(img_noisy, cv2.MORPH_OPEN, kernel)
#dst = cv2.fastNlMeansDenoisingColored(img_noisy,10,10,7,21)
#print(compare_psnr(img_noisy,median,255.))

#cv2.imshow('median_filtering',median)
#cv2.imwrite('median_filtering_2.png',compare)

plt.figure()
plt.imshow(I_1)

plt.figure()
plt.imshow(I_5)

plt.figure()
plt.imshow(I_10)

plt.figure()
plt.imshow(I_20)

plt.figure()
plt.hist(I.ravel(),64,[0 ,256])

plt.figure()
plt.imshow(I)

plt.figure()
plt.imshow(img)
plt.show()
