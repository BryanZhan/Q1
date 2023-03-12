import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

def MSE(img1,img2):
    w,h=img1.shape
    mse=0
    for i in range(w):
        for j in range(h):
            mse=mse+(img1[i,j]-img2[i,j])**2
    return mse/(w*h)

def Q1_a(F):
    rlt=np.zeros(F.shape,dtype=complex)
    l=[]
    for y in range(0,F.shape[0]):
        for x in range(0,F.shape[1]):
            temp=math.sqrt(F[y][x].real*F[y][x].real+F[y][x].imag*F[y][x].imag)
            rlt[y][x]=F[y][x]
            l.append(temp)
    l=sorted(l,reverse=True)
    threshold=l[int(len(l)/4)]
    for y in range(0,F.shape[0]):
        for x in range(0,F.shape[1]):
            temp=math.sqrt(F[y][x].real*F[y][x].real+F[y][x].imag*F[y][x].imag)        
            if(temp<=threshold):
                rlt[y][x]=0+0j
    return rlt

def Q1_b(inverse_F):
    rlt=[]
    for i in range(0,len(inverse_F)):
        temp=Q1_a(inverse_F[i])
        rlt.append(temp)
    return rlt

src=cv.imread("bridge.jpg",cv.IMREAD_GRAYSCALE)
"Q1a"
F=np.fft.fft2(np.float32(src))
F=np.fft.fftshift(F)
F_Q1a=Q1_a(F)
f_Q1a=np.real(np.fft.ifft2(np.fft.ifftshift(F_Q1a)))
output_Q1a=cv.normalize(f_Q1a*255,None,0,255,cv.NORM_MINMAX)
print("MSE of src and imageA: %f" %MSE(src,output_Q1a))
cv.imwrite("imageA.jpg",np.uint8(output_Q1a))

"Q1b"
l_of_src=[]
for j in range(0,16):
    for i in range(0,16):
        block=src[j*16:(j+1)*16,i*16:(i+1)*16]
        l_of_src.append(block)
l_of_F=[]
for i in range(0,len(l_of_src)):
    l_of_F.append(np.fft.fftshift(np.fft.fft2(l_of_src[i])))
l_of_F_Q1b=Q1_b(l_of_F)
f_Q1B=np.zeros(src.shape)
for i in range(0,len(l_of_F_Q1b)):
    f_Q1B[int(i/16)*16:int((i/16)+1)*16,int(i%16)*16:int((i%16)+1)*16]=np.real(np.fft.ifft2(np.fft.ifftshift(l_of_F_Q1b[i][:,:])))

output_Q1B=cv.normalize(f_Q1B*255,None,0,255,cv.NORM_MINMAX)
print("MSE of src and imageB: %f" %MSE(src,output_Q1B))
cv.imwrite("imageB.jpg",np.uint8(output_Q1B))

r_src=cv.pyrDown(src)
r_F=np.fft.fft2(r_src)
r_Fshift=np.fft.fftshift(r_F)
temp=Q1_a(r_Fshift)
F_Q1C=np.zeros(src.shape,dtype=complex)
F_Q1C[64:192,64:192]=temp
f_Q1C=np.real(np.fft.ifft2(np.fft.ifftshift(F_Q1C)))

output_Q1C=cv.normalize(f_Q1C*255,None,0,255,cv.NORM_MINMAX)
#plt.imshow(np.uint8(output_Q1C),cmap-gray)
print("MSE of src and imageB: %f" %MSE(src,output_Q1C))
cv.imwrite("imageC.jpg ",output_Q1C)