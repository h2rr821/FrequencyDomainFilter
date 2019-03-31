# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 17:42:25 2019

@author: h2r
"""

import cv2
import numpy as np
from math import exp
from math import pi

def main():
    img=cv2.imread('lena_gray_256.tif', 0)
    
    cv2.imshow('org', img)
    height,width=img.shape
    
    #get the image of double size
    imgPad=np.empty([height*2,width*2],'uint8')
    
    #1. pad the image with 0
    imgPad[0:height, 0:width]=img
    
    cv2.imshow('pad', imgPad)
    
    # nomalize??===========================================
    
    #for row in range(0, height):
    #    for col in range(0,width):
    #        imgPad[row, col]=imgPad[row,col]/255;
    
    heightPad, widthPad=imgPad.shape
    
    #2. center by multiply (-1)^(x+y)
    # multiply f(x,y) by (-1)^(x+y)
    for row in range(0,heightPad):
        for col in range(0,widthPad):
            imgPad[row,col]=imgPad[row,col]*((-1)**(row+col))
    
    #3. Fourier transform of the centered and padded image.=>F(u,v)
    fimagePad = np.fft.fft2(imgPad)
    
    #fimagePad=cv2.dft(np.float32(imgPad), flags=cv2.DFT_COMPLEX_OUTPUT)
    #fimagePad = np.log(1+np.abs(fimagePad))
    
    
    
    #cv2.normalize(fimagePad,fimagePad,0,255)
    #cv2.imshow('fimagePad',fimagePad)
    
    #fimagePad_img=fimagePad.astype(np.uint8)
    
    #4. generate Gaussian N(u,v) ================================= 
    Gaussian_filter=np.empty([height*2,width*2],'float32')
    d0=10
    for row in range(0,heightPad):
        for col in range(0, widthPad):
            d2=(row-height)*(row-height)+(col-width)*(col-width)
            Gaussian_filter[row,col]=exp(-(d2)/(2*d0*d0))
    
    cv2.imshow('Gaussian_filter', Gaussian_filter)
    
    #5. generate Laplacian L(U,V)================================
    x=np.empty([height*2,width*2],'float32')
    y=np.empty([height*2,width*2],'float32')
    
    for row in range(0,heightPad):
        for i in range(-height, height):
            y[row, height+i]=i
            
    for i in range(-width,width):
        for col in range(0,widthPad):
            x[width+i, col]=i
    
    Laplacian_filter=np.empty([height*2,width*2],'float32')
    for row in range(0, heightPad):
        for col in range(0,widthPad):
            d2=x[row, col]*x[row,col]+y[row,col]*y[row,col]
            Laplacian_filter[row,col]=-4*pi*pi*d2
            
    #cv2.imshow('Laplacian_filter',Laplacian_filter)    

    fimagePad2=np.empty([height*2,width*2],'float32')  
    
    #6. F(u,v).H(u,v).L(u,v)
    for row in range(0, heightPad):
        for col in range(0,widthPad):
            fimagePad2[row, col]=fimagePad[row,col]*Gaussian_filter[row, col]*Laplacian_filter[row,col]
   
    #7.invert Fourier transform 
    i_imagePad=np.fft.ifft2(fimagePad2)
    #i_imagePad = cv2.idft(fimagePad2)
    
    #8 uncenter by multiply (-1)^(x+y)
    for row in range(0,heightPad):
        for col in range(0,widthPad):
            i_imagePad[row,col]=i_imagePad[row,col]*((-1)**(row+col))
    
    i_imagePad=i_imagePad.astype(np.float32)
    cv2.imshow('final',i_imagePad)
    
    #9 unpad image
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ =='__main__':
    main()