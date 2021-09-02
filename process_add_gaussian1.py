from skimage import io, img_as_float
import numpy as np
import cv2
import os
from PIL import Image

def Proposed_method(item_name):
    ori_image = io.imread('./input/' + item_name)
    #ori_image = Image.open('./input/' + item_name)
    o = np.array(ori_image)
    if o.shape == (256, 256):
        o = cv2.merge([o, o, o])
    B, G, R = cv2.split(o)
    k = image_mean(B)

    if k < 10:
        add_out = add_gaussian1(ori_image,item_name)
    elif k > 10 and k < 20:
        add_out = add_gaussian2(ori_image,item_name)
    elif k > 20 and k < 30:
        add_out = add_gaussian3(ori_image,item_name)
    elif k > 30 and k < 40:
        add_out = add_gaussian4(ori_image,item_name)
    elif k > 40 and k < 50:
        add_out = add_gaussian5(ori_image,item_name)
    elif k > 50 and k < 60:
        add_out = add_gaussian6(ori_image,item_name)
    elif k > 60 and k < 70:
        add_out = add_gaussian7(ori_image,item_name)
    elif k > 70 and k < 80:
        add_out = add_gaussian8(ori_image,item_name)
    else :
        add_out = add_gaussian9(ori_image,item_name)
    #io.imsave('./cfar6/' + item_name, noisy)
    return add_out

def add_gaussian1(image,name):
    mean = 0
    var = 1

    o = np.array(image)
    if o.shape == (256, 256):
        new_image = o
    else:
        B, G, R = cv2.split(o)
        # new_image = cv2.merge([B, B, B])
        new_image = B
    image = img_as_float(new_image)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    noise = np.clip(noise, 0.0, 1.0)
    noisy = image + noise
    noisy = np.clip(noisy, 0.0, 1.0)
    noisy = cv2.merge([noisy, noisy, noisy])
    io.imsave('./add_output1/' + name, noisy)
    return noisy
    #io.imshow(noisy)
    #io.show()
def add_gaussian2(image,name):
    mean = 0
    var = 0.66

    o = np.array(image)
    if o.shape == (256, 256):
        new_image = o
    else:
        B, G, R = cv2.split(o)
        # new_image = cv2.merge([B, B, B])
        new_image = B
    image = img_as_float(new_image)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    noise = np.clip(noise, 0.0, 1.0)
    noisy = image + noise
    noisy = np.clip(noisy, 0.0, 1.0)
    noisy = cv2.merge([noisy, noisy, noisy])
    io.imsave('./add_output1/' + name, noisy)
    return noisy

def add_gaussian3(image,name):
    mean = 0
    var = 0.43

    o = np.array(image)
    if o.shape == (256, 256):
        new_image = o
    else:
        B, G, R = cv2.split(o)
        # new_image = cv2.merge([B, B, B])
        new_image = B
    image = img_as_float(new_image)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    noise = np.clip(noise, 0.0, 1.0)
    noisy = image + noise
    noisy = np.clip(noisy, 0.0, 1.0)
    noisy = cv2.merge([noisy, noisy, noisy])
    io.imsave('./add_output1/' + name, noisy)
    return noisy

def add_gaussian4(image,name):
    mean = 0
    var = 0.28

    o = np.array(image)
    if o.shape == (256, 256):
        new_image = o
    else:
        B, G, R = cv2.split(o)
        # new_image = cv2.merge([B, B, B])
        new_image = B
    image = img_as_float(new_image)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    noise = np.clip(noise, 0.0, 1.0)
    noisy = image + noise
    noisy = np.clip(noisy, 0.0, 1.0)
    noisy = cv2.merge([noisy, noisy, noisy])
    io.imsave('./add_output1/' + name, noisy)
    return noisy

def add_gaussian5(image,name):
    mean = 0
    var = 0.18

    o = np.array(image)
    if o.shape == (256, 256):
        new_image = o
    else:
        B, G, R = cv2.split(o)
        # new_image = cv2.merge([B, B, B])
        new_image = B
    image = img_as_float(new_image)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    noise = np.clip(noise, 0.0, 1.0)
    noisy = image + noise
    noisy = np.clip(noisy, 0.0, 1.0)
    noisy = cv2.merge([noisy, noisy, noisy])
    io.imsave('./add_output1/' + name, noisy)
    return noisy

def add_gaussian6(image,name):
    mean = 0
    var = 0.09

    o = np.array(image)
    if o.shape == (256, 256):
        new_image = o
    else:
        B, G, R = cv2.split(o)
        # new_image = cv2.merge([B, B, B])
        new_image = B
    image = img_as_float(new_image)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    noise = np.clip(noise, 0.0, 1.0)
    noisy = image + noise
    noisy = np.clip(noisy, 0.0, 1.0)
    noisy = cv2.merge([noisy, noisy, noisy])
    io.imsave('./add_output1/' + name, noisy)
    return noisy

def add_gaussian7(image,name):
    mean = 0
    var = 0.04

    o = np.array(image)
    if o.shape == (256, 256):
        new_image = o
    else:
        B, G, R = cv2.split(o)
        # new_image = cv2.merge([B, B, B])
        new_image = B
    image = img_as_float(new_image)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    noise = np.clip(noise, 0.0, 1.0)
    noisy = image + noise
    noisy = np.clip(noisy, 0.0, 1.0)
    noisy = cv2.merge([noisy, noisy, noisy])
    io.imsave('./add_output1/' + name, noisy)
    return noisy

def add_gaussian8(image,name):
    mean = 0
    var = 0.01

    o = np.array(image)
    if o.shape == (256, 256):
        new_image = o
    else:
        B, G, R = cv2.split(o)
        # new_image = cv2.merge([B, B, B])
        new_image = B
    image = img_as_float(new_image)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    noise = np.clip(noise, 0.0, 1.0)
    noisy = image + noise
    noisy = np.clip(noisy, 0.0, 1.0)
    noisy = cv2.merge([noisy, noisy, noisy])
    io.imsave('./add_output1/' + name, noisy)
    return noisy

def add_gaussian9(image,name):
    noisy = image
    io.imsave('./add_output1/' + name, noisy)
    return noisy

def image_mean(array):
    sum = 0
    ave = 0
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            sum=sum + array[i][j]
            ave = sum / 65536
            #print(ave)
    return ave

def dir_loop(dir_path):
    for item in os.listdir(dir_path):
        print(item)
        Proposed_method(item)

def main():
    dir_loop('./input')

if __name__ == '__main__':
    main()