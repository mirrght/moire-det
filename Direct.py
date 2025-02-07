import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.colors as pclr  # plt 用于显示图片
from skimage import io, color, transform
import numpy as np
import os, sys, heapq
import cv2
from PIL import Image
import argparse
import shutil


def makwenf(image):
    img = np.array(image)
    img = np.mean(img, axis=2)
    h, w = img.shape
    img = img.astype(float)

    size = 200  # 假设矩形区域的大小为 100x100

    # 应用高斯模糊进行预处理（由找点变成找区域）
    gray = cv2.GaussianBlur(img, (size - 1, size - 1), 0)
    # 利用cv2.minMaxLoc寻找到图像中最亮和最暗的区域
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    y, x = maxLoc  # maxLoc 是最大值位置的坐标
    half_size = size // 2

    # 边界检查，确保矩形区域不会超出图像范围
    x1 = max(0, x - half_size)
    y1 = max(0, y - half_size)
    x2 = min(h - 1, x + half_size)
    y2 = min(w - 1, y + half_size)

    # 提取矩形区域
    rectangular_region = image[x1:x2, y1:y2]
    #     print(x1,x2,y1,y2)

    return toimage_max_labol(rectangular_region)


def toimage_max_labol(img):
    img = np.array(img)
    h, w, q = img.shape
    img = img.astype(float)
    image_max = np.max(img, axis=2)
    image_max_labol = np.zeros([h, w])
    for x in range(0, h):
        for y in range(0, w):
            for z in range(0, q):
                if img[x, y, z] == image_max[x, y]:
                    image_max_labol[x, y] = z / 2
    return image_max_labol


def detect_edges(gray_image):
    # 定义滤波器
    filter_kernel = np.array([[1, 1, 1],
                              [1, -8, 1],
                              [1, 1, 1]])

    # 进行滤波操作
    filtered_image = cv2.filter2D(gray_image, -1, filter_kernel)

    # 将非零值设为1，零值设为0
    edges_image = np.where(filtered_image == 0, 0, 1)
    p = np.sum(edges_image) / (edges_image.shape[0] * edges_image.shape[1])

    #     plt.imshow(edges_image, cmap='gray')
    #     plt.show()
    return p

def drgb(image):
    ci = 0.3
    k = 2
    ci_in = 0.2
    ci_max = 2.1  # 2.1
    p_min = 0.1  # 10^(-3)

    # 检测边缘
    img_wen = makwenf(image)

    p = detect_edges(img_wen)

    return p

def Direct_main(positiveImagePath,negativeImagePath):


    right1 = 0
    count1 = 0

    scores = []
    labels = []

    p_min = 0.1  # 0.1
    print(positiveImagePath)
    for root, dirs, files in os.walk(positiveImagePath):
        for file in files:

            # 检查文件扩展名是否为图像类型（.jpg或.png）
            if file.lower().endswith((".jpg", ".png")):
                # 构建图像文件的完整路径
                image_path = os.path.join(root, file)

                print(file)

                # 使用OpenCV打开图像文件
                image = io.imread(image_path)

                p = drgb(image)

                scores += [p]

                if p > p_min:
                    out = 1
                    shutil.copy(image_path, './moire pattern/' + file)
                else:
                    out = 0

                count1 += 1
                if out == 1:
                    right1 += 1

                labels += [1]



    right2 = 0
    count2 = 0

    for root, dirs, files in os.walk(negativeImagePath):
        for file in files:
            # 检查文件扩展名是否为图像类型（.jpg或.png）
            if file.lower().endswith((".jpg", ".png")):
                # 构建图像文件的完整路径
                image_path = os.path.join(root, file)

                # 使用OpenCV打开图像文件
                image = io.imread(image_path)
                print(file)

                p = drgb(image)

                scores += [p]

                if p > p_min:
                    out = 1
                else:
                    out = 0

                count2 += 1
                if out == 0:
                    right2 += 1

                labels += [0]


    print(count2)

    # precise = right1 / (right1 + count2 - right2)
    # recall = right1 / count1
    # acc = (right1 + right2) / (count1 + count2)
    # print("precise: %.2f" % (precise * 100))
    #
    # print("recall: %.2f" % (recall * 100))
    #
    # print("acc: %.2f" % (acc * 100))


if __name__ == '__main__':
    Direct_main('F:/ta/摩尔纹课题/data/摩尔纹图片', 'F:/ta/摩尔纹课题/data/正常图片')
