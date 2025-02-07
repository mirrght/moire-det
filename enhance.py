import cv2
from PIL import Image
from PIL import ImageEnhance
from numpy.ma import array
import numpy as np
import os
import argparse
from os.path import isfile, join

def main(args):

    imagepath = (args.imagepath)
    enhancepath = (args.enhancepath)

    del_file(enhancepath)

    for parent, dirnames, filenames in os.walk(imagepath):
        for filename in filenames:
            print('filename is: ' + filename)
            # 把文件名添加到一起后输出
            if filename!='.DS_Store':
                currentPath = os.path.join(parent, filename)
                operate(currentPath, filename, enhancepath)

def del_file(path_data):
    for i in os.listdir(path_data) :# os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + "/" + i#当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data) == True:#os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            del_file(file_data)

def operate(currentPath, filename, targetPath):
    # 读取图像
    image = Image.open(currentPath)

    f = (os.path.splitext(filename)[0])
    image.save(join(targetPath, filename))
    # image.show()
    # 增强亮度 bh_
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.07
    image_brightened_h = enh_bri.enhance(brightness)
    # image_brightened_h.show()
    image_brightened_h.save(join(targetPath, f+'1.jpg'))  # 保存

    # 降低亮度 bl_
    enh_bri_low = ImageEnhance.Brightness(image)
    brightness = 0.87
    image_brightened_low = enh_bri_low.enhance(brightness)
    # image_brightened_low.show()
    image_brightened_low.save(join(targetPath, f+'2.jpg'))

    # # 改变色度 co_
    # enh_col = ImageEnhance.Color(image)
    # color = 0.8
    # image_colored = enh_col.enhance(color)
    # # image_colored.show()
    # image_colored.save(join(targetPath, f+'3.jpg'))

    # 改变对比度 cont_
    enh_con = ImageEnhance.Contrast(image)
    contrast = 0.8
    image_contrasted = enh_con.enhance(contrast)
    # image_contrasted.show()
    image_contrasted.save(join(targetPath, f+'4.jpg'))

    # 改变锐度 sha_
    enh_sha = ImageEnhance.Sharpness(image)
    sharpness = 3.0
    image_sharp = enh_sha.enhance(sharpness)
    # image_sharp.show()
    image_sharp.save(join(targetPath, f+'5.jpg'))

    # # y方向上的缩放 yre_
    # # image.show()
    # w = image.width
    # h = image.height
    # print(w, h)
    # out_ww = image.resize((w, h + 40))  # 拉伸成高为h的正方形
    # # out_ww.show()
    # out_ww_1 = np.array(out_ww)
    # out_w_2 = out_ww_1[30:(h - 10), 0:w]  # 开始的纵坐标，开始的横坐标
    # out_w_2 = Image.fromarray(out_w_2)
    # # out_w_2.show()
    # out_w_2.save(join(targetPath, f+'6.jpg'))
    #
    # # x方向上的缩放 xre_
    # # image.show()
    # out_hh = image.resize((w + 80, h))  # 拉伸成宽为w的正方形,width,height
    # # out_hh.show()
    # out_hh_1 = array(out_hh)
    # out_h_2 = out_hh_1[0:h, 40:(w + 40)]
    # out_h_2 = Image.fromarray(out_h_2)
    # # out_h_2.show()
    # out_h_2.save(join(targetPath, f+'7.jpg'))
    #
    # # x左方向的平移 xl_
    # # 平移矩阵[[1,0,-10]，[0,1,-12]]
    # image.show()
    # w = image.width
    # h = image.height
    # M = np.array([[1, 0, -80], [0, 1, 0]], dtype=np.float32)
    # image_cv_change = cv2.warpAffine(image_cv, M, (w, h))
    # image_cv_change_RGB = cv2.cvtColor(image_cv_change, cv2.COLOR_BGR2RGB)
    # image_cv_change = Image.fromarray(image_cv_change_RGB)
    # # image_cv_change.show()
    # image_cv_change.save(join(targetPath, f+'8.jpg'))
    #
    # # x右方向的平移 xr_
    # # image.show()
    # w = image_cv.width
    # h = image_cv.height
    # M = np.array([[1, 0, 80], [0, 1, 0]], dtype=np.float32)
    # image_cv_change = cv2.warpAffine(image_cv, M, (w, h))
    # image_cv_change_RGB = cv2.cvtColor(image_cv_change, cv2.COLOR_BGR2RGB)
    # image_cv_change = Image.fromarray(image_cv_change_RGB)
    # # image_cv_change.show()
    # image_cv_change.save(join(targetPath, f+'9.jpg'))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('imagepath', type=str, help='Directory with positive (Moiré pattern) images.')
    parser.add_argument('enhancepath', type=str, help='Directory with negative (Normal) images.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    # main(parse_arguments(sys.argv[1:]))\
    # main(parse_arguments(['F:/ta/摩尔纹课题/src/positiveImages',
    #                       'F:/ta/摩尔纹课题/src/negativeImages',
    #                       0]))
    main(parse_arguments(['./data/moire',
                          './data/enhancem']))

