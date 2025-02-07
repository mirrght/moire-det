import argparse
import heapq
import os
import shutil
import sys
import io
import base64

# import cv2
import PIL
import scipy
import numpy as np
import streamlit as st
from test import *
from GAN_MPD import test_main
# from train import train_main
# from Direct import *
from PIL import Image
# from streamlit_pandas_profiling import st_profile_report
from PIL import Image
from PIL import Image
from streamlit_option_menu import option_menu

# from streamlit_card import card
# from streamlit_extras.mandatory_date_range import date_range_picker
# from streamlit_extras.metric_cards import style_metric_cards

def extract_features(image):
    img = np.array(image)
    h, w, q = img.shape
    img = img.astype(float)
    image_max = np.max(img, axis=2)
    image_max_labol = np.zeros([h, w])
    for x in range(0, h):
        for y in range(0, w):
            for z in range(0, q):
                if img[x][y][z] == image_max[x][y]:
                    image_max_labol[x][y] = z / 2
    return image_max_labol

    # 提取特征图

# 假设特征提取函数为 extract_features
def makwenf(image):
    img = np.array(image)
    img = np.mean(img, axis=2)
    h, w = img.shape
    img = img.astype(float)

    size = 200  # 假设矩形区域的大小为 100x100

    # # 应用高斯模糊进行预处理（由找点变成找区域）
    # gray = cv2.GaussianBlur(img, (size - 1, size - 1), 0)
    # # 利用cv2.minMaxLoc寻找到图像中最亮和最暗的区域
    # (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    pil_image = PIL.Image.fromarray(img).convert('L')  # 转换为灰度图像
    blurred_image = pil_image.filter(PIL.ImageFilter.GaussianBlur(radius=(size / 2)))

    # 转换回 NumPy 数组
    gray = np.array(blurred_image)

    # 寻找图像中最亮和最暗的区域
    minVal = np.min(gray)
    maxVal = np.max(gray)

    # 找到最亮和最暗区域的坐标
    minLoc = np.unravel_index(np.argmin(gray), gray.shape)
    maxLoc = np.unravel_index(np.argmax(gray), gray.shape)

    x,y = maxLoc  # maxLoc 是最大值位置的坐标
    half_size = size // 2

    # 边界检查，确保矩形区域不会超出图像范围
    x1 = max(0, x - half_size)
    y1 = max(0, y - half_size)
    x2 = min(h - 1, x + half_size)
    y2 = min(w - 1, y + half_size)

    # 提取矩形区域
    rectangular_region = image[x1:x2, y1:y2]

    return rectangular_region


def detect_edges(gray_image):
    # 定义滤波器
    filter_kernel = np.array([[1, 1, 1],
                              [1, -8, 1],
                              [1, 1, 1]])

    # 进行滤波操作
    # filtered_image = cv2.filter2D(gray_image, -1, filter_kernel)
    filtered_image = scipy.signal.convolve2d(gray_image, filter_kernel, mode='same', boundary='wrap')

    # 将非零值设为1，零值设为0
    edges_image = np.where(filtered_image == 0, 0, 1)
    p = np.sum(edges_image) / (edges_image.shape[0] * edges_image.shape[1])

    #     plt.imshow(edges_image, cmap='gray')
    #     plt.show()
    return p

# 设置应用程序主题
st.set_page_config(page_title="My Streamlit App", layout="wide", initial_sidebar_state="expanded")

# 设置页面标题
# 使用CSS来设置标题颜色
# st.title("基于占优RGB通道特征的摩尔纹检测系统")


# 主体函数开始

image_sign = Image.open('./figures/KDD25-Logo.png')

st.sidebar.image(image_sign)


placeholder = st.empty()


# 侧边栏
with st.sidebar:
    st.markdown("Dominant RGB Channel Feature Moiré Pattern Detection (DCFMD) ")
    st.write("some examples")

    selected = option_menu("Functions", [ 'Dominant RGB Channel Coding','DIRECT Method','MCNN-MPD Method','GAN-MPD Method'],
        icons=['bi bi-boxes','list-task','bi-bezier','bi-box-arrow-in-up-right'], menu_icon="cast", default_index=1)


if selected == 'Dominant RGB Channel Coding':

    placeholder.empty()
    with placeholder.container():
        st.info('Dominant RGB Channel Coding')
        # 加载本地图像

        image_path = './figures/MRBI_2_m.png'  # 本地图片路径

        image = Image.open(image_path)
        image_array = np.array(image)

        # 假设特征提取函数为 extract_features


        feature_map = extract_features(image_array)

        # 显示原图和特征图
        st.subheader("图像展示")
        col1, col2 = st.columns(2)

        with col1:
            st.write("### 原图")
            st.image(image, caption="原图", use_column_width=True)

        with col2:
            st.write("### 特征图")
            st.image(feature_map, caption="特征图", use_column_width=True)

        # 添加 JavaScript 实现交互功能
        # 添加 JavaScript 实现交互功能
        # 添加 JavaScript 实现交互功能
        st.markdown("""
            <script>
            function updateFeatureMap(x, y) {
                const featureMap = document.querySelector(".stImage:nth-of-type(2) img");
                const rect = featureMap.getBoundingClientRect();
                const scaleX = featureMap.naturalWidth / rect.width;
                const scaleY = featureMap.naturalHeight / rect.height;

                const size = 100; // 区域大小
                const startX = Math.max(0, x * scaleX - size / 2);
                const startY = Math.max(0, y * scaleY - size / 2);
                const endX = Math.min(featureMap.naturalWidth, startX + size);
                const endY = Math.min(featureMap.naturalHeight, startY + size);

                // 创建一个 canvas 来显示局部区域
                const canvas = document.createElement('canvas');
                canvas.width = size;
                canvas.height = size;
                const ctx = canvas.getContext('2d');

                ctx.drawImage(featureMap, startX, startY, size, size, 0, 0, size, size);
                document.getElementById("cropped-feature-map").innerHTML = "";
                document.getElementById("cropped-feature-map").appendChild(canvas);
            }

            document.addEventListener("DOMContentLoaded", function() {
                const featureMapImg = document.querySelector(".stImage:nth-of-type(2) img");
                if (featureMapImg) {
                    featureMapImg.addEventListener("mousemove", (event) => {
                        const rect = event.target.getBoundingClientRect();
                        const x = event.clientX - rect.left;
                        const y = event.clientY - rect.top;
                        updateFeatureMap(x, y);
                    });
                }
            });
            </script>
            """, unsafe_allow_html=True)

        # 显示特征图局部区域
        st.write("### 特征图局部区域")
        st.markdown('<div id="cropped-feature-map"></div>', unsafe_allow_html=True)

elif selected == 'DIRECT Method':

    placeholder.empty()
    with placeholder.container():
        st.info('DIRECT Method')

        image_path = './figures/MRBI_2_m.png'  # 本地图片路径

        image = Image.open(image_path)
        image_array = np.array(image)

        image_wen = makwenf(image_array)
        feature_map = extract_features(image_wen)

        P = detect_edges(feature_map)

        # 显示图像
        st.subheader("图像展示")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("### 原图")
            st.image(image, caption="原图", use_column_width=True)

        with col2:
            st.write("### 局部图")
            st.image(image_wen, caption="局部图", use_column_width=True)

        with col3:
            st.write("### 局部图特征图")
            st.image(feature_map, caption="特征图", use_column_width=True)

        if P > 0.1:
            st.write("### It is a moire image.")
        else:
            st.write("### It is a normal image.")




        # def image_to_base64(image):
        #     if isinstance(image, np.ndarray):
        #         image = Image.fromarray(image)
        #
        #     # 确保图像是 RGB 模式
        #     if image.mode != 'RGB':
        #         image = image.convert('RGB')
        #
            # buffered = io.BytesIO()
            # image.save(buffered, format="PNG")
            # return base64.b64encode(buffered.getvalue()).decode()

        # # 添加箭头和框的说明
        # st.markdown("""
        #     <div style="text-align: center;">
        #         <div style="display: inline-block; position: relative;">
        #             <img src="data:image/png;base64,{}" style="width: 100px; height: auto;"/>
        #             <div style="position: absolute; top: 40px; left: 70px; width: 50px; height: 2px; background-color: black;"/>
        #             <div style="position: absolute; top: 90px; left: 80px; transform: rotate(45deg); width: 20px; height: 2px; background-color: black;"/>
        #             <div style="position: absolute; top: 90px; left: 80px; transform: rotate(-45deg); width: 20px; height: 2px; background-color: black;"/>
        #         </div>
        #         <p>原图 → 局部图</p>
        #     </div>
        #     <div style="text-align: center;">
        #         <div style="display: inline-block; position: relative;">
        #             <img src="data:image/png;base64,{}" style="width: 100px; height: auto;"/>
        #             <div style="position: absolute; top: 40px; left: 70px; width: 50px; height: 2px; background-color: black;"/>
        #             <div style="position: absolute; top: 90px; left: 80px; transform: rotate(45deg); width: 20px; height: 2px; background-color: black;"/>
        #             <div style="position: absolute; top: 90px; left: 80px; transform: rotate(-45deg); width: 20px; height: 2px; background-color: black;"/>
        #         </div>
        #         <p>局部图 → 特征图</p>
        #     </div>
        #     """.format(image_to_base64(image), image_to_base64(feature_map)), unsafe_allow_html=True)
        #
        # # 添加进一步分析或说明的内容
        # st.write("### 分析说明")
        # st.write("在此处添加关于原图、局部图和特征图的分析和说明。")




elif selected == 'MCNN-MPD Method':

    placeholder.empty()
    with placeholder.container():
        st.info('MCNN-MPD Method')

        dataset = st.selectbox("请选择数据集名称:", ("VINmoire", "MoireFace", "MRBI", "FHDMi"), key='my_selectbox')
        folder_path = './figures/' + dataset
        # 检查用户是否输入了文件夹目录
        if dataset:
            # 在界面中显示输入的文件夹目录路径
            st.write("数据集名称", './figures/' + dataset)

            # 添加按钮，触发检测操作
            if st.button("开始检测"):
                # 调用摩尔纹检测方法对文件夹中的图像进行检测
                main(parse_arguments(['./model/CNN_MPD_' + dataset + '.h5',  # 模型h5文件
                                      folder_path,  # 测试摩尔纹图片目录
                                      './empty']))  # 测试正常图片目录

                # 在界面中显示检测结果
                st.write("检测结果：")
                # 指定摩尔纹图片的路径
                # 指定文件夹路径
                folder_path = r"./moire pattern"

                # 遍历文件夹中的所有文件
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    if os.path.isfile(file_path):
                        # 读取图片
                        image = Image.open(file_path)

                        # 在界面中显示图片
                        st.image(image, caption=filename, width=300)

elif selected == 'GAN-MPD Method':

    placeholder.empty()
    with placeholder.container():
        st.info('GAN-MPD Method')

        dataset = st.selectbox("请选择数据集名称:", ("VINmoire", "MoireFace", "MRBI", "FHDMi"), key='my_selectbox')
        folder_path = './figures/' + dataset
        # 检查用户是否输入了文件夹目录
        if dataset:
            # 在界面中显示输入的文件夹目录路径
            st.write("数据集名称", './figures/' + dataset)

            # 添加按钮，触发检测操作
            if st.button("开始检测"):
                # 调用摩尔纹检测方法对文件夹中的图像进行检测
                test_main(parse_arguments(['./model/GAN_MPD_' + dataset + '.h5',  # 模型h5文件
                                      folder_path,  # 测试摩尔纹图片目录
                                      './empty']))  # 测试正常图片目录

                # 在界面中显示检测结果
                st.write("检测结果：")
                # 指定摩尔纹图片的路径
                # 指定文件夹路径
                folder_path = r"./moire pattern"

                # 遍历文件夹中的所有文件
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    if os.path.isfile(file_path):
                        # 读取图片
                        image = Image.open(file_path)

                        # 在界面中显示图片
                        st.image(image, caption=filename, width=300)

# # 使用自定义样式
# st.markdown('<style>h1 {color: lightblue;}</style>', unsafe_allow_html=True)
# st.markdown('<style>body{background-color: #f1f1f1;}</style>', unsafe_allow_html=True)
#
# # 创建一个选择框
# method = st.selectbox("请选择检测方法:", ("DIRECT", "MCNN-MPD"), key='my_selectbox')
#
# # 根据用户选择显示不同的内容
# if method == "DIRECT":
#     st.header("DIRECT")
#     # 根据用户选择显示不同的内容
#
#     folder_path = st.text_input("请输入需检测文件夹目录")
#
#     # 检查用户是否输入了文件夹目录
#     if folder_path:
#         # 在界面中显示输入的文件夹目录路径
#         st.write("输入的文件夹目录路径：", folder_path)
#
#         # 添加按钮，触发检测操作
#         if st.button("开始检测"):
#             # 调用摩尔纹检测方法对文件夹中的图像进行检测
#             Direct_main(folder_path, './empty')
#
#             # 在界面中显示检测结果
#             st.write("检测结果：")
#             # 指定摩尔纹图片的路径
#             # 指定文件夹路径
#             folder_path = r"./moire pattern"
#
#             # 遍历文件夹中的所有文件
#             for filename in os.listdir(folder_path):
#                 file_path = os.path.join(folder_path, filename)
#                 if os.path.isfile(file_path):
#                     # 读取图片
#                     image = Image.open(file_path)
#
#                     # 在界面中显示图片
#                     st.image(image, caption=filename, width=300)
#
# elif method == "MCNN-MPD":
#     st.header("MCNN-MPD")
#
#     # 添加复选框，允许用户选择是否重新训练模型
#     retrain_model = st.checkbox("重新训练模型")
#
#     # 如果复选框被选中，则显示重新训练的消息
#     if retrain_model:
#
#         training_path1 = st.text_input("输入用于训练的摩尔纹图片的路径")
#         training_path2 = st.text_input("输入用于训练的正常图片的路径")
#
#         # 如果用户输入了路径，则显示输入的路径
#         if training_path2:
#             st.write("输入的路径：", training_path1+","+training_path2)
#
#             st.write("正在重新训练模型...")
#
#             train_main(parse_arguments([training_path1,  #用于训练的摩尔纹图片目录
#                           training_path2,    #用于训练的正常图片目录
#                           './trainDataPositive',     #用于训练的摩尔纹图片特征图层目录
#                           './trainDataNegative',     #用于训练的正常图片特征图层目录
#                           '50']))
#
#
#             st.write("训练完成")
#
#     # 添加文件夹目录输入框
#     folder_path = st.text_input("请输入需检测文件夹目录")
#
#     # 检查用户是否输入了文件夹目录
#     if folder_path:
#         # 在界面中显示输入的文件夹目录路径
#         st.write("输入的文件夹目录路径：", folder_path)
#
#         # 添加按钮，触发检测操作
#         if st.button("开始检测"):
#             # 调用摩尔纹检测方法对文件夹中的图像进行检测
#             main(parse_arguments(['./DCFMD.h5',  # 模型h5文件
#                                   folder_path,  # 测试摩尔纹图片目录
#                                   './empty']))  # 测试正常图片目录
#
#             # 在界面中显示检测结果
#             st.write("检测结果：")
#             # 指定摩尔纹图片的路径
#             # 指定文件夹路径
#             folder_path = r"./moire pattern"
#
#             # 遍历文件夹中的所有文件
#             for filename in os.listdir(folder_path):
#                 file_path = os.path.join(folder_path, filename)
#                 if os.path.isfile(file_path):
#                     # 读取图片
#                     image = Image.open(file_path)
#
#                     # 在界面中显示图片
#                     st.image(image, caption=filename, width=300)

