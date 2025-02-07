#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:1'
# config=tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True
# sess=tf.compat.v1.Session(config=config) 
# 卷积神经网络需要使用GPU因此需要在这里声明


# In[2]:


# tf.config.list_physical_devices('GPU')

# tf.debugging.set_log_device_placement(True)
 


# In[3]:


# mirrored_strategy = tf.distribute.MirroredStrategy()


# ### 导入各个库，以及库中所需模块

# In[4]:


from tensorflow import keras
import numpy as np
# from utils import set_soft_gpu
from tensorflow.keras.layers import Dense, Reshape, Input, Embedding,BatchNormalization,ReLU,Conv2DTranspose,Conv2D,LeakyReLU,Dropout,Convolution2D, MaxPooling2D, Activation, Flatten, Add, Multiply, Maximum, concatenate
import time
# import import_ipynb
from createTrainingData import createTrainingData

#导入需要的python库


# ### 记录实验中的各个规定好的数据size  

# In[5]:


noise_dim = 100 #生成的输入的噪声维度
image_shape = (56, 36, 1) #图像的尺寸
label_dim = 10 #图像对应标签二值化为长度为10的向量
batchsize = 64 #一次带入数据的批次
epoch = 20 #迭代次数


# In[6]:


import pickle
# 保存列表到文件
def save_list_to_file(lst, filename):
    with open(filename, 'wb') as f:
        pickle.dump(lst, f)

# 从文件中加载列表
def load_list_from_file(filename):
    with open(filename, 'rb') as f:
        lst = pickle.load(f)
    return lst


# ### 生成器模型
# 采用了一种有技巧性的方法 使用keras的两种模型搭建方式结合

# In[7]:


def get_generator(img_shape):
    
    num_classes=2
    
    height, width, depth = img_shape
    kernel_size_1 = 7 # 7x7 kernels
    kernel_size_2 = 3 # 3x3 kernels
    pool_size = 2 # 2x2 pooling
    conv_depth_1 = 32 #  每个卷积层 32 个核
    conv_depth_2 = 16 # 之后16个核
    drop_prob_1 = 0.25 # 卷积层 dropout 概率0.25
    drop_prob_2 = 0.5 # 全连接层 dropout 概率 0.5
    hidden_size = 8 # 全连接层神经元

    inpV = Input(shape=(height, width, depth))
    inpML = Input(shape=(height, width, depth))
    inpHH = Input(shape=(height, width, depth))
    inpHS = Input(shape=(height, width, depth))

    conv_1_V = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same', activation='relu')(inpV)
    conv_1_ML = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same', activation='relu')(inpML)
    conv_1_HH = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same', activation='relu')(inpHH)
    conv_1_HS = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same', activation='relu')(inpHS)
    pool_1_V = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_V)
    pool_1_ML = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_ML)
    pool_1_HH = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_HH)
    pool_1_HS = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_HS)

    avg_ML_HH_HS = Maximum()([pool_1_ML, pool_1_HH, pool_1_HS])
    inp_merged = Multiply()([pool_1_V, avg_ML_HH_HS])
    C4 = Convolution2D(conv_depth_1, (kernel_size_2, kernel_size_2), padding='same', activation='relu')(inp_merged)
    S2 = MaxPooling2D(pool_size=(2, 2))(C4)
    drop_1 = Dropout(drop_prob_1)(S2)
    C5 = Convolution2D(conv_depth_2, (kernel_size_2, kernel_size_2), padding='same', activation='relu')(drop_1)
    S3 = MaxPooling2D(pool_size=(pool_size, pool_size))(C5)
    C6 = Convolution2D(conv_depth_2, (kernel_size_2, kernel_size_2), padding='same', activation='relu')(S3)
    S4 = MaxPooling2D(pool_size=(pool_size, pool_size))(C6)
    drop_2 = Dropout(drop_prob_1)(S4)

    flat = Flatten()(drop_2)
    hidden = Dense(32, activation='relu')(flat)
    hidden = Dense(16, activation='relu')(hidden)
    hidden = Dense(hidden_size, activation='relu')(hidden)
    drop_3 = Dropout(drop_prob_2)(hidden)
    out1 = Dense(num_classes,activation='softmax')(drop_3)
    
    
    model = keras.Model(inputs=[inpV, inpML, inpHH, inpHS], outputs=[out1], name="generator")
    model.summary()
    return model


# def get_generator(height, width, depth, num_classes):
#     kernel_size_1 = 7  # 7x7 kernels
#     kernel_size_2 = 3  # 3x3 kernels
#     pool_size = 2  # 2x2 pooling
#     conv_depth_1 = 32  # 每个卷积层 32 个核
#     conv_depth_2 = 16  # 之后16个核
#     drop_prob_1 = 0.25  # 卷积层 dropout 概率0.25
#     drop_prob_2 = 0.5  # 全连接层 dropout 概率 0.5
#     hidden_size = 16  # 全连接层神经元
# 
#     inpV = Input(shape=(height, width, depth))
#     inpML = Input(shape=(height, width, depth))
#     inpHH = Input(shape=(height, width, depth))
#     inpHS = Input(shape=(height, width, depth))
# 
#     conv_1_V = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same', activation='relu')(inpV)
#     conv_1_ML = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same', activation='relu')(inpML)
#     conv_1_HH = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same', activation='relu')(inpHH)
#     conv_1_HS = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same', activation='relu')(inpHS)
#     pool_1_V = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_V)
#     pool_1_ML = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_ML)
#     pool_1_HH = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_HH)
#     pool_1_HS = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_HS)
# 
#     avg_ML_HH_HS = Maximum()([pool_1_ML, pool_1_HH, pool_1_HS])
#     inp_merged = Multiply()([pool_1_V, avg_ML_HH_HS])
#     C4 = Convolution2D(conv_depth_1, (kernel_size_2, kernel_size_2), padding='same', activation='relu')(inp_merged)
#     S2 = MaxPooling2D(pool_size=(4, 4))(C4)
#     drop_1 = Dropout(drop_prob_1)(S2)
#     C5 = Convolution2D(conv_depth_2, (kernel_size_2, kernel_size_2), padding='same', activation='relu')(drop_1)
#     S3 = MaxPooling2D(pool_size=(pool_size, pool_size))(C5)
#     C6 = Convolution2D(conv_depth_2, (kernel_size_2, kernel_size_2), padding='same', activation='relu')(S3)
#     S4 = MaxPooling2D(pool_size=(pool_size, pool_size))(C6)
#     drop_2 = Dropout(drop_prob_1)(S4)
# 
#     flat = Flatten()(drop_2)
#     hidden = Dense(hidden_size, activation='relu')(flat)
#     drop_3 = Dropout(drop_prob_2)(hidden)
#     out = Dense(num_classes, activation='softmax')(drop_3)
# 
#     model = Model(inputs=[inpV, inpML, inpHH, inpHS], outputs=out, name="generator")
# 
#     return model

# ### 实例化生成器

# In[8]:


generator = get_generator(image_shape) #给定输入数据的shape


# ### 判别器模型

# In[9]:


def get_discriminator(img_shape):
    
  
    num_classes=2
    
    height, width, depth = img_shape
    kernel_size_1 = 7 # 7x7 kernels
    kernel_size_2 = 3 # 3x3 kernels
    pool_size = 2 # 2x2 pooling
    conv_depth_1 = 32 #  每个卷积层 32 个核
    conv_depth_2 = 16 # 之后16个核
    drop_prob_1 = 0.25 # 卷积层 dropout 概率0.25
    drop_prob_2 = 0.5 # 全连接层 dropout 概率 0.5
    hidden_size = 8 # 全连接层神经元

    inpV = Input(shape=(height, width, depth))
    inpML = Input(shape=(height, width, depth))
    inpHH = Input(shape=(height, width, depth))
    inpHS = Input(shape=(height, width, depth))
    
    label = Input(shape=(num_classes))
    
    conv_1_V = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same', activation='relu')(inpV)
    conv_1_ML = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same', activation='relu')(inpML)
    conv_1_HH = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same', activation='relu')(inpHH)
    conv_1_HS = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same', activation='relu')(inpHS)
    pool_1_V = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_V)
    pool_1_ML = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_ML)
    pool_1_HH = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_HH)
    pool_1_HS = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_HS)

    avg_ML_HH_HS = Maximum()([pool_1_ML, pool_1_HH, pool_1_HS])
    inp_merged = Multiply()([pool_1_V, avg_ML_HH_HS])

    C7 = Convolution2D(conv_depth_1, (kernel_size_2, kernel_size_2), padding='same', activation='relu')(inp_merged)
    S5 = MaxPooling2D(pool_size=(4, 4))(C7)
    drop_4 = Dropout(drop_prob_1)(S5)
    C8 = Convolution2D(conv_depth_2, (kernel_size_2, kernel_size_2), padding='same', activation='relu')(drop_4)
    S6 = MaxPooling2D(pool_size=(pool_size, pool_size))(C8)
    C9 = Convolution2D(conv_depth_2, (kernel_size_2, kernel_size_2), padding='same', activation='relu')(S6)
    S7 = MaxPooling2D(pool_size=(pool_size, pool_size))(C9)
    drop_5 = Dropout(drop_prob_1)(S7)

    flat1 = Flatten()(drop_5)
    hidden1 = Dense(hidden_size, activation='relu')(flat1)
    
    drop_6 = Dropout(drop_prob_2)(hidden1)
    hidden2 = Dense(4, activation='relu')(drop_6)    
    conl = concatenate([hidden2, label])
    hidden3 = Dense(2, activation='relu')(conl) 
    out2 = Dense(1, activation='sigmoid')(hidden3)

    model = keras.Model(inputs=[inpV, inpML,inpHH, inpHS,label], outputs=[out2], name="discriminator")
    model.summary()
    return model


# ### 实例化判别器模型

# In[10]:


discriminator = get_discriminator(image_shape)


# ### 损失函数 交叉熵 wgan_gp

# In[11]:


cross_entropy = keras.losses.BinaryCrossentropy(from_logits=False)
#from_logits=True代表着y_pre不是概率值 在损失函数中计算交叉熵之前会首先执行一步sortmax


# In[12]:


from functools import partial

def d_loss_fn(real_logits,fake_logits):
    return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

def g_loss_fn(fake_logits):
    return -tf.reduce_mean(fake_logits)

    return d_loss_fn, g_loss_fn

def gradient_penalty(discriminator, real_img_V,real_img_ML,real_img_HH,real_img_HS, real_img_label, fake_iamge_label):
    
    real_img_label = tf.cast(real_img_label, tf.float32)
    fake_iamge_label = tf.cast(fake_iamge_label, tf.float32)
    alpha = tf.random.uniform((len(real_img_label), 1), 0., 1.)
    diff = real_img_label - fake_iamge_label
    inter = real_img_label + (alpha * diff)
    with tf.GradientTape() as tape:
        tape.watch(inter)
        predictions = discriminator([real_img_V,real_img_ML,real_img_HH,real_img_HS,inter], training=True)
    gradients = tape.gradient(predictions, [inter])[0]
#     slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),axis = [1,2,3]))
    return tf.reduce_mean((gradients - 1.) ** 2)


# In[13]:


#准确率函数
def binary_accuracy(label, pred):
    acc = tf.keras.metrics.BinaryAccuracy()
    acc.reset_states()
    acc.update_state(label, pred)
    return acc.result()


# ### 实例化优化器

# In[14]:


# generator_optimizer = keras.optimizers.legacy.Adam(0.0002, beta_1=0.5)
# discriminator_optimizer = keras.optimizers.legacy.Adam(0.0002, beta_1=0.5)
#
#
# # In[15]:
#
#
# def step( real_img_V,real_img_ML,real_img_HH,real_img_HS, real_img_label,batchsize):
#
#     real_d_label = tf.ones([len(real_img_label),1], tf.float32)# 代表生成图像输入判别器希望得到的标签
# #     real_img_label = tf.reshape(real_img_label,(real_img_label.shape[0],))
#
#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#
#         fake_iamge_label = generator([real_img_V,real_img_ML,real_img_HH,real_img_HS], training=True)#根据图像得到标签
#         d_fake = discriminator([real_img_V,real_img_ML,real_img_HH,real_img_HS, fake_iamge_label], training=False)#根据图像和生成的标签得到判别器判别的结果
#
# #         generator_loss = cross_entropy(real_d_label,d_fake )+cross_entropy(real_img_label,fake_iamge_label )#生成器的损失
#         generator_loss = g_loss_fn(d_fake )+cross_entropy(real_img_label,fake_iamge_label )#生成器的损失
#         generator_accuracy = binary_accuracy(real_img_label,fake_iamge_label )#生成器准确率
#         generator_grads = gen_tape.gradient(generator_loss, generator.trainable_variables)#生成器关于损失函数求梯度
#         generator_optimizer.apply_gradients(zip(generator_grads, generator.trainable_variables))#生成器更新参数
#
#         img_V = tf.concat((real_img_V, real_img_V), axis=0)
#         img_ML = tf.concat((real_img_ML, real_img_ML), axis=0)
#         img_HH = tf.concat((real_img_HH, real_img_HH), axis=0)
#         img_HS = tf.concat((real_img_HS, real_img_HS), axis=0)
#         img_label = tf.concat((real_img_label, fake_iamge_label), axis=0)#生成和真实标签
#
#
#         d_label = tf.concat((tf.ones((len(real_img_label), 1), tf.float32), tf.zeros((len(real_img_label), 1), tf.float32)), axis=0)#判别器想要得到的判别结果
#         pred = discriminator([img_V, img_ML, img_HH, img_HS, img_label], training=True)#判别器输出的结果
#
# #         discriminator_loss = cross_entropy(d_label, pred)#判别器的损失值
#
#         d_real = discriminator([ real_img_V,real_img_ML,real_img_HH,real_img_HS, real_img_label], training=True)
#         discriminator_loss = d_loss_fn(d_real, d_fake)#判别器的损失值
#         gp = gradient_penalty(discriminator,real_img_V,real_img_ML,real_img_HH,real_img_HS, real_img_label, fake_iamge_label)
#         discriminator_loss += gp * 1.0
#
#         discriminator_accuracy = binary_accuracy(d_label, pred)#判别器准确率
#         discriminator_grads = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables)#判别器求解梯度
#         discriminator_optimizer.apply_gradients(zip(discriminator_grads, discriminator.trainable_variables))#判别器更新梯度
#
#     return discriminator_loss, discriminator_accuracy, generator_loss, generator_accuracy, fake_iamge_label
#
#
# # In[16]:
#
#
# def train( dataset,EPOCH,BATCH_SIZE):
#     t0 = time.time()
# #     loss = np.zeros(num_epochs)
#     v_accm = 0
#     dataset1=dataset.shuffle(16)
#     ds=dataset1.batch(BATCH_SIZE//2)
#
#     for ep in range(EPOCH):
#
#         for t, (real_img_V,real_img_ML,real_img_HH,real_img_HS, real_img_label) in enumerate(ds):
#             if t ==1:
#                 fake_iamge_label = generator([real_img_V,real_img_ML,real_img_HH,real_img_HS], training=True)
#                 v_acc = binary_accuracy(real_img_label,fake_iamge_label )
# #                 if v_acc > v_accm:
# #                     v_accm = v_acc
# #                     print("ep={}".format(v_accm))
# #                     fi = './model/GAN_MPD_'+dataset_name+'.h5'
# #                     generator.save(fi)
#             else:
#                 d_loss, d_acc, g_loss, g_acc, g_img_label = step(real_img_V,real_img_ML,real_img_HH,real_img_HS, real_img_label,BATCH_SIZE)
#             if t % 40 == 0:
#                 t1 = time.time()
#                 print("ep={} | time={:.1f} | t={} | d_acc={:.2f} | g_acc={:.2f} | d_loss={:.2f} | g_loss={:.2f}".format(
#                     ep, t1-t0, t, d_acc.numpy(), g_acc.numpy(), d_loss.numpy(), g_loss.numpy(), ))
#                 t0 = t1
#     fi = './model/GAN_MPD_'+dataset_name+'.h5'
#     generator.save(fi)


# In[29]:


from train import *

def main(args):
    positiveImagePath = (args.positiveImages)
    negativeImagePath = (args.negativeImages)
    numEpochs = (args.epochs)
    positiveTrainImagePath = args.trainingDataPositive
    negativeTrainImagePath = args.trainingDataNegative
    
#     createTrainingData(positiveImagePath, negativeImagePath, positiveTrainImagePath, negativeTrainImagePath)

    # 读取特征图片
    X_V, X_ML, X_HH, X_HS,X_index, Y, imageCount = readDatatz(positiveImagePath, negativeImagePath, positiveTrainImagePath, negativeTrainImagePath)

    #切分训练集测试集
    X_V_train,X_ML_train,X_HH_train,X_HS_train,Y_train,X_V_test,X_ML_test,X_HH_test,X_HS_test,Y_test = trainTestSplit(X_V, X_ML, X_HH, X_HS, X_index, Y, imageCount)
    
    Y_train = np_utils.to_categorical(Y_train, 2) # One-hot encode the labels
#     Y_test = np_utils.to_categorical(Y_test, 2) # One-hot encode the labels
    
    dataset = tf.data.Dataset.from_tensor_slices((X_V_train,X_ML_train,X_HH_train,X_HS_train,Y_train))
    
    LATENT_DIM = 100
    IMG_SHAPE = (56, 36, 1)
    LABEL_DIM = 10
    BATCH_SIZE = 32
    EPOCH = 20

    train(dataset,numEpochs,BATCH_SIZE)
    
    evaluate(generator, X_V_test,X_ML_test,X_HH_test,X_HS_test,Y_test)
    Y_pre=generator([ X_V_train,X_ML_train,X_HH_train,X_HS_train])
    generator_accuracy = binary_accuracy(Y_train,Y_pre )
    print(generator_accuracy)
    
if __name__ == '__main__':
    
    dataset_name = 'MoireFace'  # VINmoire MRBI FHDMi MoireFace

    folder_path1 = '../data/' + dataset_name + '/train/positive'
    folder_path2 = '../data/' + dataset_name + '/train/negative'
    main(parse_arguments([folder_path1,  # 用于训练的摩尔纹图片目录
                                folder_path2,  # 用于训练的正常图片目录
                                './trainDataPositive',  # 用于训练的摩尔纹图片特征图层目录
                                './trainDataNegative',  # 用于训练的正常图片特征图层目录
                                '80']))
#     main(parse_arguments(['./data/MoireIDT/positive/screen',#MoireFace FHDMi MRBI MoireIDT /screen
#                           './data/MoireIDT/negative',
#                           './trainDataPositive',
#                           './trainDataNegative',
#                           '10']))
    


# In[16]:


from test import *

positiveTestImagePath = './testDataPositive'
negativeTestImagePath = './testDataNegative'


def evaluate(model, X_V_test,X_ML_test,X_HH_test,X_HS_test,y_test,fileNames,positiveImagePath, negativeImagePath):

    model_out = model.predict([X_V_test,X_ML_test,X_HH_test,X_HS_test])
    passCnt = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # filename1 = '../output/'+ method_name + '_' + dataset_name
    # save_list_to_file(model_out[:,1], filename1)
    # 
    # filename2 = '../output/'+ method_name + '_' + dataset_name +'_y'
    # save_list_to_file(y_test, filename2)


    directory = './moire pattern'

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        # 如果是文件，删除文件
        if os.path.isfile(file_path):
            os.remove(file_path)

    for i in range(len(y_test)):

        print(model_out[i, :])
        print(y_test[i])
        print(',')

        if np.argmax(model_out[i, :]) == y_test[i]:
            str_label='Pass'
            passCnt = passCnt + 1
        else:
            str_label='Fail'

        if y_test[i] ==0:
            if np.argmax(model_out[i, :]) == y_test[i]:
                TP = TP + 1
            else:
                FN = FN + 1
        else:
            if np.argmax(model_out[i, :]) == y_test[i]:
                TN = TN + 1
            else:
                FP = FP + 1

        if np.argmax(model_out[i, :]) == 0:

            # makwen(fileNames[i],positiveImagePath, negativeImagePath)
            try:
                image = Image.open(join(positiveImagePath, fileNames[i]))

            except Exception:
                image = Image.open(join(negativeImagePath, fileNames[i]))

            image.save(directory +'/'+ fileNames[i])


             

    start = "\033[1m"
    end = "\033[0;0m"
    print(start + 'confusion matrix (test / validation)' + end)
    print(start + 'true positive:  '+ end + str(TP))
    print(start + 'false positive: '+ end + str(FP))
    print(start + 'true negative:  '+ end + str(TN))
    print(start + 'false negative: '+ end + str(FN))
    print('\n')
    print(start + 'accuracy:  ' + end + "{:.4f} %".format(100*(TP+TN)/(TP+FP+FN+TN)))
    print(start + 'precision: ' + end + "{:.4f} %".format(100*TP/(TP + FP)))
    print(start + 'recall:  ' + end + "{:.4f} %".format(100*TP/(TP + FN)))
    print('\n')
    print('predict label:'+ end + str(np.around(model_out[:,0])))


def test_main(args):
    weights_file = (args.weightsFile)
    positiveImagePath = (args.positiveTestImages)
    negativeImagePath = (args.negativeTestImages)
    
    createTrainingData(positiveImagePath, negativeImagePath, positiveTestImagePath, negativeTestImagePath)

    #读取特征
    X_V, X_ML, X_HH, X_HS, X_index, Y, imageCount, fileNames = readDatatz(positiveImagePath, negativeImagePath, positiveTestImagePath, negativeTestImagePath)

    X_V = X_V.reshape((imageCount, HEIGHT, WIDTH, depth))
    X_ML = X_ML.reshape((imageCount, HEIGHT, WIDTH, depth))
    X_HH = X_HH.reshape((imageCount, HEIGHT, WIDTH, depth))
    X_HS = X_HS.reshape((imageCount, HEIGHT, WIDTH, depth))
    
    generator.load_weights(weights_file)
    evaluate(generator,X_V,X_ML,X_HH,X_HS, Y,fileNames,positiveImagePath, negativeImagePath)
    

if __name__ == '__main__':
    # main(parse_arguments(sys.argv[1:]))
    dataset_name = 'VINmoire'  # VINmoire MRBI FHDMi MoireFace
    method_name = 'GAN_MPD'

    folder_path1 = '../data/' + dataset_name + '/test/positive'
    folder_path2 = '../data/' + dataset_name + '/test/negative'

    modelpath = './model/' + method_name+ '_' + dataset_name + '.h5'
    test_main(parse_arguments([modelpath,         #模型h5文件
                          folder_path1,           #测试摩尔纹图片目录
                          folder_path2]))                   #测试正常图片目录



# In[ ]:




