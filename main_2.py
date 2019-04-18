import numpy as np

np.random.seed(1337)  # for reproducibility
import keras
from keras import metrics
import tensorflow as tf
from keras import applications
import keras.backend.tensorflow_backend as KTF
from keras.datasets import mnist
from keras.models import Sequential
import keras.layers as L
from keras.layers import Dense, Dropout, Activation, Flatten,Reshape,BatchNormalization,Conv2DTranspose,Conv3DTranspose,merge
from keras.layers import Convolution2D, MaxPooling1D,Conv2D,Conv1D,Lambda,LSTM,LSTMCell,Conv3D,Convolution3D,ConvLSTM2D,MaxPool3D,ConvRecurrent2D,AveragePooling1D
import keras.layers.recurrent as R
from keras.optimizers import RMSprop, SGD,Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping,CSVLogger
import scipy.io as scio
import os,shutil,h5py
import time
from keras.utils import np_utils
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras.utils import plot_model
import matplotlib.pyplot as plt
from my_function import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# session = tf.Session(config=config)
# # 璁剧疆session
# KTF.set_session(session)

batch_size = 64
Fold = 5
Earlystop = 50
nb_classes = 80
epochs = 400000
# MAX_Q_SIZE = 160
# WORKERS = 8  #16
LEARNING_RATE = 0.0001
monitor_index = 'acc'
col = 600000
SubSample = 6000
# Pad_to = 299
fft_number = 150
NFFT = 51
noverlap = 12
# Generate_List = False
# fft_number = int(Pad_to/2+1) #150  #time_steps = int(pad_to/2+1),deppend on :pad_to=299
Pad_to = fft_number*2-1
step_input_size = int((col-NFFT)/(NFFT-noverlap))+1 #153


PATH = './keras_result/1/'
List_path = './data/freesound_audio_tagging_2019/'
# makedir(List_path)
# All_list_name = 'all_list.txt'
makedir(PATH)
shutil.copy('main_2.py',PATH)
shutil.copy('my_function.py',PATH)
train_list = 'train_list.csv'
test_list = 'test_list.csv'
# Data_Path = './data/radar_action/'
# All_list_path = PATH+All_list_name

# if Generate_List:
#     generate_radar_all_list(Data_Path=Data_Path, List_name=All_list_name, List_Save_Path=List_path)
#     generate_radar_train_test_list(Path=List_path, All_List_Path=List_path+All_list_name, Train_list=train_list, Test_list=test_list,
#                                    Fold=Fold, Reture_number=False)
#
# shutil.copy(List_path + 'all_list.txt', PATH)
# shutil.copy(List_path + 'shuff_all_data_list.txt', PATH)
# shutil.copy(List_path + 'data_distribution.txt', PATH)
# f = open(All_list_path, 'r')
# All_data_n = len(f.readlines())
# print('all data size :', All_data_n)
# f.close()

def write_summary(s_line):
    f = open(PATH+'summary.txt', 'a+')
    f.write(s_line+'\n')
    f.close()
    return

# for fnum in range(Fold):
#     Sub_path = PATH + 'fold' + str(fnum) + '/'
#     Sub_List_path = List_path + 'fold' + str(fnum) + '/'
#     makedir(Sub_path)
#     shutil.copy(Sub_List_path + train_list, Sub_path)
#     shutil.copy(Sub_List_path + test_list, Sub_path)
# makedir(PATH)
# All_list_name = 'all_list.txt'
# makedir(PATH)
# shutil.copy('keras_1Dconv_2_gt.py',PATH)
# shutil.copy('my_function.py',PATH)
# train_list  = 'train_list.txt'
# test_list = 'test_list.txt'
# Data_Path = './data/radar_action/'
# All_list_path = PATH+All_list_name
# if os.path.exists(All_list_path):
#     f = open(All_list_path, 'r')
#     All_data_n = len(f.readlines())
#     print('all data size :', All_data_n)
#     f.close()
#     pass
# else :
#     generate_radar_all_list(Data_Path=Data_Path,List_name=All_list_name,List_Save_Path=PATH)
#     f = open(All_list_path, 'r')
#     All_data_n = len(f.readlines())
#     print('all data size :', All_data_n)
#     f.close()
#
# if os.path.exists(PATH+'fold4'):
#     pass
# else :
#     generate_radar_train_test_list(Path=PATH,All_List_Path=All_list_path,Train_list = train_list,Test_list = test_list,Fold =Fold,Reture_number=False)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        # plt.show()
        plt.savefig(PATH + 'result.pdf')
        plt.close()

class train_config():
    input_shape = (SubSample,)

class fft_1Dconv(object):
    def __init__(self, config):
        self._input_shape = config.input_shape
        self._built_model()
    def _built_model(self):
        # 构建模型
        inputs = Input(shape=self._input_shape)
        x = Reshape((SubSample, 1))(inputs)

        x = self.conv1d_bn(x, 16, 51, strides=2, padding='same')
        x = self.conv1d_bn(x, 32, 7, strides=2, padding='same')
        x = MaxPooling1D(4, strides=4)(x)

        x = self.mydesen_block(x, 64, 'tanh')
        x = MaxPooling1D(4, 4)(x)

        x = self.mydesen_block(x, 64, 'tanh')
        x = MaxPooling1D(4, 4)(x)

        x = self.mydesen_block(x, 64, 'tanh')
        x = MaxPooling1D(4, 4)(x)

        # x = self.mydesen_block(x, 16, 'tanh')
        # x = MaxPooling1D(2, 2)(x)
        #
        # x = self.mydesen_block(x, 16, 'tanh')
        # x = MaxPooling1D(2, 2)(x)
        #
        # x = self.mydesen_block(x, 16, 'tanh')
        # x = MaxPooling1D(2, 2)(x)
        #
        # x = self.mydesen_block(x, 16, 'tanh')
        # x = MaxPooling1D(2, 2)(x)
        #
        # x = self.mydesen_block(x, 16, 'tanh')
        # x = MaxPooling1D(2, 2)(x)

        # x = self.mydesen_block(x, 16, 'tanh')
        # x = MaxPooling1D(2, 2)(x)

        x = Flatten()(x)
        # x = Dropout(0.6)(x)
        x = Dense(128, activation='relu', name='my_dense_0', trainable=True)(x)
        x = Dropout(0.5)(x)
        output = Dense(nb_classes, activation='sigmoid', name='my_dense_1', trainable=True)(x)
        model = Model(inputs=inputs, outputs=output)
        return model

    @staticmethod
    def antirectifier(x):
        # x = 10 * K.exp(tf.add(K.abs(x), np.spacing(1)))
        # if x>=0:
        #     x = 10*K.log(tf.add(x,1))
        # else :
        #     x = -10 * K.log(tf.add(-x, 1))
        # x = K.pow(x,1/3)
        # x = K.abs(x)/(tf.add(x,0.000001))
        x = 1*K.tanh(1*x)
        return x

    @staticmethod
    def antirectifier_output_shape(input_shape):
        shape = list(input_shape)
        if len(shape) == 2:  # only valid for 2D tensors
            shape[-1] *= 2
        return tuple(shape)

    @staticmethod
    def oneD_output_shape(input_shape):
        shape = list(input_shape[0])
        # if len(shape) == 2:  # only valid for 2D tensors
        #     shape[-1] *= 2
        return tuple(shape)

    @staticmethod
    def my_init_real(shape, name=None):
        w_matrix = np.zeros(shape)
        for k in range(fft_number):
            for i in range(NFFT):
                w_matrix[0, i, 0, k] = cos(2 * k * pi * (i / Pad_to))
        return K.variable(w_matrix, name=name)

    @staticmethod
    def my_init_imag(shape, name=None):
        w_matrix = np.zeros(shape)
        for k in range(fft_number):
            for i in range(NFFT):
                w_matrix[0, i, 0, k] = -sin(2 * k * pi * (i / Pad_to))
        return K.variable(w_matrix, name=name)

    @staticmethod
    def res(x):
        x = tf.add(x[0], x[1] * 0.17)
        # x = 1*K.tanh(1*x)
        return x

    @staticmethod
    def divide_NFFT(x):
        x = x / NFFT
        return x

    @staticmethod
    def power_f(x):
        x = K.pow(x,0.5)
        return x

    @staticmethod
    def log_lbd(x):
        x = 10 * K.log(tf.add(K.abs(x), np.spacing(1))) / K.log(10.0)
        # if x>=0:
        #     x = 10*K.log(tf.add(x,1))
        # else :
        #     x = -10 * K.log(tf.add(-x, 1))
        # x = K.pow(x,1/3)
        # x = K.abs(x)/(tf.add(x,0.000001))
        # x = 0.4 * K.tanh(1 * x)
        return x

    @staticmethod
    def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), name=None):
        """Utility function to apply conv + BN.
        # Arguments
            x: input tensor.
            filters: filters in `Conv2D`.
            num_row: height of the convolution kernel.
            num_col: width of the convolution kernel.
            padding: padding mode in `Conv2D`.
            strides: strides in `Conv2D`.
            name: name of the ops; will become `name + '_conv'`
                for the convolution and `name + '_bn'` for the
                batch norm layer.

        # Returns
            Output tensor after applying `Conv2D` and `BatchNormalization`.
        """
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        if K.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 3
        x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False, name=conv_name)(x)
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        x = Activation('relu', name=name)(x)
        return x

    @staticmethod
    def conv1d_bn(x, filters, k_size, padding='same', strides=(1, 1), activation='tanh', name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        if K.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 2
        x = Conv1D(
            filters, k_size,
            strides=strides,
            padding=padding,
            use_bias=False,
            name=conv_name)(x)
        # x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        x = Activation(activation, name=name)(x)
        return x

    def mydesen_block(self, x, out_channels=256, activation='tanh'):
        x_base = self.conv1d_bn(x, int(out_channels * 1 / 4), 1, strides=1, padding='same', activation=activation)
        x_2 = self.conv1d_bn(x_base, int(out_channels / 4), 5, strides=1, padding='same', activation=activation)
        # x_2 = self.conv1d_bn(x_2, int(out_channels / 4), 3, strides=1, padding='same', activation=activation)

        # x_3 = self.conv1d_bn(x, int(out_channels * 3 / 4), 1, strides=1, padding='same', activation=activation)
        x_3 = self.conv1d_bn(x_base, int(out_channels / 4), 3, strides=1, padding='same', activation=activation)

        # x_4 = self.conv1d_bn(x, int(out_channels / 4), 1, strides=1, padding='same', activation=activation)

        x_5 = AveragePooling1D(3, strides=1, padding='same')(x)
        x_5 = self.conv1d_bn(x_5, int(out_channels / 4), 1, strides=1, activation=activation)

        x = L.concatenate([x, x_2, x_3, x_base, x_5], axis=-1, name=None)
        return x

    # def mydesen_block(self, x, out_channels=256, activation='tanh'):
    #     x_2 = self.conv1d_bn(x, int(out_channels * 3 / 16), 1, strides=1, padding='same', activation=activation)
    #     x_2 = self.conv1d_bn(x_2, int(out_channels / 4), 5, strides=1, padding='same', activation=activation)
    #     # x_2 = self.conv1d_bn(x_2, int(out_channels / 4), 3, strides=1, padding='same', activation=activation)
    #
    #     x_3 = self.conv1d_bn(x, int(out_channels * 3 / 4), 1, strides=1, padding='same', activation=activation)
    #     x_3 = self.conv1d_bn(x_3, int(out_channels / 4), 3, strides=1, padding='same', activation=activation)
    #
    #     x_4 = self.conv1d_bn(x, int(out_channels / 4), 1, strides=1, padding='same', activation=activation)
    #
    #     x_5 = AveragePooling1D(3, strides=1, padding='same')(x)
    #     x_5 = self.conv1d_bn(x_5, int(out_channels / 4), 1, strides=1, activation=activation)
    #
    #     x = L.concatenate([x, x_2, x_3, x_4, x_5], axis=-1, name=None)
    #     return x

history = LossHistory()

# for fnum in range(1):
train_con = train_config()
train_model = fft_1Dconv(train_con)
model = train_model._built_model()
keras.utils.plot_model(model, to_file=PATH + 'model.pdf', show_shapes=True, show_layer_names=True, rankdir='TB')
model.summary()
# if fnum == 0:
model.summary(print_fn=write_summary)
# 缂栬瘧妯″瀷
optimizer = Adam(lr=LEARNING_RATE)
# optimizer = SGD(lr = LEARNING_RATE, momentum = 0.9, decay = 0.0, nesterov = True)
# model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])

# Sub_path = PATH + 'fold' + str(fnum) + '/'
# makedir(Sub_path)


best_model_file = PATH + 'crnn.h5'
best_model = ModelCheckpoint(best_model_file, monitor='val_'+monitor_index, verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_'+monitor_index, factor=0.5,
                  patience=20, verbose=1, min_lr=0.00001)
early_stop = EarlyStopping(monitor='val_'+monitor_index, patience=Earlystop, verbose=1)
result_save = CSVLogger(PATH + 'result.txt',append=True)
# 璁㘹粌妯″瀷
print('begin training...')
time_start = time.time()
TRAIN_X, TRAIN_Y = generate_audio_bacth_faster(List_path + train_list, colum=col, nb_classes=nb_classes,
                NFFT=NFFT, noverlap=noverlap, frequce=44100, sub_sample=SubSample, pad_to=Pad_to, shuffle=False, stft_form=False, stft_handle=False, only_real=False)
TEST_X, TEST_Y = generate_audio_bacth_faster(List_path + test_list, colum = col, nb_classes=nb_classes,
                NFFT=NFFT, noverlap=noverlap, frequce=44100, sub_sample=SubSample, pad_to=Pad_to, shuffle=False, stft_form=False, stft_handle=False, only_real=False)
print('data generate finished!')
print(TRAIN_X[0])
print(TRAIN_Y[0])
hist = model.fit(TRAIN_X, TRAIN_Y, batch_size=batch_size, epochs=epochs,
                 verbose=1, validation_data=(TEST_X, TEST_Y),
                 callbacks=[best_model, reduce_lr, early_stop, result_save, history])

history.loss_plot('epoch')

time_end = time.time()
used_time = int(time_end-time_start)
print('used time:%d s'%(used_time))
with open(PATH + 'used_time.txt', 'a+') as t1:
    t1.write('Used time:%d s'%(used_time) + '\n')


# summary_result(PATH=PATH, Fold=Fold, Earlystop=Earlystop)

raw_ssf = open(List_path+'sample_submission.csv','r')
raw_lines = raw_ssf.readlines()

ssf = open(PATH+'sample_submission1.csv','w')
ssf.write(raw_lines[0])
raw_lines = raw_lines[1:]

for i in range(len(raw_lines)):
    test_p = List_path + 'test/' + raw_lines[i].strip().split(',')[0]
    test_data = test_audio(test_p, colum=col, NFFT=NFFT, noverlap=noverlap, frequce=44100, sub_sample=SubSample,
                           pad_to=Pad_to, stft_form=False, stft_handle=False, only_real=False)
    if i % batch_size == 0:
        test_batch_d = test_data
    else:
        test_batch_d = np.vstack((test_batch_d, test_data))

    if (i-batch_size+1)%batch_size == 0 or i == len(raw_lines)-1:
        y_bacth = model.predict(test_batch_d, batch_size=batch_size)
        y_bacth = np.where(y_bacth>=0.5, 1, 0)
        if i == (batch_size - 1):
            y_all = y_bacth
        else:
            y_all = np.vstack((y_all, y_bacth))

for i in range(len(raw_lines)):
    to_write = raw_lines[i].strip().split(',')[0]
    for j in range(nb_classes):
        to_write = ','.join([to_write, str(y_all[i][j])])
    ssf.write(to_write+'\n')

ssf.close()
raw_ssf.close()
