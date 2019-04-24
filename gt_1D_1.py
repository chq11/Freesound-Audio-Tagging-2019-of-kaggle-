import numpy as np
import sklearn.metrics
# from sklearn.preprocessing import StandardScaler
np.random.seed(1337)  # for reproducibility
import keras,wave,signal,librosa
from keras.losses import sparse_categorical_crossentropy
from keras import metrics
import tensorflow as tf
from numpy import *
from keras import applications
import keras.backend.tensorflow_backend as KTF
from keras.datasets import mnist
from keras.models import Sequential
import keras.layers as L
from keras.layers import Dense, Dropout, Activation, Flatten,Reshape,BatchNormalization,Conv2DTranspose,Conv3DTranspose,merge,GlobalAveragePooling1D,GlobalMaxPool1D,MaxPooling2D
from keras.layers import Convolution1D, MaxPool1D,Conv2D,Conv1D,Lambda,LSTM,LSTMCell,Conv3D,Convolution3D,ConvLSTM2D,MaxPool3D,ConvRecurrent2D,AveragePooling1D,MaxPooling1D
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
# from my_function import *
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
epochs = 999999
# MAX_Q_SIZE = 160
# WORKERS = 8  #16
LEARNING_RATE = 0.001
monitor_index = 'loss'
col = 600000
SubSample = 30000
MAX_Q_SIZE = 16
WORKERS = 8

# Pad_to = 299
fft_number = 150
nfft = 20 #ms
nvlt = 10 #ms

NFFT = int(round(44.1*nfft))
noverlap = int(round(44.1*nvlt))
# Generate_List = False
# fft_number = int(Pad_to/2+1) #150  #time_steps = int(pad_to/2+1),deppend on :pad_to=299
Pad_to = fft_number*2-1
step_input_size = int((SubSample-NFFT)/(NFFT-noverlap))+1 #153


PATH = './keras_result/11/'
List_path = './data/freesound_audio_tagging_2019/'
DataPath = './data/freesound_audio_tagging_2019/'
# makedir(List_path)
# All_list_name = 'all_list.txt'
def makedir(PATH):
    path = PATH.strip()
    path = path.rstrip("/")  # 去除尾部 / 符号
    isExists = os.path.exists(path)  # 判断结果
    if not isExists:  # 如果不存在则创建目录 # 创建目录操作函数
        os.makedirs(path)
        print(path + ' was created successfully')
    else:  # 如果目录存在则不创建，并提示目录已存在
        print(path + ' was already exists')
    return

makedir(PATH)
shutil.copy('gt_1D_1.py',PATH)
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

def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data - min_data) / (max_data - min_data + 0.0001)
    return data - 0.5

def load_audio_file(file_path, input_length=SubSample):
    data = librosa.core.load(file_path, sr=44100)[0]  # , sr=16000
    if len(data) > input_length:
        max_offset = len(data) - input_length
        offset = np.random.randint(max_offset)
        data = data[offset:(input_length + offset)]

    else:
        if input_length > len(data):
            max_offset = input_length - len(data)
            offset = np.random.randint(max_offset)
        else:
            offset = 0

        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

    data = audio_norm(data)
    return data

def handle_data2fft_form(Train_x, Test_x=None,Train_y = None, Test_y = None,Fold=5,Test_size_split=4,step_input_size = 153,NFFT = 51,noverlap = 12):
    if Test_x is not None:
        A_X = vstack((Train_x, Test_x))
        A_Y = vstack((Train_y, Test_y))
    else :
        A_X = Train_x
        A_Y = Train_y
    All_size = A_X.shape[0]
    new_all_size = int(All_size/(Fold*Test_size_split))*(Fold*Test_size_split)
    new_A_X = A_X[:new_all_size]
    new_A_Y = A_Y[:new_all_size]
    for i in range(step_input_size):  # 153
        # for j in range(self._bunch_size):
        in_put_1 = np.reshape(new_A_X[:, i * (NFFT - noverlap):NFFT + i * (NFFT - noverlap)],
                              [new_all_size, NFFT, -1])
        try:
            in_put_2 = np.concatenate([in_put_2, in_put_1], axis=2)
        except:
            in_put_2 = in_put_1
    in_put_2 = np.transpose(in_put_2, [0, 2, 1])
    new_A_X = in_put_2
    if Test_x is not None:
        print('data change to fft form succeed')
    return new_A_X,new_A_Y


def generate_audio_bacth(Datapath, ListPath,colum = 600000,nb_classes = 80,NFFT = 51,noverlap = 12,frequce = 44100,
                         sub_sample = 6000,pad_to = 299,shuffle=True,stft_form = True,stft_handle = False,conv3D = False,only_real = False):
    '''
    according to the data list(txt file) generate data batch
    :param Path: data list path
    :param batch_size:
    :param colum:
    :param nb_classes:
    :param NFFT:
    :param noverlap:
    :param frequce:
    :param pad_to:
    :param shuffle:
    :param stft_form: if true ,change data to stft format
    :param stft_handle: if ture ,data will use short time fourier transform
    :return:
    '''
    sub_freq = int(frequce/int(colum / sub_sample))
    # colum = int(colum/(int(frequce/sub_freq)))
    step_input_size = int((sub_sample - NFFT) / (NFFT - noverlap)) + 1 #153
    f = open(ListPath, 'r')
    lines = f.readlines()
    f.close()
    N = len(lines)
    # N = 1000
    if shuffle:
        random.shuffle(lines)

    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0

        X_batch = np.zeros((current_batch_size, sub_sample))
        Y_batch = np.zeros((current_batch_size, nb_classes))

        index_j = 0
        for l in range(current_index, current_index + current_batch_size):
            line = lines[l]
            d_label = line.strip().split(',')[1:]
            # D_label = np.zeros([1, 80])
            for i in range(nb_classes):
                Y_batch[index_j, i] = np.int(d_label[i])


            # print(ALL_Y)
            wav_p = line.strip().split(',')[0]
            new_wave_d = load_audio_file(Datapath+wav_p, sub_sample)

            # wave_data = librosa.core.load(Datapath+wav_p, sr=frequce)[0] #, sr=16000
            # wave_shape = wave_data.shape
            # new_wave_d = np.zeros([1, colum])
            # if wave_shape[0] >= colum:
            #     new_wave_d[0, :] = wave_data[int((wave_shape[0] - colum) / 2):int((wave_shape[0] - colum) / 2) + colum]
            # else:
            #     new_wave_d[0,
            #     int((colum - wave_shape[0]) / 2):int((colum - wave_shape[0]) / 2) + wave_shape[0]] = wave_data
            #
            # new_wave_d = new_wave_d[0, 0::int(colum / sub_sample)]
            # new_wave_d = (new_wave_d - new_wave_d.mean()) / (new_wave_d.std() + 0.0001)
            X_batch[index_j] = new_wave_d
            index_j += 1
        if stft_form:
            X_batch, Y_batch = handle_data2fft_form(Train_x=X_batch, Test_x=None, Train_y=Y_batch, Test_y=None,
                                                Fold=1, Test_size_split=1, step_input_size=step_input_size, NFFT=NFFT,
                                                noverlap=noverlap)

        if stft_handle:
            New_X_batch = np.zeros((current_batch_size, int(pad_to / 2 + 1), step_input_size))
            for i in range(N):
                freqs_1, t_1, spec_1 = signal.spectrogram(X_batch[i], fs=sub_freq, window=('hamming'), nperseg=NFFT,
                                                          noverlap=noverlap,
                                                          nfft=pad_to, detrend='constant', return_onesided=True,
                                                          scaling='density', axis=-1, mode='complex')
                if only_real:
                    spec_1 = 10 * np.log10(abs(spec_1.real) + np.spacing(1))
                else:
                    spec_1 = 10 * np.log10(abs(spec_1) + np.spacing(1))
                New_X_batch[i] = spec_1
                X_batch = New_X_batch

        if conv3D:
            X_batch = np.expand_dims(X_batch, axis=2)
            X_batch = np.expand_dims(X_batch, axis=4)

        yield (X_batch, Y_batch)

def generate_audio_bacth_faster(Datapath, ListPath,colum = 600000,nb_classes = 80,NFFT = 51,noverlap = 12,frequce = 44100,
                         sub_sample = 6000,pad_to = 299,shuffle=True,stft_form = True,stft_handle = False,conv3D = False,only_real = False, datatype = 'int8'):
    '''
    according to the data list(txt file) generate data batch
    :param Path: data list path
    :param batch_size:
    :param colum:
    :param nb_classes:
    :param NFFT:
    :param noverlap:
    :param frequce:
    :param pad_to:
    :param shuffle:
    :param stft_form: if true ,change data to stft format
    :param stft_handle: if ture ,data will use short time fourier transform
    :return:
    '''
    sub_freq = int(frequce/int(colum / sub_sample))
    # colum = int(colum/(int(frequce/sub_freq)))
    step_input_size = int((sub_sample - NFFT) / (NFFT - noverlap)) + 1 #153
    f = open(ListPath, 'r')
    lines = f.readlines()
    N = len(lines)
    # N = 1000
    if shuffle:
        random.shuffle(lines)
    ALL_X = np.zeros((N, sub_sample))
    ALL_Y = np.zeros((N, nb_classes))
    for l in range(N):
        line = lines[l]
        d_label = line.strip().split(',')[1:]
        # D_label = np.zeros([1, 80])
        for i in range(nb_classes):
            if datatype == 'int8':
                ALL_Y[l, i] = np.int(d_label[i])
            else:
                ALL_Y[l, i] = np.float(d_label[i])

        # print(ALL_Y)
        wav_p = line.strip().split(',')[0]
        new_wave_d = load_audio_file(Datapath+wav_p, sub_sample)

        # wave_data = librosa.core.load(Datapath+wav_p, sr=frequce)[0] #, sr=16000
        # wave_shape = wave_data.shape
        # new_wave_d = np.zeros([1, colum])
        # if wave_shape[0] >= colum:
        #     new_wave_d[0, :] = wave_data[int((wave_shape[0] - colum) / 2):int((wave_shape[0] - colum) / 2) + colum]
        # else:
        #     new_wave_d[0,
        #     int((colum - wave_shape[0]) / 2):int((colum - wave_shape[0]) / 2) + wave_shape[0]] = wave_data
        #
        # new_wave_d = new_wave_d[0, 0::int(colum / sub_sample)]
        # new_wave_d = (new_wave_d - new_wave_d.mean()) / (new_wave_d.std() + 0.0001)
        ALL_X[l] = new_wave_d
    if stft_form:
        ALL_X, ALL_Y = handle_data2fft_form(Train_x=ALL_X, Test_x=None, Train_y=ALL_Y, Test_y=None,
                                            Fold=1, Test_size_split=1, step_input_size=step_input_size, NFFT=NFFT,
                                            noverlap=noverlap)

    if stft_handle:
        New_X_batch = np.zeros((N, int(pad_to / 2 + 1), step_input_size))
        for i in range(N):
            freqs_1, t_1, spec_1 = signal.spectrogram(ALL_X[i], fs=sub_freq, window=('hamming'), nperseg=NFFT,
                                                      noverlap=noverlap,
                                                      nfft=pad_to, detrend='constant', return_onesided=True,
                                                      scaling='density', axis=-1, mode='complex')
            if only_real:
                spec_1 = 10 * np.log10(abs(spec_1.real) + np.spacing(1))
            else:
                spec_1 = 10 * np.log10(abs(spec_1) + np.spacing(1))
            New_X_batch[i] = spec_1
            ALL_X = New_X_batch

    if conv3D:
        ALL_X = np.expand_dims(ALL_X, axis=2)
        ALL_X = np.expand_dims(ALL_X, axis=4)

    return (ALL_X, ALL_Y)

def test_audio(lines,colum = 600000,NFFT = 51,noverlap = 12,frequce = 44100,
                         sub_sample = 6000,pad_to = 299,stft_form = True,stft_handle = False,conv3D = False,only_real = False):
    '''
    according to the data list(txt file) generate data batch
    :param Path: data list path
    :param batch_size:
    :param colum:
    :param nb_classes:
    :param NFFT:
    :param noverlap:
    :param frequce:
    :param pad_to:
    :param shuffle:
    :param stft_form: if true ,change data to stft format
    :param stft_handle: if ture ,data will use short time fourier transform
    :return:
    '''
    sub_freq = int(frequce/int(colum / sub_sample))
    # colum = int(colum/(int(frequce/sub_freq)))
    step_input_size = int((sub_sample - NFFT) / (NFFT - noverlap)) + 1 #153
    ALL_X = np.zeros((1, sub_sample))

    line = lines

    # print(ALL_Y)
    wav_p = line.strip().split(',')[0]
    new_wave_d = load_audio_file(wav_p, sub_sample)
    # wave_data = librosa.core.load(wav_p, sr=44100)[0]  # , sr=16000
    # wave_shape = wave_data.shape
    # new_wave_d = np.zeros([1, colum])
    # if wave_shape[0] >= colum:
    #     new_wave_d[0, :] = wave_data[int((wave_shape[0] - colum) / 2):int((wave_shape[0] - colum) / 2) + colum]
    # else:
    #     new_wave_d[0,
    #     int((colum - wave_shape[0]) / 2):int((colum - wave_shape[0]) / 2) + wave_shape[0]] = wave_data
    #
    # new_wave_d = new_wave_d[0, 0::int(colum / sub_sample)]
    # new_wave_d = (new_wave_d - new_wave_d.mean()) / (new_wave_d.std() + 0.0001)
    ALL_X[0] = new_wave_d
    if stft_form:
        ALL_X, _ = handle_data2fft_form(Train_x=ALL_X, Test_x=None, Train_y=None, Test_y=None,
                                            Fold=1, Test_size_split=1, step_input_size=step_input_size, NFFT=NFFT,
                                            noverlap=noverlap)

    if stft_handle:
        New_X_batch = np.zeros((1, int(pad_to / 2 + 1), step_input_size))
        for i in range(1):
            freqs_1, t_1, spec_1 = signal.spectrogram(ALL_X[i], fs=sub_freq, window=('hamming'), nperseg=NFFT,
                                                      noverlap=noverlap,
                                                      nfft=pad_to, detrend='constant', return_onesided=True,
                                                      scaling='density', axis=-1, mode='complex')
            if only_real:
                spec_1 = 10 * np.log10(abs(spec_1.real) + np.spacing(1))
            else:
                spec_1 = 10 * np.log10(abs(spec_1) + np.spacing(1))
            New_X_batch[i] = spec_1
            ALL_X = New_X_batch

    if conv3D:
        ALL_X = np.expand_dims(ALL_X, axis=2)
        ALL_X = np.expand_dims(ALL_X, axis=4)

    return ALL_X

def write_summary(s_line):
    f = open(PATH+'summary.txt', 'a+')
    f.write(s_line+'\n')
    f.close()
    return

def my_accuracy(y_true, y_pred):
    return K.cast(K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)+0.012499, dtype= 'int8')
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

def _one_sample_positive_class_precisions(scores, truth):
    """Calculate precisions for each true class for a single sample.

    Args:
      scores: np.array of (num_classes,) giving the individual classifier scores.
      truth: np.array of (num_classes,) bools indicating which classes are true.

    Returns:
      pos_class_indices: np.array of indices of the true classes for this sample.
      pos_class_precisions: np.array of precisions corresponding to each of those
        classes.
    """
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    # Only calculate precisions if there are some true classes.
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    # Retrieval list of classes for this sample.
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)
    # Which of these is a true label?
    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    # Precision of retrieval list truncated at each hit, in order of pos_labels.
    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits


# All-in-one calculation of per-class lwlrap.
def calculate_per_class_lwlrap(truth, scores):
    """Calculate label-weighted label-ranking average precision.

    Arguments:
      truth: np.array of (num_samples, num_classes) giving boolean ground-truth
        of presence of that class in that sample.
      scores: np.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.

    Returns:
      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
        class.
      weight_per_class: np.array of (num_classes,) giving the prior of each
        class within the truth labels.  Then the overall unbalanced lwlrap is
        simply np.sum(per_class_lwlrap * weight_per_class)
    """
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    # Space to store a distinct precision value for each class on each sample.
    # Only the classes that are true for each sample will be filled in.
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = (
            _one_sample_positive_class_precisions(scores[sample_num, :],
                                                  truth[sample_num, :]))
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (
            precision_at_hits)
    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    # Form average of each column, i.e. all the precisions assigned to labels in
    # a particular class.
    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    # overall_lwlrap = simple average of all the actual per-class, per-sample precisions
    #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)
    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
    #                = np.sum(per_class_lwlrap * weight_per_class)
    return per_class_lwlrap, weight_per_class


# Calculate the overall lwlrap using sklearn.metrics function.
def calculate_overall_lwlrap_sklearn(truth, scores):
    """Calculate the overall lwlrap using sklearn.metrics.lrap."""
    # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
    sample_weight = np.sum(truth > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    overall_lwlrap = sklearn.metrics.label_ranking_average_precision_score(
        truth[nonzero_weight_sample_indices, :] > 0,
        scores[nonzero_weight_sample_indices, :],
        )
    return overall_lwlrap

# Accumulator object version.
class lwlrap_accumulator(object):
    """Accumulate batches of test samples into per-class and overall lwlrap."""

    def __init__(self):
        self.num_classes = 0
        self.total_num_samples = 0

    def accumulate_samples(self, batch_truth, batch_scores):
        """Cumulate a new batch of samples into the metric.

        Args:
          truth: np.array of (num_samples, num_classes) giving boolean
            ground-truth of presence of that class in that sample for this batch.
          scores: np.array of (num_samples, num_classes) giving the
            classifier-under-test's real-valued score for each class for each
            sample.
        """
        assert batch_scores.shape == batch_truth.shape
        num_samples, num_classes = batch_truth.shape
        if not self.num_classes:
            self.num_classes = num_classes
            self._per_class_cumulative_precision = np.zeros(self.num_classes)
            self._per_class_cumulative_count = np.zeros(self.num_classes,
                                                        dtype=np.int)
        assert num_classes == self.num_classes
        for truth, scores in zip(batch_truth, batch_scores):
            pos_class_indices, precision_at_hits = (
                _one_sample_positive_class_precisions(scores, truth))
            self._per_class_cumulative_precision[pos_class_indices] += (
                precision_at_hits)
            self._per_class_cumulative_count[pos_class_indices] += 1
        self.total_num_samples += num_samples

    def per_class_lwlrap(self):
        """Return a vector of the per-class lwlraps for the accumulated samples."""
        return (self._per_class_cumulative_precision /
                np.maximum(1, self._per_class_cumulative_count))

    def per_class_weight(self):
        """Return a normalized weight vector for the contributions of each class."""
        return (self._per_class_cumulative_count /
                float(np.sum(self._per_class_cumulative_count)))

    def overall_lwlrap(self):
        """Return the scalar overall lwlrap for cumulated samples."""
        return np.sum(self.per_class_lwlrap() * self.per_class_weight())

class train_config():
    input_shape = (SubSample,)

class fft_1Dconv(object):
    def __init__(self, config):
        self._input_shape = config.input_shape
        self._built_model()
    def _built_model(self):
        # inp = Input(shape=self._input_shape)
        # img_1 = Reshape((SubSample, 1))(inp)
        # # inp = Input(shape=(input_length, 1))
        # img_1 = Convolution1D(16, kernel_size=9, activation="relu", padding="valid")(img_1)
        # img_1 = Convolution1D(16, kernel_size=9, activation="relu", padding="valid")(img_1)
        # img_1 = MaxPool1D(pool_size=16)(img_1)
        # img_1 = Convolution1D(64, kernel_size=3, activation="relu", padding="valid")(img_1)
        # img_1 = Convolution1D(64, kernel_size=3, activation="relu", padding="valid")(img_1)
        # img_1 = MaxPool1D(pool_size=4)(img_1)
        # img_1 = Convolution1D(64, kernel_size=3, activation="relu", padding="valid")(img_1)
        # img_1 = Convolution1D(64, kernel_size=3, activation="relu", padding="valid")(img_1)
        # img_1 = MaxPool1D(pool_size=4)(img_1)
        # img_1 = Convolution1D(256, kernel_size=3, activation="relu", padding="valid")(img_1)
        # img_1 = Convolution1D(256, kernel_size=3, activation="relu", padding="valid")(img_1)
        # img_1 = GlobalMaxPool1D()(img_1)
        #
        # dense_1 = Dense(512, activation="relu")(img_1)
        # dense_1 = Dense(256, activation="relu")(dense_1)
        # dense_1 = Dense(nb_classes, activation="softmax")(dense_1)
        #
        # model = Model(inputs=inp, outputs=dense_1)
        #
        # return model
        # 构建模型
        inputs = Input(shape=self._input_shape)
        x = Reshape((SubSample, 1))(inputs)

        x = self.conv1d_bn(x, 32, 256, strides=2, padding='same')
        x = self.conv1d_bn(x, 64, 128, strides=2, padding='same')
        x = MaxPooling1D(2, strides=2)(x)
        # x = BatchNormalization()(x)

        for i in range(6):
            x = self.mydesen_blockA(x, 128, 'tanh')
            x = MaxPooling1D(2, 2)(x)
            # x = BatchNormalization()(x)
        for i in range(3):
            x = self.mydesen_blockD(x, 128, 'tanh')
            x = MaxPooling1D(2, 2)(x)

        # x = self.mydesen_blockA(x, 64, 'tanh')
        # x = MaxPooling1D(2, 2)(x)
        # x = BatchNormalization()(x)
        #
        # x = self.mydesen_blockA(x, 64, 'tanh')
        # x = MaxPooling1D(2, 2)(x)
        # x = BatchNormalization()(x)
        #
        # x = self.mydesen_blockB(x, 64, 'tanh')
        # x = MaxPooling1D(2, 2)(x)
        # x = BatchNormalization()(x)
        #
        # x = self.mydesen_blockB(x, 64, 'tanh')
        # x = MaxPooling1D(2, 2)(x)
        # x = BatchNormalization()(x)
        #
        # x = self.mydesen_blockB(x, 64, 'tanh')
        # x = MaxPooling1D(2, 2)(x)
        # x = BatchNormalization()(x)
        #
        # x = self.mydesen_blockC(x, 64, 'tanh')
        # x = MaxPooling1D(2, 2)(x)
        # x = BatchNormalization()(x)
        #
        # x = self.mydesen_blockC(x, 64, 'tanh')
        # x = MaxPooling1D(2, 2)(x)
        # x = BatchNormalization()(x)

        # x = self.mydesen_blockC(x, 64, 'tanh')
        # x = MaxPooling1D(4, 4)(x)
        # x = GlobalAveragePooling1D(name='avg_pool')(x)
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
        x = Dropout(0.6)(x)
        x = Dense(1024, activation='relu', name='my_dense_0', trainable=True)(x)
        # x = Dropout(0.5)(x)
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

    def mydesen_blockA(self, x, out_channels=256, activation='tanh'):
        x_2 = self.conv1d_bn(x, int(out_channels * 1 / 8), 1, strides=1, padding='same', activation=activation)
        x_2 = self.conv1d_bn(x_2, int(out_channels / 4), 5, strides=1, padding='same', activation=activation)
        # x_2 = self.conv1d_bn(x_2, int(out_channels / 4), 3, strides=1, padding='same', activation=activation)

        x_3 = self.conv1d_bn(x, int(out_channels * 1 / 8), 1, strides=1, padding='same', activation=activation)
        x_3 = self.conv1d_bn(x_3, int(out_channels / 4), 3, strides=1, padding='same', activation=activation)

        x_4 = self.conv1d_bn(x, int(out_channels / 4), 1, strides=1, padding='same', activation=activation)

        x_5 = AveragePooling1D(3, strides=1, padding='same')(x)
        x_5 = self.conv1d_bn(x_5, int(out_channels / 4), 1, strides=1, activation=activation)

        x = L.concatenate([x, x_2, x_3, x_4, x_5], axis=-1, name=None)
        return x

    def mydesen_blockB(self, x, out_channels=256, activation='tanh'):
        x_2 = self.conv1d_bn(x, int(out_channels * 1 / 8), 1, strides=1, padding='same', activation=activation)
        x_2 = self.conv1d_bn(x_2, int(out_channels * 6 / 16), 5, strides=1, padding='same', activation=activation)
        # x_2 = self.conv1d_bn(x_2, int(out_channels / 4), 3, strides=1, padding='same', activation=activation)

        x_3 = self.conv1d_bn(x, int(out_channels * 1 / 8), 1, strides=1, padding='same', activation=activation)
        x_3 = self.conv1d_bn(x_3, int(out_channels * 2 / 16), 3, strides=1, padding='same', activation=activation)

        x_4 = self.conv1d_bn(x, int(out_channels / 4), 1, strides=1, padding='same', activation=activation)

        x_5 = AveragePooling1D(3, strides=1, padding='same')(x)
        x_5 = self.conv1d_bn(x_5, int(out_channels / 4), 1, strides=1, activation=activation)

        x = L.concatenate([x, x_2, x_3, x_4, x_5], axis=-1, name=None)
        return x

    def mydesen_blockC(self, x, out_channels=256, activation='tanh'):
        x_2 = self.conv1d_bn(x, int(out_channels * 1 / 8), 1, strides=1, padding='same', activation=activation)
        x_2 = self.conv1d_bn(x_2, int(out_channels * 7 / 16), 5, strides=1, padding='same', activation=activation)
        # x_2 = self.conv1d_bn(x_2, int(out_channels / 4), 3, strides=1, padding='same', activation=activation)

        x_3 = self.conv1d_bn(x, int(out_channels * 1 / 8), 1, strides=1, padding='same', activation=activation)
        x_3 = self.conv1d_bn(x_3, int(out_channels * 1 / 16), 3, strides=1, padding='same', activation=activation)

        x_4 = self.conv1d_bn(x, int(out_channels / 4), 1, strides=1, padding='same', activation=activation)

        x_5 = AveragePooling1D(3, strides=1, padding='same')(x)
        x_5 = self.conv1d_bn(x_5, int(out_channels / 4), 1, strides=1, activation=activation)

        x = L.concatenate([x, x_2, x_3, x_4, x_5], axis=-1, name=None)
        return x

    def mydesen_blockD(self, x, out_channels=256, activation='tanh'):
        x_2 = self.conv1d_bn(x, int(out_channels * 1 / 8), 1, strides=1, padding='same', activation=activation)
        x_2 = self.conv1d_bn(x_2, int(out_channels * 7 / 16), 11, strides=1, padding='same', activation=activation)
        # x_2 = self.conv1d_bn(x_2, int(out_channels / 4), 3, strides=1, padding='same', activation=activation)

        x_3 = self.conv1d_bn(x, int(out_channels * 1 / 8), 1, strides=1, padding='same', activation=activation)
        x_3 = self.conv1d_bn(x_3, int(out_channels * 1 / 16), 9, strides=1, padding='same', activation=activation)

        x_4 = self.conv1d_bn(x, int(out_channels / 4), 7, strides=1, padding='same', activation=activation)

        x_5 = AveragePooling1D(3, strides=1, padding='same')(x)
        x_5 = self.conv1d_bn(x_5, int(out_channels / 4), 5, strides=1, padding='same', activation=activation)

        x = L.concatenate([x, x_2, x_3, x_4, x_5], axis=-1, name=None)
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
# model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy',my_accuracy])
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc',my_accuracy])

# Sub_path = PATH + 'fold' + str(fnum) + '/'
# makedir(Sub_path)


best_model_file = PATH + 'crnn.h5'
best_model = ModelCheckpoint(best_model_file, monitor='val_'+monitor_index, verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_'+monitor_index, factor=0.5,
                  patience=20, verbose=1, min_lr=0.00001)
early_stop = EarlyStopping(monitor='val_'+monitor_index, patience=Earlystop, verbose=1)
result_save = CSVLogger(PATH + 'result.txt',append=True)

train_data_lines = open(List_path + train_list).readlines()
# Check if image path exists.
nbr_train = len(train_data_lines)
print('# Train Number: {}.'.format(nbr_train))
# global steps_per_epoch
steps_per_epoch = int(ceil(nbr_train * 1. / batch_size))

val_data_lines = open(List_path + test_list).readlines()
nbr_val = len(val_data_lines)
print('# Val Number: {}.'.format(nbr_val))
validation_steps = int(ceil(nbr_val * 1. / batch_size))

# 璁㘹粌妯″瀷
print('begin training...')
time_start = time.time()
TRAIN = generate_audio_bacth(DataPath, List_path + train_list, colum=col, nb_classes=nb_classes,
                NFFT=NFFT, noverlap=noverlap, frequce=44100, sub_sample=SubSample, pad_to=Pad_to, shuffle=False, stft_form=False, stft_handle=False, only_real=False)
TEST = generate_audio_bacth(DataPath, List_path + test_list, colum = col, nb_classes=nb_classes,
                NFFT=NFFT, noverlap=noverlap, frequce=44100, sub_sample=SubSample, pad_to=Pad_to, shuffle=False, stft_form=False, stft_handle=False, only_real=False)
print('data generate finished!')
# print(TRAIN_X[0],TRAIN_X[0].shape)
# print(TRAIN_Y[0],TRAIN_Y[0].shape)
hist = model.fit_generator(TRAIN, steps_per_epoch=steps_per_epoch, epochs=epochs,
                 verbose=1, validation_data=TEST, validation_steps=validation_steps,
                  callbacks=[best_model, reduce_lr, early_stop, result_save, history], max_q_size=MAX_Q_SIZE, workers=WORKERS, pickle_safe=True)

history.loss_plot('epoch')

time_end = time.time()
used_time = int(time_end-time_start)
print('used time:%d s'%(used_time))
with open(PATH + 'used_time.txt', 'a+') as t1:
    t1.write('Used time:%d s'%(used_time) + '\n')


# summary_result(PATH=PATH, Fold=Fold, Earlystop=Earlystop)
#summery---------------------------------------------------------------------------------------------
raw_ssf = open(List_path+'sample_submission.csv','r')
raw_lines = raw_ssf.readlines()

ssf = open(PATH+'sample_submission1.csv','w')
ssf.write(raw_lines[0])
raw_lines = raw_lines[1:]

for i in range(len(raw_lines)):
    test_p = DataPath + 'test/' + raw_lines[i].strip().split(',')[0]
    test_data = test_audio(test_p, colum=col, NFFT=NFFT, noverlap=noverlap, frequce=44100, sub_sample=SubSample,
                           pad_to=Pad_to, stft_form=False, stft_handle=False, only_real=False)
    if i % batch_size == 0:
        test_batch_d = test_data
    else:
        test_batch_d = np.vstack((test_batch_d, test_data))

    if (i-batch_size+1)%batch_size == 0 or i == len(raw_lines)-1:
        y_bacth = model.predict(test_batch_d, batch_size=batch_size)
        # y_bacth = np.where(y_bacth>=0.5, 1, 0)
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
#test pred---------------------------------------------------------------------------------------------
r_m = open(PATH+'result_metrics.txt', 'a+')

raw_ssf = open(List_path+'test_list.csv','r')
raw_lines = raw_ssf.readlines()

ssf = open(PATH+'test_pred.csv','w')
# ssf.write(raw_lines[0])
# raw_lines = raw_lines[1:]

for i in range(len(raw_lines)):
    test_p = DataPath + raw_lines[i].strip().split(',')[0]
    test_data = test_audio(test_p, colum=col, NFFT=NFFT, noverlap=noverlap, frequce=44100, sub_sample=SubSample,
                           pad_to=Pad_to, stft_form=False, stft_handle=False, only_real=False)
    if i % batch_size == 0:
        test_batch_d = test_data
    else:
        test_batch_d = np.vstack((test_batch_d, test_data))

    if (i-batch_size+1)%batch_size == 0 or i == len(raw_lines)-1:
        y_bacth = model.predict(test_batch_d, batch_size=batch_size)
        # y_bacth = np.where(y_bacth>=0.5, 1, 0)
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
_, Y = generate_audio_bacth_faster(DataPath, List_path + test_list, colum = col, nb_classes=nb_classes,
                NFFT=NFFT, noverlap=noverlap, frequce=44100, sub_sample=SubSample, pad_to=Pad_to, shuffle=False, stft_form=False, stft_handle=False, only_real=False)

truth = Y
scores = y_all
print("test lwlrap from sklearn.metrics =", calculate_overall_lwlrap_sklearn(truth, scores))
r_m.write("test lwlrap from sklearn.metrics =" + str(calculate_overall_lwlrap_sklearn(truth, scores)) + '\n')
#train pred---------------------------------------------------------------------------------------------
raw_ssf = open(List_path+'train_list.csv','r')
raw_lines = raw_ssf.readlines()

ssf = open(PATH+'train_pred.csv','w')
# ssf.write(raw_lines[0])
# raw_lines = raw_lines[1:]

for i in range(len(raw_lines)):
    test_p = DataPath + raw_lines[i].strip().split(',')[0]
    test_data = test_audio(test_p, colum=col, NFFT=NFFT, noverlap=noverlap, frequce=44100, sub_sample=SubSample,
                           pad_to=Pad_to, stft_form=False, stft_handle=False, only_real=False)
    if i % batch_size == 0:
        test_batch_d = test_data
    else:
        test_batch_d = np.vstack((test_batch_d, test_data))

    if (i-batch_size+1)%batch_size == 0 or i == len(raw_lines)-1:
        y_bacth = model.predict(test_batch_d, batch_size=batch_size)
        # y_bacth = np.where(y_bacth>=0.5, 1, 0)
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

_, Y = generate_audio_bacth_faster(DataPath, List_path + train_list, colum = col, nb_classes=nb_classes,
                NFFT=NFFT, noverlap=noverlap, frequce=44100, sub_sample=SubSample, pad_to=Pad_to, shuffle=False, stft_form=False, stft_handle=False, only_real=False)
truth = Y
scores = y_all
print("train lwlrap from sklearn.metrics =", calculate_overall_lwlrap_sklearn(truth, scores))
r_m.write("train lwlrap from sklearn.metrics =" + str(calculate_overall_lwlrap_sklearn(truth, scores)) + '\n')