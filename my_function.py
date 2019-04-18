#import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from numpy import *
import numpy as np
import os,time,cv2
import scipy.io as scio
from scipy import signal
import wave

def handle_data(Train_x, Test_x,Train_y, Test_y,Fold,Test_size_split=4):
    A_X = vstack((Train_x, Test_x))
    A_Y = vstack((Train_y, Test_y))
    All_size = A_X.shape[0]
    new_all_size = int(All_size/(Fold*Test_size_split))*(Fold*Test_size_split)
    new_A_X = A_X[:new_all_size]
    new_A_Y = A_Y[:new_all_size]
    return new_A_X,new_A_Y

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

def test_data2fft_form(test_data,step_input_size = 153,NFFT = 51,noverlap = 12):
    for i in range(step_input_size):  # 153
        # for j in range(self._bunch_size):
        in_put_1 = test_data[i * (NFFT - noverlap):NFFT + i * (NFFT - noverlap)]
        in_put_1 = np.reshape(in_put_1,[1, NFFT, -1])
        try:
            in_put_2 = np.concatenate([in_put_2, in_put_1], axis=2)
        except:
            in_put_2 = in_put_1
    in_put_2 = np.transpose(in_put_2, [0, 2, 1])
    newdata = in_put_2
    print('data change to fft form succeed')
    return newdata

def generate_video_all_list(Data_Path = '',List_name = '',List_Save_Path = ''):
    '''
    :param Data_Path: dataset's path,main path maybe has many sub_path saving difference kinds of data.
    :param List_Save_Path: generate a list including all data under the Data_Path with label or other message.
    :return: NONE.
    '''
    File_dir = Data_Path
    # file_name(File_dir)
    dir1_name = os.listdir(File_dir)
    # print(len(list_dir))
    dir1_num = len(dir1_name)
    i = 0
    while i < dir1_num:
        if os.path.isdir(File_dir + dir1_name[i]) is not True:
            dir1_name.pop(i)
            dir1_num -= 1
        i += 1
    dir2_name = [0] * len(dir1_name)
    for i in range(len(dir1_name)):
        dir2_list = os.listdir(File_dir + dir1_name[i])
        dir2_list.pop(dir2_list.index('Annotation'))
        dir2_name[i] = dir2_list

    first_path_num = len(dir1_name)
    second_path_num = [0] * first_path_num
    for i in range(first_path_num):
        second_path_num[i] = len(dir2_name[i])

    # print(dir1_name)
    # print(dir2_name)
    i = 0
    j = 0
    f = open(List_Save_Path+List_name, 'w')
    f.write('path fps high wide channel scene label \n')
    while i < first_path_num:
        while j < second_path_num[i]:
            path = File_dir + dir1_name[i] + '/' + dir2_name[i][j]
            dir3_name = os.listdir(path)
            dir3_num = len(dir3_name)
            ii = 0
            while ii < dir3_num:
                if os.path.splitext(str(dir3_name[ii]))[1] != '.avi':
                    dir3_name.pop(ii)
                    dir3_num -= 1
                    ii -= 1
                ii += 1
            for k in range(len(dir3_name)):
                video_path = path + '/' + dir3_name[k]
                fps, h, w, c = fps_resolution(video_path)
                if fps >= 60 and h == 240 and w == 320 and c == 3:
                    wt = path + '/' + dir3_name[k] + ' ' + str(fps) + ' ' + str(h) + ' ' + str(w) + ' ' + str(c) + ' ' + str(j) + ' ' + str(i) + '\n'
                    f.write(wt)
            j += 1
        j = 0
        i += 1
    f.close()
    return

def generate_video_train_test_list(Path = ' ',All_List_Path = ' ',Train_list = ' ',Test_list = ' ',Fold =5,Reture_number = True):
    f = open(All_List_Path, 'r')
    row = len(f.readlines())
    print('original data ',row)
    f.close()
    f = open(All_List_Path, 'r')
    lines = f.readlines()[1:]
    lines = shuffle(lines)
    shuf_f = open(Path+'shuff_all_data_list.txt','w')
    for shuf_l in lines:
        shuf_f.write(shuf_l)
    shuf_f.close()
    for fnum in range(Fold):
        Sub_Path = Path + 'fold' + str(fnum) + '/'
        makedir(Sub_Path)
        test_size = 1/Fold
        train_lines = []
        val_lines = []
        dic = {}
        for line in lines:
            path, fps, high, wide, channel, scene, label = line.strip().split()
            dic.setdefault(label, []).append(path + ' ' + fps + ' ' + high + ' ' + wide + ' ' + channel + ' ' + scene)

        ftrain = open(Sub_Path+Train_list, 'w')
        ftest = open(Sub_Path+Test_list, 'w')
        for label in dic.keys():
            path = dic[label]
            #path = shuffle(dic[label], random_state=2017)
            print('Type {}, # action: {}'.format(label, len(path)))
            fold_bacth_size = int(len(path) * test_size)
            for v_path in path[(fold_bacth_size*fnum):(fold_bacth_size*(fnum+1))]:
                to_write = '{} {}\n'.format(v_path, label)
                val_lines.append(to_write)
                # ftrain.write(to_write)
            for v_path in (path[:(fold_bacth_size*fnum)] + path[(fold_bacth_size*(fnum+1)):]):
                to_write = '{} {}\n'.format(v_path, label)
                train_lines.append(to_write)
                # ftest.write(to_write)

        train_lines = shuffle(train_lines, random_state=2017)
        val_lines = shuffle(val_lines, random_state=2017)
        print('# train: {}, # val: {}'.format(len(train_lines), len(val_lines)))

        for train_line in train_lines:
            ftrain.write(train_line)

        for test_line in val_lines:
            ftest.write(test_line)

        ftrain.close()
        ftest.close()
        f.close()
    print('list generated')
    if Reture_number:
        return row
    else:
        return

def fps_resolution(path):
    cap = cv2.VideoCapture(path)
    fps = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:   # 这里必须加上判断视频是否读取结束的判断,否则播放到最后一帧的时候出现问题了
            if fps == 0:
                try:
                    h,w,c = frame.shape
                except:
                    h,w = frame.shape
                    c = 1
            fps += 1
        else:
            break
    return fps,h,w,c

def generate_video_bacth(Path,batch_size,class_n = 11,fps_need = 60,gray = False,shuffle = False):
    f = open(Path, 'r')
    lines = f.readlines()
    N = len(lines)
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
        video_batch = []
        label_batch = []
        for i in range(current_index, current_index + current_batch_size):
            path, fps, high, wide, channel, scene, label = lines[i].strip().split()
            count = 0
            index = 0
            video = []
            cap = cv2.VideoCapture(path)
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    if count == round(int(fps) * index / fps_need):
                        if gray:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame = np.reshape(frame, [1, 240, 320, 3])
                        try:
                            video = np.append(video, frame, 0)
                        except:
                            video = frame
                        index += 1
                else:
                    break
                count += 1
            video = np.reshape(video, [1, fps_need, 240, 320, 3])
            lb = [0] * class_n
            lb[int(label)] = 1
            lb = np.reshape(lb, [1, class_n])
            label = lb
            try:
                video_batch = np.append(video_batch, video, 0)
                label_batch = np.append(label_batch, label, 0)
            except:
                video_batch = video
                label_batch = label
        yield video_batch,label_batch

def generate_radar_all_list(Data_Path = '',List_name = '',List_Save_Path = ''):
    '''
    :param Data_Path: dataset's path,main path maybe has many sub_path saving difference kinds of data.
    :param List_Save_Path: generate a list including all data under the Data_Path with label or other message.
    :return: NONE.
    '''
    File_dir = Data_Path
    # file_name(File_dir)
    dir1_name = os.listdir(File_dir)
    # print(len(list_dir))
    dir1_num = len(dir1_name)
    i = 0
    while i < dir1_num:
        if os.path.isdir(File_dir + dir1_name[i]) is not True:
            dir1_name.pop(i)
            dir1_num -= 1
        i += 1
    dir2_name = [0] * len(dir1_name)
    for i in range(len(dir1_name)):
        dir2_list = os.listdir(File_dir + dir1_name[i])
        dir2_name[i] = dir2_list

    first_path_num = len(dir1_name)
    second_path_num = [0] * first_path_num
    for i in range(first_path_num):
        second_path_num[i] = len(dir2_name[i])

    i = 0
    j = 0
    f = open(List_Save_Path+List_name, 'w')
    f.write('path people_name label \n')
    while i < first_path_num:
        while j < second_path_num[i]:
            path = File_dir + dir1_name[i] + '/' + dir2_name[i][j]
            dir3_name = os.listdir(path)
            for k in range(len(dir3_name)):
                wt = path + '/' + dir3_name[k]  + ' ' + str(j) + ' ' + str(i) + '\n'
                f.write(wt)
            j += 1
        j = 0
        i += 1
    f.close()
    return

def generate_radar_train_test_list(Path = ' ',All_List_Path = ' ',Train_list = ' ',Test_list = ' ',Fold =5,Reture_number = True):
    f = open(All_List_Path, 'r')
    row = len(f.readlines())
    print('original data ',row)
    f.close()
    f = open(All_List_Path, 'r')
    lines = f.readlines()[1:]
    lines = shuffle(lines)

    shuf_f = open(Path+'shuff_all_data_list.txt','w')
    for shuf_l in lines:
        shuf_f.write(shuf_l)
    shuf_f.close()

    data_dtbt = open(Path + 'data_distribution.txt', 'w')
    dtbt_lines= []
    dtbt_lines.append('original data '+str(row)+'\n')
    for fnum in range(Fold):
        Sub_Path = Path + 'fold' + str(fnum) + '/'
        makedir(Sub_Path)
        test_size = 1/Fold
        train_lines = []
        val_lines = []
        dic = {}
        for line in lines:
            path, people_name, label = line.strip().split()
            dic.setdefault(label, []).append(path + ' ' + people_name)

        ftrain = open(Sub_Path+Train_list, 'w')
        ftest = open(Sub_Path+Test_list, 'w')
        for label in dic.keys():
            path = dic[label]
            #path = shuffle(dic[label], random_state=2017)
            print('Type {}, # action: {}'.format(label, len(path)))
            dtbt_to_w = 'Type {}, # action: {}\n'.format(label, len(path))
            dtbt_lines.append(dtbt_to_w)
            fold_bacth_size = int(len(path) * test_size)
            for v_path in path[(fold_bacth_size*fnum):(fold_bacth_size*(fnum+1))]:
                to_write = '{} {}\n'.format(v_path, label)
                val_lines.append(to_write)
                # ftrain.write(to_write)
            for v_path in (path[:(fold_bacth_size*fnum)] + path[(fold_bacth_size*(fnum+1)):]):
                to_write = '{} {}\n'.format(v_path, label)
                train_lines.append(to_write)
                # ftest.write(to_write)

        train_lines = shuffle(train_lines, random_state=2017)
        val_lines = shuffle(val_lines, random_state=2017)
        print('# train: {}, # val: {}'.format(len(train_lines), len(val_lines)))
        dtbt_to_w = '# train: {}, # val: {}\n'.format(len(train_lines), len(val_lines))
        dtbt_lines.append(dtbt_to_w)

        for train_line in train_lines:
            ftrain.write(train_line)

        for test_line in val_lines:
            ftest.write(test_line)

        ftrain.close()
        ftest.close()
        f.close()

    for dtbt_line in dtbt_lines:
        data_dtbt.write(dtbt_line)
    data_dtbt.close()

    print('list generated')
    if Reture_number:
        return row
    else:
        return

def generate_radar_bacth(Path,batch_size = 64,colum = 6000,nb_classes = 3,
                    NFFT = 51,noverlap = 12,frequce = 2000,sub_freq = 2000,pad_to = 299,shuffle=False,stft_form = True,stft_handle = False,only_real = False,fast = False):
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
    if fast:
        colum = int(colum / (int(frequce / sub_freq)))
        step_input_size = int((colum - NFFT) / (NFFT - noverlap)) + 1  # 153
        f = open(Path, 'r')
        lines = f.readlines()
        N = len(lines)
        if shuffle:
            random.shuffle(lines)
        ALL_X = np.zeros((N, colum))
        ALL_Y = np.zeros((N, nb_classes))
        for l in range(N):
            path, people_name, label = lines[l].strip().split()
            data_path = path
            data = scio.loadmat(data_path)
            Data_label = data['d']
            dat = Data_label[0, 0::int(frequce / sub_freq)]
            ALL_X[l] = dat
            ALL_Y[l, np.int8(label)] = 1
        batch_index = 0
        while True:
            current_index = (batch_index * batch_size) % N
            if N >= (current_index + batch_size):
                current_batch_size = batch_size
                batch_index += 1
            else:
                current_batch_size = N - current_index
                batch_index = 0
            X_batch = ALL_X[current_index: current_index + current_batch_size]
            Y_batch = ALL_Y[current_index: current_index + current_batch_size]

            if stft_form:
                X_batch, Y_batch = handle_data2fft_form(Train_x=X_batch, Test_x=None, Train_y=Y_batch, Test_y=None,
                                                        Fold=1, Test_size_split=1, step_input_size=step_input_size,
                                                        NFFT=NFFT,
                                                        noverlap=noverlap)

            if stft_handle:
                New_X_batch = np.zeros((current_batch_size, int(pad_to / 2 + 1), step_input_size))
                for i in range(current_batch_size):
                    freqs_1, t_1, spec_1 = signal.spectrogram(X_batch[i], fs=frequce, window=('hamming'), nperseg=NFFT,
                                                              noverlap=noverlap,
                                                              nfft=pad_to, detrend='constant', return_onesided=True,
                                                              scaling='density', axis=-1, mode='complex')
                    if only_real:
                        spec_1 = 10 * np.log10(abs(spec_1.real) + np.spacing(1))
                    else:
                        spec_1 = 10 * np.log10(abs(spec_1) + np.spacing(1))
                    New_X_batch[i] = spec_1
                X_batch = New_X_batch

            yield (X_batch, Y_batch)
    else:
        colum = int(colum/(int(frequce/sub_freq)))
        step_input_size = int((colum - NFFT) / (NFFT - noverlap)) + 1 #153
        f = open(Path, 'r')
        lines = f.readlines()
        N = len(lines)
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
            X_batch = np.zeros((current_batch_size, colum))
            Y_batch = np.zeros((current_batch_size, nb_classes))
            data_index = 0
            for i in range(current_index, current_index + current_batch_size):
                path, people_name, label = lines[i].strip().split()
                data_path = path
                data = scio.loadmat(data_path)
                Data_label = data['d']
                dat = Data_label[0, 0::int(frequce/sub_freq)]
                X_batch[data_index] = dat
                Y_batch[data_index,np.int8(label)] = 1
                data_index += 1
            if stft_form:
                X_batch, Y_batch = handle_data2fft_form(Train_x=X_batch, Test_x=None, Train_y=Y_batch, Test_y=None,
                                                    Fold=1, Test_size_split=1, step_input_size=step_input_size, NFFT=NFFT,
                                                    noverlap=noverlap)

            if stft_handle:
                New_X_batch = np.zeros((current_batch_size, int(pad_to/2+1),step_input_size))
                for i in range(current_batch_size):
                    freqs_1, t_1, spec_1 = signal.spectrogram(X_batch[i], fs=frequce, window=('hamming'), nperseg=NFFT, noverlap=noverlap,
                                                              nfft=pad_to, detrend='constant', return_onesided=True,
                                                              scaling='density', axis=-1, mode='complex')
                    # spec_1, freqs_1, t_1, im_1 = plt.specgram(X_batch[i], NFFT=NFFT, Fs=frequce, Fc=None, detrend=None, window=None,
                    #                                           noverlap=noverlap, cmap=None, xextent=None, pad_to=pad_to,
                    #                                           sides='onesided',
                    #                                           scale_by_freq=None, mode='psd', scale=None, vmin=None,
                    #                                           vmax=None,
                    #                                           hold=None, data=None)
                    if only_real:
                        spec_1 = 10 * np.log10(abs(spec_1.real) + np.spacing(1))
                    else:
                        spec_1 = 10 * np.log10(abs(spec_1) + np.spacing(1))
                    New_X_batch[i] = spec_1
                X_batch = New_X_batch

            yield (X_batch, Y_batch)

def generate_radar_bacth_faster(Path,colum = 6000,nb_classes = 7,NFFT = 51,noverlap = 12,frequce = 2000,
                         sub_freq = 2000,pad_to = 299,shuffle=False,stft_form = True,stft_handle = False,conv3D = False,only_real = False):
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
    colum = int(colum/(int(frequce/sub_freq)))
    step_input_size = int((colum - NFFT) / (NFFT - noverlap)) + 1 #153
    f = open(Path, 'r')
    lines = f.readlines()
    N = len(lines)
    if shuffle:
        random.shuffle(lines)
    ALL_X = np.zeros((N, colum))
    ALL_Y = np.zeros((N, nb_classes))
    for l in range(N):
        path, people_name, label = lines[l].strip().split()
        data_path = path
        data = scio.loadmat(data_path)
        Data_label = data['d']
        dat = Data_label[0, 0::int(frequce / sub_freq)]
        ALL_X[l] = dat
        ALL_Y[l, np.int8(label)] = 1
    if stft_form:
        ALL_X, ALL_Y = handle_data2fft_form(Train_x=ALL_X, Test_x=None, Train_y=ALL_Y, Test_y=None,
                                            Fold=1, Test_size_split=1, step_input_size=step_input_size, NFFT=NFFT,
                                            noverlap=noverlap)

    if stft_handle:
        New_X_batch = np.zeros((N, int(pad_to / 2 + 1), step_input_size))
        for i in range(N):
            freqs_1, t_1, spec_1 = signal.spectrogram(ALL_X[i], fs=frequce, window=('hamming'), nperseg=NFFT,
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


def generate_audio_bacth_faster(Path,colum = 600000,nb_classes = 80,NFFT = 51,noverlap = 12,frequce = 44100,
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
    f = open(Path, 'r')
    lines = f.readlines()
    N = len(lines)
    if shuffle:
        random.shuffle(lines)
    ALL_X = np.zeros((N, sub_sample))
    ALL_Y = np.zeros((N, nb_classes))
    for l in range(N):
        line = lines[l]
        d_label = line.strip().split(',')[1:]
        # D_label = np.zeros([1, 80])
        for i in range(nb_classes):
            ALL_Y[0, i] = np.int(d_label[i])

        # print(ALL_Y)
        wav_p = line.strip().split(',')[0]
        # 打开wav文件 ，open返回一个的是一个Wave_read类的实例，通过调用它的方法读取WAV文件的格式和数据。
        f = wave.open(wav_p, "rb")
        # 读取格式信息
        # 一次性返回所有的WAV文件的格式信息，它返回的是一个组元(tuple)：声道数, 量化位数（byte单位）, 采
        # 样频率, 采样点数, 压缩类型, 压缩类型的描述。wave模块只支持非压缩的数据，因此可以忽略最后两个信息
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        # print(nchannels)
        # 读取波形数据
        # 读取声音数据，传递一个参数指定需要读取的长度（以取样点为单位）
        str_data = f.readframes(nframes)
        f.close()
        # 将波形数据转换成数组
        # 需要根据声道数和量化单位，将读取的二进制数据转换为一个可以计算的数组
        wave_data = np.fromstring(str_data, dtype=np.short)
        wave_data = wave_data.T
        wave_shape = wave_data.shape
        new_wave_d = np.zeros([1, colum])
        if wave_shape[0] >= colum:
            new_wave_d[0, :] = wave_data[int((wave_shape[0] - colum) / 2):int((wave_shape[0] - colum) / 2) + colum]
        else:
            new_wave_d[0,
            int((colum - wave_shape[0]) / 2):int((colum - wave_shape[0]) / 2) + wave_shape[0]] = wave_data

        new_wave_d = new_wave_d[0, 0::int(colum / sub_sample)]
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


def generate_radar_bacth_original(Path,batch_size = 64,colum = 6000,nb_classes = 3,
                    NFFT = 51,noverlap = 12,frequce = 2000,sub_freq = 2000,pad_to = 299,
                                  shuffle=False,stft_form = True,stft_handle = False,only_real = False,use_phase = False):
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
    colum = int(colum/(int(frequce/sub_freq)))
    step_input_size = int((colum - NFFT) / (NFFT - noverlap)) + 1 #153
    f = open(Path, 'r')
    lines = f.readlines()
    N = len(lines)
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
        X_batch = np.zeros((current_batch_size, colum))
        Y_batch = np.zeros((current_batch_size, nb_classes))
        data_index = 0
        for i in range(current_index, current_index + current_batch_size):
            path, people_name, label = lines[i].strip().split()
            data_path = path
            data = scio.loadmat(data_path)
            Data_label = data['d']
            dat = Data_label[0, 0::int(frequce/sub_freq)]
            X_batch[data_index] = dat
            Y_batch[data_index,np.int8(label)] = 1
            data_index += 1
        if stft_form:
            X_batch, Y_batch = handle_data2fft_form(Train_x=X_batch, Test_x=None, Train_y=Y_batch, Test_y=None,
                                                Fold=1, Test_size_split=1, step_input_size=step_input_size, NFFT=NFFT,
                                                noverlap=noverlap)

        if stft_handle:
            New_X_batch = np.zeros((current_batch_size, int(pad_to/2+1),step_input_size))
            for i in range(current_batch_size):
                freqs_1, t_1, spec_1 = signal.spectrogram(X_batch[i], fs=frequce, window=('rectangular'), nperseg=NFFT, noverlap=noverlap,
                                                          nfft=pad_to, detrend=False, return_onesided=True,
                                                          scaling='spectrum', axis=-1, mode='complex')
                # spec_1, freqs_1, t_1, im_1 = plt.specgram(X_batch[i], NFFT=NFFT, Fs=frequce, Fc=None, detrend=None, window=None,
                #                                           noverlap=noverlap, cmap=None, xextent=None, pad_to=pad_to,
                #                                           sides='onesided',
                #                                           scale_by_freq=None, mode='psd', scale=None, vmin=None,
                #                                           vmax=None,
                #                                           hold=None, data=None)
                if only_real:
                    spec_1 = 10 * np.log10(abs(spec_1.real) + np.spacing(1))
                else:
                    spec_1 = 10 * np.log10(abs(spec_1) + np.spacing(1))
                New_X_batch[i] = spec_1
            if use_phase:
                phase_X_batch = np.zeros((current_batch_size, int(pad_to/2+1),step_input_size))
                for i in range(current_batch_size):
                    freqs_1, t_1, spec_1 = signal.spectrogram(X_batch[i], fs=frequce, window=('rectangular'),
                                                              nperseg=NFFT, noverlap=noverlap,
                                                              nfft=pad_to, detrend=False, return_onesided=True,
                                                              scaling='spectrum', axis=-1, mode='phase')
                    phase_X_batch[i] = spec_1
                New_X_batch = New_X_batch[:,:,:,np.newaxis]
                # print(New_X_batch.shape)
                phase_X_batch = phase_X_batch[:,:,:,np.newaxis]
                New_X_batch = np.concatenate([New_X_batch,phase_X_batch],-1)
            X_batch = New_X_batch

        yield (X_batch, Y_batch)

def generate_radar_bacth_original_tanh(Path,batch_size = 64,colum = 6000,nb_classes = 3,
                    NFFT = 51,noverlap = 12,frequce = 2000,sub_freq = 2000,pad_to = 299,
                                  shuffle=False,stft_form = True,stft_handle = False,only_real = False,use_phase = False,tanh_s = [1,1],fast = False):
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
    colum = int(colum/(int(frequce/sub_freq)))
    step_input_size = int((colum - NFFT) / (NFFT - noverlap)) + 1 #153
    f = open(Path, 'r')
    lines = f.readlines()
    N = len(lines)
    if shuffle:
        random.shuffle(lines)
    if fast:
        ALL_X = np.zeros((N, colum))
        ALL_Y = np.zeros((N, nb_classes))
        for l in range(N):
            path, people_name, label = lines[l].strip().split()
            data_path = path
            data = scio.loadmat(data_path)
            Data_label = data['d']
            dat = Data_label[0, 0::int(frequce / sub_freq)]
            ALL_X[l] = dat
            ALL_Y[l, np.int8(label)] = 1
    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0

        if fast:
            X_batch = ALL_X[current_index: current_index + current_batch_size]
            Y_batch = ALL_Y[current_index: current_index + current_batch_size]
        else :
            X_batch = np.zeros((current_batch_size, colum))
            Y_batch = np.zeros((current_batch_size, nb_classes))
            data_index = 0
            for i in range(current_index, current_index + current_batch_size):
                path, people_name, label = lines[i].strip().split()
                data_path = path
                data = scio.loadmat(data_path)
                Data_label = data['d']
                dat = Data_label[0, 0::int(frequce/sub_freq)]
                X_batch[data_index] = dat
                Y_batch[data_index,np.int8(label)] = 1
                data_index += 1

        if stft_form:
            X_batch, Y_batch = handle_data2fft_form(Train_x=X_batch, Test_x=None, Train_y=Y_batch, Test_y=None,
                                                Fold=1, Test_size_split=1, step_input_size=step_input_size, NFFT=NFFT,
                                                noverlap=noverlap)

        if stft_handle:
            New_X_batch = np.zeros((current_batch_size, int(pad_to/2+1),step_input_size))
            for i in range(current_batch_size):
                freqs_1, t_1, spec_1 = signal.spectrogram(X_batch[i], fs=frequce, window=('rectangular'), nperseg=NFFT, noverlap=noverlap,
                                                          nfft=pad_to, detrend=False, return_onesided=True,
                                                          scaling='spectrum', axis=-1, mode='complex')
                # spec_1, freqs_1, t_1, im_1 = plt.specgram(X_batch[i], NFFT=NFFT, Fs=frequce, Fc=None, detrend=None, window=None,
                #                                           noverlap=noverlap, cmap=None, xextent=None, pad_to=pad_to,
                #                                           sides='onesided',
                #                                           scale_by_freq=None, mode='psd', scale=None, vmin=None,
                #                                           vmax=None,
                #                                           hold=None, data=None)
                if only_real:
                    spec_1 = 10 * np.log10(abs(spec_1.real) + np.spacing(1))
                else:
                    spec_1 = 10 * np.log10(abs(spec_1) + np.spacing(1))
                New_X_batch[i] = spec_1
            if use_phase:
                phase_X_batch = np.zeros((current_batch_size, int(pad_to/2+1),step_input_size))
                for i in range(current_batch_size):
                    freqs_1, t_1, spec_1 = signal.spectrogram(X_batch[i], fs=frequce, window=('rectangular'),
                                                              nperseg=NFFT, noverlap=noverlap,
                                                              nfft=pad_to, detrend=False, return_onesided=True,
                                                              scaling='spectrum', axis=-1, mode='phase')
                    phase_X_batch[i] = tanh_s[0]*tanh(tanh_s[1]*spec_1)
                New_X_batch = New_X_batch[:,:,:,np.newaxis]
                # print(New_X_batch.shape)
                phase_X_batch = phase_X_batch[:,:,:,np.newaxis]
                New_X_batch = np.concatenate([New_X_batch,phase_X_batch],-1)
            X_batch = New_X_batch

        yield (X_batch, Y_batch)

def generate_radar_bacth_mystft(Path,batch_size = 64,colum = 6000,nb_classes = 3,
                    NFFT = 51,noverlap = 12,frequce = 2000,sub_freq = 2000,pad_to = 299,shuffle=False,stft_form = True,stft_handle = False,only_real = False):
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
    colum = int(colum/(int(frequce/sub_freq)))
    step_input_size = int((colum - NFFT) / (NFFT - noverlap)) + 1 #153
    f = open(Path, 'r')
    lines = f.readlines()
    N = len(lines)
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
        X_batch = np.zeros((current_batch_size, colum))
        Y_batch = np.zeros((current_batch_size, nb_classes))
        data_index = 0
        for i in range(current_index, current_index + current_batch_size):
            path, people_name, label = lines[i].strip().split()
            data_path = path
            data = scio.loadmat(data_path)
            Data_label = data['d']
            dat = Data_label[0, 0::int(frequce/sub_freq)]
            X_batch[data_index] = dat
            Y_batch[data_index,np.int8(label)] = 1
            data_index += 1
        if stft_form:
            X_batch, Y_batch = handle_data2fft_form(Train_x=X_batch, Test_x=None, Train_y=Y_batch, Test_y=None,
                                                Fold=1, Test_size_split=1, step_input_size=step_input_size, NFFT=NFFT,
                                                noverlap=noverlap)

        if stft_handle:
            X_batch, Y_batch = handle_data2fft_form(Train_x=X_batch, Test_x=None, Train_y=Y_batch, Test_y=None,
                                                    Fold=1, Test_size_split=1, step_input_size=step_input_size,
                                                    NFFT=NFFT,
                                                    noverlap=noverlap)
            New_X_batch = np.zeros((current_batch_size, int(pad_to/2+1),step_input_size))

            w_metrix_real = np.zeros([NFFT, int(pad_to/2+1)])
            for k in range(int(pad_to/2+1)):
                for i in range(NFFT):
                    w_metrix_real[i, k] = cos(2 * k * pi * (i / NFFT) / (colum / int(sub_freq/2+1)))
            w_metrix_imag = np.zeros([NFFT, int(pad_to/2+1)])
            for k in range(int(pad_to/2+1)):
                for i in range(NFFT):
                    w_metrix_imag[i, k] = -sin(2 * k * pi * (i / NFFT) / (colum / int(sub_freq/2+1)))

            for i in range(current_batch_size):
                real_p = np.transpose(np.dot(X_batch[i], w_metrix_real), [1, 0])
                imag_p = np.transpose(np.dot(X_batch[i], w_metrix_imag), [1, 0])
                # spec_1, freqs_1, t_1, im_1 = plt.specgram(X_batch[i], NFFT=NFFT, Fs=frequce, Fc=None, detrend=None, window=None,
                #                                           noverlap=noverlap, cmap=None, xextent=None, pad_to=pad_to,
                #                                           sides='onesided',
                #                                           scale_by_freq=None, mode='psd', scale=None, vmin=None,
                #                                           vmax=None,
                #                                           hold=None, data=None)
                if only_real:
                    test_mystft = 10 * np.log10(abs(real_p / NFFT) + np.spacing(1))
                else:
                    test_mystft = np.sqrt(np.power(real_p, 2) + np.power(imag_p, 2))
                    test_mystft = 10 * np.log10(abs(test_mystft / NFFT) + np.spacing(1))
                New_X_batch[i] = test_mystft
            X_batch = New_X_batch

        yield (X_batch, Y_batch)

def test_radar_bacth(Path,colum = 6000,nb_classes = 3,NFFT = 51,noverlap = 12,frequce = 2000,pad_to = 299,stft_form = True,stft_handle = False):
    '''
    according to the data list(txt file) generate test data batch
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
    step_input_size = int((colum - NFFT) / (NFFT - noverlap)) + 1 #153
    f = open(Path, 'r')
    lines = f.readlines()
    N = len(lines)

    X_batch = np.zeros((N, colum))
    Y_batch = np.zeros((N, nb_classes))
    data_index = 0
    for i in range(N):
        path, people_name, label = lines[i].strip().split()
        data_path = path
        data = scio.loadmat(data_path)
        Data_label = data['d']
        dat = Data_label[0, :colum]
        X_batch[data_index] = dat
        Y_batch[data_index,np.int8(label)] = 1
        data_index += 1
    if stft_form:
        X_batch, Y_batch = handle_data2fft_form(Train_x=X_batch, Test_x=None, Train_y=Y_batch, Test_y=None,
                                            Fold=1, Test_size_split=1, step_input_size=step_input_size, NFFT=NFFT,
                                            noverlap=noverlap)

    if stft_handle:
        New_X_batch = np.zeros((N, int(pad_to/2+1),step_input_size))
        for i in range(N):
            freqs_1, t_1, spec_1 = signal.spectrogram(X_batch[i], fs=frequce, window=('hamming'), nperseg=NFFT, noverlap=noverlap,
                                                      nfft=pad_to, detrend='constant', return_onesided=True,
                                                      scaling='density', axis=-1, mode='psd')
            spec_1 = 10 * np.log10(abs(spec_1) + np.spacing(1))
            New_X_batch[i] = spec_1
        X_batch = New_X_batch

    return (X_batch, Y_batch)

def generate_data(all_x,all_y,fold,fold_index = 0):
    All_size = all_x.shape[0]
    fold_batch = int(All_size/fold)
    ts_x = all_x[fold_index*fold_batch:(fold_index+1)*fold_batch]
    ts_y = all_y[fold_index*fold_batch:(fold_index+1)*fold_batch]
    if fold_index == 0:
        tra_x = all_x[(fold_index+1)*fold_batch:]
        tra_y = all_y[(fold_index+1)*fold_batch:]
    elif fold_index == int(fold-1):
        tra_x = all_x[:fold_index*fold_batch]
        tra_y = all_y[:fold_index*fold_batch]
    else :
        tra_x = vstack((all_x[:fold_index*fold_batch],all_x[(fold_index + 1) * fold_batch:]))
        tra_y = vstack((all_y[:fold_index*fold_batch],all_y[(fold_index + 1) * fold_batch:]))
    return tra_x,tra_y,ts_x,ts_y

def generate_list(files_list,fold = 5,fold_index = 0):
    '''
    other function is generate_batch
    :param files_list:
    :param fold:
    :param fold_index:
    :return:
    '''
    All_size = len(files_list)
    fold_batch = int(All_size / fold)
    test_list = files_list[fold_index * fold_batch:(fold_index + 1) * fold_batch]
    if fold_index == 0:
        train_list = files_list[(fold_index + 1) * fold_batch:]
    elif fold_index == int(fold - 1):
        train_list = files_list[:fold_index * fold_batch]
    else:
        train_list = files_list[:fold_index * fold_batch] + files_list[(fold_index + 1) * fold_batch:]
    return train_list, test_list

def generator_batch(data_list,path,batch_size = 64,colum = 6000,nb_classes = 3,step_input_size = 153,
                    NFFT = 51,noverlap = 12,shuffle=False,exist_list = True,fnum = 0,Training = True):
    '''
    data in same fold,has shape[6003],6000 are features,3 are label.
    other function :generate_list
    :param data_list:
    :param path:
    :param batch_size:
    :param colum:
    :param nb_classes:
    :param step_input_size:
    :param NFFT:
    :param noverlap:
    :param shuffle:
    :param exist_list:
    :param fnum:
    :param Training:
    :return:
    '''
    N = len(data_list)
    if shuffle:
        random.shuffle(data_list)
    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0

        X_batch = np.zeros((current_batch_size, colum))
        Y_batch = np.zeros((current_batch_size, nb_classes))
        data_index = 0
        for i in range(current_index, current_index + current_batch_size):
            data_name = data_list[i]
            if exist_list:
                if ((Training and fnum != 0) or (not Training and fnum == 0)) and i == 0:
                    data_name = data_name[1:-1]
                else:
                    data_name = data_name[2:-1]
            data_path = path+data_name
            data = scio.loadmat(data_path)
            Data_label = data['d']
            dat = Data_label[0,:colum]
            label = Data_label[0,colum:]
            X_batch[data_index] = dat
            Y_batch[data_index] = label
            data_index += 1

        X_batch, Y_batch = handle_data2fft_form(Train_x = X_batch, Test_x = None,Train_y = Y_batch, Test_y = None,
                                                Fold = 1,Test_size_split=1,step_input_size= step_input_size,NFFT = NFFT,noverlap = noverlap)

        yield (X_batch, Y_batch)

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

def acc_var_result(test_batch_size,sess,test_fold,tX,tY,test_epoch_number):
    for k in range(test_epoch_number):
        var_1 = sess.run(test_fold._cost,
                         feed_dict={test_fold._xs: tX[k * test_batch_size:(k + 1) * test_batch_size, :],
                                    test_fold._ys: tY[k * test_batch_size:(k + 1) * test_batch_size, :]})
        accuracy_1 = sess.run(test_fold.accuracy, feed_dict={
            test_fold._xs: tX[k * test_batch_size:(k + 1) * test_batch_size, :],
            test_fold._ys: tY[k * test_batch_size:(k + 1) * test_batch_size, :]})
        # accuracy_2 = sess.run(test_fold[2].accuracy, feed_dict={
        #     test_fold[2]._xs: test_x[k * test_batch_size:(k + 1) * test_batch_size, :],
        #     test_fold[2]._ys: test_y[k * test_batch_size:(k + 1) * test_batch_size, :]})
        if k == 0:
            var = var_1
            accuracy = accuracy_1
            # play_acc = accuracy_2
        else:
            var = var + var_1
            accuracy = accuracy + accuracy_1
            # play_acc = play_acc + accuracy_2
    acry = accuracy / test_epoch_number
    return acry,var

def check_point(i,PATH,start_time_sum,var,acry,f,sess,saver,check_point_monitor = 'var',
                early_stop_var = 0,early_stop_acc = 0,early_stop_numb = 0,early_stop = 50,epoch = 200):

    if check_point_monitor == 'var':
        if var < early_stop_var:
            early_stop_numb = 0
            early_stop_var = var
            early_stop_acc = acry
            print('check point')
            saver.save(sess, PATH + 'model.ckpt')
        else:
            early_stop_numb = early_stop_numb + 1
            if early_stop_numb == early_stop:
                i = epoch + 1
                end_time_sum = time.time()
                sum_time = end_time_sum - start_time_sum
                need_h = sum_time // 3600
                need_m = (sum_time % 3600) // 60
                need_s = (sum_time % 3600) % 60
                f.write('early stop , used time : %d h %d m %d s\n' % (need_h, need_m, need_s))
                f.write('best var is : %f,accuracy is %f' % (early_stop_var,early_stop_acc))
                print('early stop , used time : %d h %d m %d s' % (need_h, need_m, need_s))
                print('best var is : %f,accuracy is %f' % (early_stop_var,early_stop_acc))
    elif check_point_monitor == 'acc':
        if acry > early_stop_acc:
            early_stop_numb = 0
            early_stop_acc = acry
            print('check point')
            saver.save(sess, PATH + 'model.ckpt')
        else:
            early_stop_numb = early_stop_numb + 1
            if early_stop_numb == early_stop:
                i = epoch + 1
                end_time_sum = time.time()
                sum_time = end_time_sum - start_time_sum
                need_h = sum_time // 3600
                need_m = (sum_time % 3600) // 60
                need_s = (sum_time % 3600) % 60
                f.write('early stop , used time : %d h %d m %d s\n' % (need_h, need_m, need_s))
                f.write('best acc is : %f' % early_stop_acc)
                print('early stop , used time : %d h %d m %d s' % (need_h, need_m, need_s))
                print('best acc is :', early_stop_acc)
    return i,early_stop_numb,early_stop_acc,early_stop_var

def learning_rate_reduce(learn_r,var,acry,check_point_monitor = 'var',
                lr_var = 0,lr_acc = 0,lr_numb = 0):
    lr_factor = 0.5
    patience = 20
    min_learning_rate = 0.00001
    if check_point_monitor == 'var':
        if var < lr_var:
            lr_numb = 0
            lr_var = var
        else:
            lr_numb = lr_numb + 1
            if lr_numb == patience:
                lr_numb = 0
                learn_r = lr_factor*learn_r
                if learn_r <= min_learning_rate :
                    learn_r = min_learning_rate
                print('learning rate reduce to %f' % (learn_r))
    elif check_point_monitor == 'acc':
        if acry > lr_acc:
            lr_numb = 0
            lr_acc = acry
        else:
            lr_numb = lr_numb + 1
            if lr_numb == patience:
                lr_numb = 0
                learn_r = lr_factor * learn_r
                if learn_r <= min_learning_rate:
                    learn_r = min_learning_rate
                print('learning rate reduce to %f' % (learn_r))
    return learn_r,lr_numb,lr_acc,lr_var

def show_time(i,real_epoch,need_t, acry, var, train_acc, loss, l_r = 0.0001):
    if need_t >= 3600:
        need_h = need_t // 3600
        need_m = (need_t % 3600) // 60
        need_s = (need_t % 3600) % 60
        print('%d/%d'%(real_epoch,i), 'epoch','train_acc:%f,test_acc:%f,train_loss:%f,test_var:%f'%(train_acc, acry, loss, var),
              'learning rate: %f' % (l_r),'ETA : %d h %d m %d s' % (need_h, need_m, need_s))

    elif need_t >= 60 and need_t < 3600:
        need_m = need_t // 60
        need_s = need_t % 60
        print('%d/%d'%(real_epoch,i), 'epoch','train_acc:%f,test_acc:%f,train_loss:%f,test_var:%f'%(train_acc, acry, loss, var),
              'learning rate: %f' % (l_r),'ETA : 0 h %d m %d s' % (need_m, need_s))
    else:
        need_s = need_t % 60
        print('%d/%d'%(real_epoch,i), 'epoch','train_acc:%f,test_acc:%f,train_loss:%f,test_var:%f'%(train_acc, acry, loss, var),
              'learning rate: %f' % (l_r),'ETA : 0 h 0 m %d s' % (need_s))
    return

def summary_result(PATH,Fold=5,Earlystop=50):
    write_all_result = 0
    acc_save = {}
    for fnum in range(Fold):
        Sub_path = PATH + 'fold' + str(fnum) + '/'
        new = []
        i = 1
        with open(Sub_path + 'result.txt', 'r') as f:
            for line in f:
                # if i % 2 == 0:
                #     pass
                # else:
                temp1 = line.strip('\n')  # 去掉每行最后的换行符'\n'
                # temp1 = temp1.strip(' ')
                temp2 = temp1.split(',')  # 以','为标志，将每行分割成列表
                if i == 1:
                    temp2.insert(0, 'fold')
                    #temp2.append('used_time')
                new.append(temp2)  # 将上一步得到的列表添加到new中
                i = i + 1
        with open(PATH + 'all_result.txt', 'a+') as f1:
            if write_all_result == 0:
                new_1 = str(new[0]) + '\n'
                f1.write(new_1)
                write_all_result = write_all_result + 1
            ad_fold = new[-(Earlystop + 1)]
            acc_save[fnum] = ad_fold[4]
            ad_fold.insert([0][0], 'fold' + str(fnum))
            #ad_fold.append(str(used_time))
            ad_fold = str(ad_fold) + '\n'
            f1.write(ad_fold)
        del new
        # history.loss_plot('epoch')
    all_acc = 0
    for fnum in range(Fold):
        all_acc = float(acc_save[fnum]) + all_acc
    average_acc = all_acc / Fold
    with open(PATH + 'all_result.txt', 'a+') as f1:
        f1.write(str(average_acc) + '\n')

def my_train(train_x,train_y,test_x,test_y,PATH,train_fold,test_fold,train_config,saver,sess,
    test_size_split = 4,early_stop = 50,epoch = 200000,check_point_monitor = 'acc',fold = 5,plot = False,learn_rate = 0.001):

    dataset_size, col = train_x.shape
    label_size, label_num = test_y.shape
    print('new dataset size is %d,feature %d' % (dataset_size, col))
    print('new label size is %d,label number %d' % (label_size, label_num))
    test_batch_size = int(label_size / test_size_split)
    epoch_to_show = math.ceil(dataset_size / train_config.batch_size)
    show_number = epoch // epoch_to_show
    if plot == True :
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.ion()
        plt.show()
    save_fig = 1
    loss_old = 0
    var_old = 0
    early_stop_var = 0
    early_stop_acc = 0
    # test_x = test_x.reshape([test_config.batch_size, train_config.time_steps, train_config.input_size])
    # test_y = test_y.reshape([test_config.batch_size, 1, train_config.output_size])
    start = 0
    start_next = 0
    end = 0
    i = 0
    real_epoch = 0
    early_stop_numb = 0
    lr_var = 0
    lr_acc = 0
    lr_numb = 0
    f = open(PATH + 'result.txt', 'a+')
    f.write('real_epoch epoch train_acc test_acc train_loss test_var learning_rate\n')
    start_time = time.time()
    start_time_sum = start_time
    while i <= epoch:
        start = start_next
        end_set = start + train_config.batch_size
        if end_set <= dataset_size:
            end = end_set
            start_next = start + train_config.batch_size
            batch_xs = train_x[start:end]
            batch_ys = train_y[start:end]
        else:
            end_1 = dataset_size
            end_2 = end_set - dataset_size
            batch_xs = vstack((train_x[start:end_1], train_x[0:end_2]))
            batch_ys = vstack((train_y[start:end_1], train_y[0:end_2]))
            start_next = end_2

        if i % epoch_to_show == 0:
            real_epoch = real_epoch + 1
            end_time = time.time()
            epoch_t = end_time - start_time
            start_time = end_time
            need_t = epoch_t * (show_number - i // epoch_to_show)
            need_t = round(need_t)
            # loss = sess.run(train_fold._cost, feed_dict={train_fold._xs: batch_xs, train_fold._ys: batch_ys})
            # train_acc = sess.run(train_fold.accuracy,
            #                      feed_dict={train_fold._xs: batch_xs, train_fold._ys: batch_ys})
            # print('train acc :', train_acc)
            train_acc, loss = acc_var_result(test_batch_size, sess, test_fold, tX=train_x, tY=train_y,test_epoch_number=test_size_split*(fold-1))
            acry,var =  acc_var_result(test_batch_size, sess, test_fold, tX = test_x, tY = test_y, test_epoch_number = test_size_split)
            # for k in range(test_size_split):
            #     var_1 = sess.run(test_fold._cost,
            #                      feed_dict={test_fold._xs: test_x[k * test_batch_size:(k + 1) * test_batch_size, :],
            #                                 test_fold._ys: test_y[k * test_batch_size:(k + 1) * test_batch_size, :]})
            #     accuracy_1 = sess.run(test_fold.accuracy, feed_dict={
            #         test_fold._xs: test_x[k * test_batch_size:(k + 1) * test_batch_size, :],
            #         test_fold._ys: test_y[k * test_batch_size:(k + 1) * test_batch_size, :]})
            #     # accuracy_2 = sess.run(test_fold[2].accuracy, feed_dict={
            #     #     test_fold[2]._xs: test_x[k * test_batch_size:(k + 1) * test_batch_size, :],
            #     #     test_fold[2]._ys: test_y[k * test_batch_size:(k + 1) * test_batch_size, :]})
            #     if k == 0:
            #         var = var_1
            #         accuracy = accuracy_1
            #         # play_acc = accuracy_2
            #     else:
            #         var = var + var_1
            #         accuracy = accuracy + accuracy_1
            #         # play_acc = play_acc + accuracy_2
            # acry = accuracy / test_size_split
            # p_a = play_acc / test_size_split
            f.write('%d' % real_epoch+ ' %d' % i +' %f' % train_acc + ' %f' % acry + ' %f' % loss + ' %f' % var + '%f' % learn_rate + '\n')

            i, early_stop_numb, early_stop_acc, early_stop_var = check_point(i, PATH, start_time_sum, var, acry,
                    f, sess, saver,check_point_monitor=check_point_monitor,early_stop_var=early_stop_var,
                    early_stop_acc=early_stop_acc, early_stop_numb=early_stop_numb, early_stop=early_stop, epoch=epoch)

            learn_rate, lr_numb, lr_acc, lr_var = learning_rate_reduce(learn_rate,var,acry,check_point_monitor = 'var',
                lr_var = lr_var,lr_acc = lr_acc,lr_numb = lr_numb)
            # # var_1 = sess.run(test_cnn2._cost,
            # #                  feed_dict={test_cnn2._xs: test_x[0:776, :], test_cnn2._ys: test_y[0:776, :]})
            # # var_2 = sess.run(test_cnn2._cost,
            # #                  feed_dict={test_cnn2._xs: test_x[776:1552, :], test_cnn2._ys: test_y[776:1552, :]})
            # # var_3 = sess.run(test_cnn2._cost,
            # #                  feed_dict={test_cnn2._xs: test_x[1552:2328, :], test_cnn2._ys: test_y[1552:2328, :]})
            # # var_4 = sess.run(test_cnn2._cost,
            # #                  feed_dict={test_cnn2._xs: test_x[2328:3104, :], test_cnn2._ys: test_y[2328:3104, :]})
            # # var = (var_1 + var_2 + var_3 + var_4)
            # # accuracy_1 = sess.run(test_cnn2.accuracy, feed_dict={test_cnn2._xs: test_x[0:776, :],
            # #                                                      test_cnn2._ys: test_y[0:776, :]})
            # # accuracy_2 = sess.run(test_cnn2.accuracy, feed_dict={test_cnn2._xs: test_x[776:1552, :],
            # #                                                      test_cnn2._ys: test_y[776:1552, :]})
            # # accuracy_3 = sess.run(test_cnn2.accuracy, feed_dict={test_cnn2._xs: test_x[1552:2328, :],
            # #                                                      test_cnn2._ys: test_y[1552:2328, :]})
            # # accuracy_4 = sess.run(test_cnn2.accuracy, feed_dict={test_cnn2._xs: test_x[2328:3104, :],
            # #                                                      test_cnn2._ys: test_y[2328:3104, :]})
            # # acry = (accuracy_1 + accuracy_2 + accuracy_3 + accuracy_4) / 4
            # if check_point_monitor == 'var':
            #     if var < early_stop_var:
            #         early_stop_numb = 0
            #         early_stop_var = var
            #         print('check point')
            #         saver.save(sess, PATH + 'model.ckpt')
            #     else:
            #         early_stop_numb = early_stop_numb + 1
            #         if early_stop_numb == early_stop:
            #             i = epoch + 1
            #             end_time_sum = time.time()
            #             sum_time = end_time_sum - start_time_sum
            #             need_h = sum_time // 3600
            #             need_m = (sum_time % 3600) // 60
            #             need_s = (sum_time % 3600) % 60
            #             f.write('early stop , used time : %d h %d m %d s\n' % (need_h, need_m, need_s))
            #             f.write('best var is : %f'%early_stop_var)
            #             print('early stop , used time : %d h %d m %d s' % (need_h, need_m, need_s))
            #             print('best var is :' , early_stop_var)
            # elif check_point_monitor == 'acc':
            #     if acry > early_stop_acc:
            #         early_stop_numb = 0
            #         early_stop_acc = acry
            #         print('check point')
            #         saver.save(sess, PATH + 'model.ckpt')
            #     else:
            #         early_stop_numb = early_stop_numb + 1
            #         if early_stop_numb == early_stop:
            #             i = epoch + 1
            #             end_time_sum = time.time()
            #             sum_time = end_time_sum - start_time_sum
            #             need_h = sum_time // 3600
            #             need_m = (sum_time % 3600) // 60
            #             need_s = (sum_time % 3600) % 60
            #             f.write('early stop , used time : %d h %d m %d s\n' % (need_h, need_m, need_s))
            #             f.write('best acc is : %f'%early_stop_acc)
            #             print('early stop , used time : %d h %d m %d s' % (need_h, need_m, need_s))
            #             print('best acc is :' , early_stop_acc)
            # # if var < early_stop_var :
            # #     early_stop_numb = 0
            # #     early_stop_var = var
            # #     print('check point')
            # #     saver.save(sess, PATH + 'model.ckpt')
            # # else :
            # #     early_stop_numb = early_stop_numb + 1
            # #     if early_stop_numb == early_stop :
            # #         i = epoch + 1
            # #         end_time_sum = time.time()
            # #         sum_time = end_time_sum - start_time_sum
            # #         need_h = sum_time // 3600
            # #         need_m = (sum_time % 3600) // 60
            # #         need_s = (sum_time % 3600) % 60
            # #         f.write('early stop , used time : %d h %d m %d s\n' % (need_h, need_m, need_s))
            # #         print('early stop , used time : %d h %d m %d s' % (need_h, need_m, need_s))
            if plot == True:
                k = [i - epoch_to_show, i]
                loss_new = loss
                loss1 = [loss_old, loss_new]
                loss_old = loss_new
                var_new = var
                var1 = [var_old, var_new]
                var_old = var_new
                if i > 0 and i < (epoch + 1):
                    lines = ax.plot(k, loss1, '-r', linestyle='-', lw=0.5, label='train loss')  # train
                    var_lines = ax.plot(k, var1, '-g', linestyle='-', lw=0.5, label='test loss')  # test
                    if i == epoch_to_show:
                        plt.legend(loc=1, ncol=1)
                    if loss_new <= 1 and var_new <= 1:
                        if save_fig == 1:
                            plt.savefig(PATH + '1.pdf')
                            plt.ylim(0, 1)  # 设置y轴刻度的范围，从0~20
                            save_fig = 2
                    plt.xlabel('epoch')
                    plt.ylabel('loss')
                    plt.title('loss-epoch curve')  # 对图形整体增加文本标签
                    plt.pause(0.001)
            if i == 0:
                lr_var = var
                lr_acc = acry
                early_stop_var = var
                early_stop_acc = acry
                print('accuracy is', acry, 'after', i, 'epoch')
            else:
                show_time(i,real_epoch,need_t, acry, var, train_acc, loss, l_r=learn_rate)
        # if i % epoch_to_show == 0:
        #     end_time = time.time()
        #     epoch_t = end_time - start_time
        #     start_time = end_time
        #     need_t = epoch_t * (show_time - i//epoch_to_show)
        #     need_t = round(need_t)
        #
        #     loss = sess.run(train_cnn2._cost, feed_dict={train_cnn2._xs: batch_xs, train_cnn2._ys: batch_ys})
        #     var = sess.run(test_cnn2._cost, feed_dict={test_cnn2._xs: test_x, test_cnn2._ys: test_y})
        #     if var < early_stop_var :
        #         early_stop_numb = 0
        #         early_stop_var = var
        #         saver.save(sess, PATH + 'model.ckpt')
        #     else :
        #         early_stop_numb = early_stop_numb + 1
        #         if early_stop_numb == early_stop :
        #             i = epoch + 1
        #             end_time_sum = time.time()
        #             sum_time = end_time_sum - start_time_sum
        #             need_h = sum_time // 3600
        #             need_m = (sum_time % 3600) // 60
        #             need_s = (sum_time % 3600) % 60
        #             print('early stop , used time : %d h %d m %d s' % (need_h, need_m, need_s))
        #     k = [i - epoch_to_show, i]
        #     loss_new = loss
        #     loss1 = [loss_old, loss_new]
        #     loss_old = loss_new
        #     var_new = var
        #     var1 = [var_old, var_new]
        #     var_old = var_new
        #     if i > 0 and i < (epoch + 1):
        #         lines = ax.plot(k, loss1, '-r', linestyle='-',lw=0.5, label='train loss')  # train
        #         var_lines = ax.plot(k, var1, '-g', linestyle='-',lw=0.5, label='test loss')  # test
        #         if i == epoch_to_show:
        #             plt.legend(loc=1, ncol=1)
        #         if loss_new <= 1 and var_new <= 1:
        #             if save_fig == 1:
        #                 plt.savefig(PATH + '1.pdf')
        #                 plt.ylim(0, 1)  # 设置y轴刻度的范围，从0~20
        #                 save_fig = 2
        #         plt.xlabel('epoch')
        #         plt.ylabel('loss')
        #         plt.title('loss-epoch curve')  #对图形整体增加文本标签
        #         plt.pause(0.001)
        #     if i == 0:
        #         early_stop_var = var
        #         acry = sess.run(test_cnn2.accuracy,feed_dict={test_cnn2._xs: test_x, test_cnn2._ys: test_y})
        #         f.write('%d'%i +' '+ '%f'%acry + ' %f'%loss + ' %f'%var + '\n')
        #         print('accuracy is',acry,'after',i,'epoch')
        #     else :
        #         if need_t >= 3600:
        #             need_h = need_t // 3600
        #             need_m = (need_t % 3600) // 60
        #             need_s = (need_t % 3600) % 60
        #             acry = sess.run(test_cnn2.accuracy, feed_dict={test_cnn2._xs: test_x, test_cnn2._ys: test_y})
        #             f.write('%d' % i + ' ' + '%f' % acry + ' %f'%loss + ' %f'%var + '\n')
        #             print('accuracy is', acry, 'after', i, 'epoch','need time : %d h %d m %d s' % (need_h, need_m, need_s))
        #             # print('accuracy is',sess.run(test_cnn2.accuracy,feed_dict={test_cnn2._xs: test_x, test_cnn2._ys: test_y}),'after',i,'epoch',
        #             #       'need time : %d h %d m %d s' % (need_h, need_m, need_s))
        #         elif need_t >= 60 and need_t < 3600:
        #             need_m = need_t // 60
        #             need_s = need_t % 60
        #             acry = sess.run(test_cnn2.accuracy, feed_dict={test_cnn2._xs: test_x, test_cnn2._ys: test_y})
        #             f.write('%d' % i + ' ' + '%f' % acry + ' %f'%loss + ' %f'%var + '\n')
        #             print('accuracy is', acry, 'after', i, 'epoch','need time : 0 h %d m %d s' % (need_m, need_s))
        #         else:
        #             need_s = need_t % 60
        #             acry = sess.run(test_cnn2.accuracy, feed_dict={test_cnn2._xs: test_x, test_cnn2._ys: test_y})
        #             f.write('%d' % i + ' ' + '%f' % acry + ' %f'%loss + ' %f'%var + '\n')
        #             print('accuracy is', acry, 'after', i, 'epoch','need time : 0 h 0 m %d s' % (need_s))
        sess.run(train_fold.train_op_Adam, feed_dict={train_fold._xs: batch_xs, train_fold._ys: batch_ys, train_fold._lr: learn_rate})
        i = i + 1
    # builder = tf.saved_model.builder.SavedModelBuilder("./result/2/")
    # builder.add_meta_graph_and_variables(sess, ['tag_string'])
    # builder.save()
    # saver.save(sess,"./result/model/model.ckpt")
    f.close()
    if plot == True:
        plt.savefig(PATH + '2.pdf')
        plt.close()
    # plt.show(ax)
    return