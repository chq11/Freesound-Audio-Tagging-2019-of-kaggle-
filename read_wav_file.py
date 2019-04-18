import wave
import numpy as np
import pylab as plt

f = open('./data/freesound_audio_tagging_2019/test_list_debug.csv', 'r')
lines = f.readlines()
line = lines[0]
d_label = line.strip().split(',')[1:]
D_label = np.zeros([1, 80])
for i in range(80):
    D_label[0, i] = np.int(d_label[i])

print(D_label)
wav_p = line.strip().split(',')[0]
#打开wav文件 ，open返回一个的是一个Wave_read类的实例，通过调用它的方法读取WAV文件的格式和数据。
f = wave.open(wav_p,"rb")
#读取格式信息
#一次性返回所有的WAV文件的格式信息，它返回的是一个组元(tuple)：声道数, 量化位数（byte单位）, 采
#样频率, 采样点数, 压缩类型, 压缩类型的描述。wave模块只支持非压缩的数据，因此可以忽略最后两个信息
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
# print(nchannels)
#读取波形数据
#读取声音数据，传递一个参数指定需要读取的长度（以取样点为单位）
str_data = f.readframes(nframes)
f.close()
#将波形数据转换成数组
#需要根据声道数和量化单位，将读取的二进制数据转换为一个可以计算的数组
wave_data = np.fromstring(str_data,dtype = np.short)
#将wave_data数组改为2列，行数自动匹配。在修改shape的属性时，需使得数组的总长度不变。
# wave_data.shape = -1,2
#转置数据
wave_data = wave_data.T
wave_shape = wave_data.shape
new_wave_d = np.zeros([1, 600000])
if wave_shape[0] >= 600000:
    new_wave_d[0, :] = wave_data[int((wave_shape[0]-600000)/2):int((wave_shape[0]-600000)/2)+600000]
else:
    new_wave_d[0, int((600000-wave_shape[0])/2):int((600000-wave_shape[0])/2)+wave_shape[0]] = wave_data

new_wave_d = new_wave_d[0, 0::int(600000 / 6000)]
print(new_wave_d.shape)
print(new_wave_d[3000])
print(wave_data[int((wave_shape[0])/2)])
#通过取样点数和取样频率计算出每个取样的时间。
time=np.arange(0,6000)/(600000/6000)
# time=np.arange(0,nframes)/framerate
# print(time.shape)
#print(params)
plt.figure(1)
plt.subplot(1,1,1)
#time 也是一个数组，与wave_data[0]或wave_data[1]配对形成系列点坐标
plt.plot(time,new_wave_d)
# plt.subplot(2,1,2)
# plt.plot(time,wave_data[1],c="r")
plt.xlabel("time")
plt.show()



