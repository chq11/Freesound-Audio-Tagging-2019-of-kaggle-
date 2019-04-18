import random
DataPath = './data/freesound_audio_tagging_2019/'
TrainCurated = 'train_curated'
TrainNoisy = 'train_noisy'
TrainList = 'train_list'
TestList = 'test_list'
TestRate = 0.1

SSN = 'sample_submission.csv'
l_f = open(DataPath+SSN, 'r')
label_lines_0 = l_f.readlines()[0]  #fname,Accelerating_and_revving_and_vroom,Accordion,Acoustic_guitar,Applause......
label_lines = label_lines_0.split(',')[1:]  #Accelerating_and_revving_and_vroom,Accordion,Acoustic_guitar,Applause......

def WriteLabel(save_t = ''):
    n_f = open(DataPath + save_t + '.csv', 'r')
    raw_line = n_f.readlines()

    w_f = open(DataPath + save_t + '_list.csv', 'w')

    for i in range(len(raw_line)):
        if i == 0:
            w_f.write(label_lines_0)
        else:
            c_l = raw_line[i].strip().split(',')  #['000b6cfb.wav','Motorcycle'] or ['001c054e.wav','"Raindrop','Trickle_and_dribble"']
            w_line = ''.join([DataPath, save_t, '/', c_l[0]])    #'/home/haiquan/deep_learning/fat/data/freesound_audio_tagging_2019/train_curated/000b6cfb.wav'
            if len(c_l) == 2: #the type like ['000b6cfb.wav','Motorcycle']
                for c_i in range(len(c_l) - 1):
                    for found_l in range(len(label_lines)):
                        if c_l[c_i + 1] == label_lines[found_l]:
                            w_line = ','.join([w_line, '1'])
                        else:
                            w_line = ','.join([w_line, '0'])
            else:             #the type like ['001c054e.wav','"Raindrop','Trickle_and_dribble"']
                c_l[1] = c_l[1].strip('"')
                c_l[-1] = c_l[-1].strip('"')
                # c_l is ['001c054e.wav','Raindrop','Trickle_and_dribble']
                for c_i in range(len(c_l)-1):
                    for found_l in range(len(label_lines)):
                        if c_i == 0:
                            if c_l[c_i+1] == label_lines[found_l]:
                                w_line = ','.join([w_line, '1'])
                            else:
                                w_line = ','.join([w_line, '0'])
                        else:
                            # print(w_line)
                            if c_i == 1 and found_l == 0:
                                w_line = str(w_line).split(',')
                                # print(w_line)
                            if c_l[c_i+1] == label_lines[found_l]:
                                w_line[found_l+1] = '1'
                w_line = ','.join(w_line)
            w_f.write(w_line + ',\n')
    w_f.close()
    n_f.close()

#write file
WriteLabel(TrainCurated)
WriteLabel(TrainNoisy)

#concate
w_f = open(DataPath + 'all_list.csv', 'w')
w_f.close()
w_f = open(DataPath + 'all_list.csv', 'a+')
add_line1 = open(DataPath + TrainCurated + '_list.csv', 'r').readlines()
add_line2 = open(DataPath + TrainNoisy + '_list.csv', 'r').readlines()
for i in range(len(add_line1)):
    w_f.write(add_line1[i])
for i in range(len(add_line2) - 1):
    w_f.write(add_line2[i + 1])
w_f.close()

#split train test data set
all_list_lines = open(DataPath + 'all_list.csv', 'r').readlines()[1:]
random.shuffle(all_list_lines)
TestSize = int(len(all_list_lines) * TestRate)
#TrainSize = len(all_list_lines) - TestSize
train_f = open(DataPath + TrainList + '.csv', 'w')
test_f = open(DataPath + TestList + '.csv', 'w')

for i in range(len(all_list_lines)):
    if i < TestSize:
        test_f.write(all_list_lines[i])
    else:
        train_f.write(all_list_lines[i])
train_f.close()
test_f.close()
