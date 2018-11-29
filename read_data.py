import os
import numpy as np
import re


def read_from_file(f):
    b1,b2 = [], []
    lines = f.readlines()
    for line in lines:
        w = line.split()
        if w[0] == 'b':
            b1.append([float(w[2])])
            b2.append([float(w[3])])
    return np.concatenate((b1,b2),axis=1).mean(axis=1),b1,b2


# return mean2d(a),mean2d(b),mean2d(g),mean2d(d),mean2d(t),c
# return a,b,g,d,t,c
# float(sum(c)/len(c))


def main():
    users = os.listdir(data_path)
    # print(users)
    for user_folder in users:
        sessions = os.listdir(os.path.join(data_path, user_folder))

        for session in sessions:
            filepath = os.path.join(data_path,user_folder,session)
            print(filepath)
            logfile = open(filepath + '/logfile', 'r')
            lines = logfile.readlines()
            logfile.close()

            dest_dir = os.path.join(filepath, 'beta')

            try:
                os.makedirs(dest_dir)
            except OSError:
                pass  # already exists

            for line in lines:
                line_item = re.split('\s+', line)
                eeg_filename = "robot_"+line_item[0]   # EEG file for each round
                beta_file = open(dest_dir + '/betas'+str(line_item[0]), 'w')
                label_file = open(dest_dir + '/result'+str(line_item[0]), 'w')

                result = None     # Result for the respective respective round.
                if line_item[4] == "1":
                    result = 1
                elif line_item[4] == "-1":
                    result = 0

                print(eeg_filename,result)
                f = open(filepath + '/' + eeg_filename, 'r')
                beta_mean, beta_AF7, beta_AF8= read_from_file(f)
                label_file.write(str(result))
                # beta_file.write("M"+' '+"B1"+' '+"B2")
                # beta_file.write('\n')
                for m,b1,b2 in zip(beta_mean,beta_AF7,beta_AF8):
                    beta_file.write(str(m)+' '+str(b1)+' '+str(b2))
                    beta_file.write('\n')


def preprocess_data(window_size):
    """
    Function to split data into windows
    :param window_size: input the number of time steps to consider
    :return:
    """
    users = os.listdir(data_path)
    # print(users)
    for user_folder in users:
        sessions = os.listdir(os.path.join(data_path, user_folder))
        # Consider window size 15 whis is 1.5s window without overlap...
        for session in sessions:
            filepath = os.path.join(data_path, user_folder, session,'beta')
            # print(filepath)
            files = os.listdir(filepath)
            # print(files)
            for file in files:
                if "beta" in file:
                    print(file)
                    f = open(filepath + '/' + file, 'r')
                    lines = f.readlines()
                    f.close()
                    # print(lines)
                    beta_timestamp = []
                    num = re.findall(r'\d+', file)

                    beta_file = open(filepath + '/beta' + '_' + str(num[0]), 'w')

                    for line in lines:
                        line_item = re.split('\s+', line)
                        print(line_item)
                        count = 0

                        if len(beta_timestamp) < 15:
                            beta_timestamp.append(line_item[0])
                            # beta_file.write(str(line_item[0]) + ' ')
                        else:
                            beta_timestamp = [line_item[0]]
                            # beta_file.write(str(line_item[0]) + ' ')

                        if len(beta_timestamp) == 15:
                            for i in range(len(beta_timestamp)):
                                beta_file.write(str(beta_timestamp[i])+' ')
                                count += 1
                            beta_file.write('\n')

                    if len(beta_timestamp) < 15:
                        betafile = open(filepath + '/beta' + '_' + str(num[0]), 'a+')
                        count = 0
                        for i in range(len(beta_timestamp)):
                            betafile.write(str(beta_timestamp[i])+' ')
                            count += 1

                        while count < 15:
                            betafile.write(str(0)+' ')
                            count += 1
                        betafile.write('\n')


if __name__ == '__main__':
    data_path = "/media/akilesh/data/sequencelearning_eeg"
    #main()
    preprocess_data(15)
