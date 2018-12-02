import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout, LSTM
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from matplotlib import pyplot
import os, re


class PerformancePrediction:
    def __init__(self):
        #self.data_path = "/media/akilesh/data/UCI HAR Dataset/"
        self.data_path = "/media/akilesh/data/sequencelearning_eeg"
        self.model = Sequential()
        self.data = list()
        self.label = list()
        self.num_samples = 0
        self.trainX, self.trainY, self.testX, self.testY = None, None, None, None

    # load a single file as a numpy array
    def load_file(self, filepath):
        dataframe = read_csv(filepath, header=None, delim_whitespace=True)
        # print(dataframe.values)
        return dataframe.values

    def load_dataset(self):
        print("Reading data...")
        print(self.data_path)
        users = os.listdir(self.data_path)
        # print(users)
        for user_folder in users:
            #print(user_folder)
            sessions = os.listdir(os.path.join(self.data_path, user_folder))
            # Consider window size 15 whis is .26s window without overlap...
            for session in sessions:
                filepath = os.path.join(self.data_path, user_folder, session,'beta')
                files = os.listdir(filepath)
                loaded = list()
                for file in files:

                    if str(file).startswith('beta_'):
                        # print(file)
                        f_beta = os.path.join(filepath, file) # Location of the beta file
                        loaded = self.load_file(f_beta)
                        self.num_samples = len(loaded)
                        for num in range(self.num_samples):
                            self.data.append(loaded[num])

                        num_samp = re.findall(r'\d+', file)

                        resfile = 'result'+ str(num_samp[0])
                        f_label = os.path.join(filepath, resfile)  # Location of the result file
                        loaded_label = self.load_file(f_label)
                        #print(resfile)
                        for num in range(self.num_samples):
                            self.label.append(loaded_label[0][0])


                # stack group so that features are the 3rd dimension
                # self.data = dstack(self.data)
        self.data = np.asarray(self.data)
        self.data = self.data.reshape((self.data.shape[0],self.data.shape[1],1))
        self.label = np.asarray(self.label)
        # print(self.data, self.label)
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(self.data, self.label, train_size= 0.8,
                                                                            shuffle=False)
        print("Train and Test Data Loaded...")

    def define_model(self):
        # n_timesteps, n_features, n_outputs = self.trainX[0].shape[1], self.trainX.shape[2], self.trainY.shape[1]
        n_timesteps, n_outputs = self.trainX[0].shape[0], 1

        self.model.add(LSTM(100, input_shape=(n_timesteps, 1)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(n_outputs, activation='softmax'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit and evaluate a model
    def evaluate_model(self):
        verbose, epochs, batch_size = 0, 15, 64
        print(self.model.summary())

        # fit network
        self.model.fit(self.trainX, self.trainY, epochs=epochs, batch_size=batch_size, verbose=verbose)
        # evaluate model
        _, accuracy = self.model.evaluate(self.testX, self.testY, batch_size=batch_size, verbose=0)
        return accuracy

    # summarize scores
    def summarize_results(self, scores):
        print(scores)
        m, s = np.mean(scores), np.std(scores)
        print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

    # run an experiment
    def run_experiment(self, repeats=10):
        # load data
        self.load_dataset()
        print(self.data.shape,'\n')
        print(self.trainX[0].shape, self.trainY.shape)

        # i =0
        # for items in self.trainX:
        #     print(items)
        #     print(self.trainY[i])
        #     i += 1
        #     break

        # Define an LSTM model
        self.define_model()

        # repeat experiment
        scores = list()
        for r in range(repeats):
            score = self.evaluate_model()

            score = score * 100.0
            print('>#%d: %.3f' % (r + 1, score))
            scores.append(score)
        # summarize results
        self.summarize_results(scores)

def main():
    rnn = PerformancePrediction()
    rnn.run_experiment(10)


if __name__ == '__main__':
    main()