import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import sys
import glob
import serial as serial
from sklearn.svm import SVC
import threading
from threading import Lock
import struct
import time

if sys.platform.startswith('win'):
    ports = ['COM%s' % (i + 1) for i in range(256)]
elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
    # this excludes your current terminal "/dev/tty"
    ports = glob.glob('/dev/tty[A-Za-z]*')
elif sys.platform.startswith('darwin'):
    ports = glob.glob('/dev/tty.*')
else:
    raise EnvironmentError('Unsupported platform')

# Had an issue where the port number would consistently change each time the program ran
# The fix for this was the code below - searches all ports to find the one starting with usbmodem and then selects that
result = []
for port in ports:
    try:
        s = serial.Serial(port)
        s.close()
        result.append(port)
    except (OSError, serial.SerialException):
        pass

for port in result:
    if "usbmodem" in port:
        break

if "usbmodem" not in port:
    raise ValueError("Arduino NOT Found!")

ser = serial.Serial(port, baudrate=9600, timeout=1)

feature_per_axis = 2  # number of features/axis - 2 (a and a_dot)
axis = 3  # number of axis - 3 (xyz)
nr_boundaries = 5
patterns = 3  # number of control modes user wishes to have
repeats = 5  # number of repititions they must complete during training of the same pattern
numCounters = 5  # number of counter examples

n_features = feature_per_axis * axis * nr_boundaries  # size of the feature vector

trained_patterns = []  # patterns * repeats sized list
counter_examples = []  # numCounters sized list

good_trained_patterns = []  # true trained pattern
good_counter_examples = []  # true trained pattern

exitBoolean = False
lock = threading.Lock()  # used for the thread, allows to wait for ENTER key press to start and stop listening during training

counter = 1  # counter that labels the data as it enters the file

stopped_arduino = True  # at the start, we do not receive data .. we wait to tell Arduino when to start

RELOAD_GOOD_PARAMETERS = False  # ONLY CHANGE IF YOU WANT TO RELOAD THE PARAMETERS FROM THE GOOD SET ONTO THE CHIP


# This is the thread, simultaneously runs along side the training below and listens for an ENTER keypress
def inputQuit():
    global exitBoolean
    global stopped_arduino

    userInput = input("Press ENTER when finished\n")

    if userInput == '':
        with lock:
            # exitBoolean = True

            # print('Sending to Arduino START_WAIT')
            ser.write(b'START_WAIT')

            # print("****" + str(ser.in_waiting) + "****")
            ser.reset_input_buffer()
        # print("****" + str(ser.in_waiting) + "****")

        # print("CHange the State of 'stopped_arduino' flag to TRUE")
        # stopped_arduino = True

        return

    # the main training function


print("STARTING TRAINING:")

# loops through the number of patterns and repeats
for i in range(patterns):
    print("Training Pattern " + str(i + 1) + ":")
    for j in range(repeats):
        print("Repeat movement " + str(repeats - j) + " more times to train pattern.")
        filename = "pattern" + str(i + 1) + "_" + str(j + 1) + ".csv"

        # print('State of stopped_arduino flag is: ' + str(stopped_arduino))
        if (RELOAD_GOOD_PARAMETERS == False):
            with open(filename, "w+") as f:
                userInput1 = input("Press ENTER when ready")

                if (userInput1 == ''):
                    quit_thread = threading.Thread(target=inputQuit, args=[])
                    quit_thread.start()

                    checkFlag = False

                    if stopped_arduino:
                        # print("Sending to Arduino END_WAIT ... changing 'stopped_arduino' to False")
                        ser.write(b'END_WAIT')
                        stopped_arduino = False

                    # ard_counter = 1
                    # millis = int(round(time.time() * 1000))

                    while (checkFlag == False):

                        try:
                            if ser.in_waiting == 0:
                                continue

                            line = ser.readline().strip().decode('ascii')  # reads the output of the Arduino
                        except:
                            continue

                        # print(line)
                        # communication between Python and Arduino, this is how we know when to start and stop
                        if line.startswith('SYS:START_WAIT'):
                            checkFlag = True
                            stopped_arduino = True
                            exitBoolean = True
                            break

                        if line.startswith('SYS:'):
                            continue

                        if len(line) > 0:
                            # edits the output from the chip with the count
                            try:
                                line = str(counter) + line[line.index(","):]
                                print(line)
                                counter = counter + 1
                            except:
                                continue

                        f.write(str(line) + "\n")  # writes

                        with lock:
                            checkFlag = exitBoolean

                    counter = 1

        exitBoolean = False
        # creates a vector from the file created by writing the outputs of the chip
        trained_patterns += [pd.read_csv(filename,
                                         names=['count', 'time_ms', 'a_x', 'a_dot_x', 'a_ddot_x', 'a_y', 'a_dot_y',
                                                'a_ddot_y', 'a_z', 'a_dot_z', 'a_ddot_z'])]

        if (RELOAD_GOOD_PARAMETERS == True):
            good_trained_patterns += [pd.read_csv('good' + filename,
                                                  names=['count', 'time_ms', 'a_x', 'a_dot_x', 'a_ddot_x', 'a_y',
                                                         'a_dot_y', 'a_ddot_y', 'a_z', 'a_dot_z', 'a_ddot_z'])]

    print("Training Pattern " + str(i + 1) + " complete!")

exitBoolean = False

# Almost copy and paste of the above code, less iterations because this is only for counter-examples,
# but same communication protocol and writing
print("STARTING COUNTER EXAMPLE TRAINING:")
for i in range(numCounters):
    print("Training Counter-Example: " + str(i + 1) + " out of " + str(numCounters))
    filename = "counter" + str(i + 1) + ".csv"
    if (RELOAD_GOOD_PARAMETERS == False):
        with open(filename, "w+") as f:
            userInput1 = input("Press ENTER when ready")
            ser.reset_input_buffer()
            if (userInput1 == ''):
                quit_thread = threading.Thread(target=inputQuit, args=[])
                quit_thread.start()
                checkFlag = False

                if stopped_arduino:
                    ser.write(b'END_WAIT')
                    stopped_arduino = False

                while (checkFlag == False):

                    try:
                        if ser.in_waiting == 0:
                            continue

                        line = ser.readline().strip().decode('ascii')
                    except:
                        continue

                    # print(line)

                    if line.startswith('SYS:START_WAIT'):
                        checkFlag = True
                        stopped_arduino = True
                        exitBoolean = True
                        break

                    if line.startswith('SYS:'):
                        continue

                    if len(line) > 0:

                        try:
                            line = str(counter) + line[line.index(","):]
                            print(line)
                            counter = counter + 1
                        except:
                            continue

                    f.write(str(line) + "\n")

                    with lock:
                        checkFlag = exitBoolean

                counter = 1

    exitBoolean = False
    counter_examples += [pd.read_csv(filename,
                                     names=['count', 'time_ms', 'a_x', 'a_dot_x', 'a_ddot_x', 'a_y', 'a_dot_y',
                                            'a_ddot_y', 'a_z', 'a_dot_z', 'a_ddot_z'], skiprows=2)]

    if (RELOAD_GOOD_PARAMETERS == True):
        good_counter_examples += [pd.read_csv('good' + filename,
                                              names=['count', 'time_ms', 'a_x', 'a_dot_x', 'a_ddot_x', 'a_y', 'a_dot_y',
                                                     'a_ddot_y', 'a_z', 'a_dot_z', 'a_ddot_z'], skiprows=2)]

print("TRAINING COMPLETE")

ser.write(b'END_TRAINING')

time.sleep(3)

x_plot = 'count'

# creates a feature vector and labels vector for training the SVM and then classification
feature_vectors = np.zeros((repeats * (patterns + 1), n_features))
good_feature_vectors = np.zeros((repeats * (patterns + 1), n_features))
labels_vector = np.zeros((repeats * (patterns + 1),), dtype=int)
for i in range(patterns + 1):
    labels_vector[i * repeats:(
                                          i + 1) * repeats] = i  # fills the labels_vector with the pattern number (first set of repeats is pattern 0, etc.)

for didx, dataset in enumerate(trained_patterns + counter_examples):  # change back to name w/o good_
    curr_data = dataset.copy()
    curr_data[x_plot] -= curr_data[x_plot].iloc[0]

    boundaries = np.linspace(0, len(curr_data), nr_boundaries + 1)  # 6 means partition time into 5 groups
    for idx in range(len(boundaries) - 1):
        data_chunk = curr_data[int(boundaries[idx]):int(
            boundaries[idx + 1])]  # partitions the file in a predefined amount of boundaries for featurization

        # used features are acceleration in x,y,z and jerk in x,y,z (no longer us a double dot A.K.A jounce)
        mean_ax = np.mean(data_chunk['a_x'])  # np.mean takes derivative
        mean_a_dot_x = np.mean(data_chunk['a_dot_x'])
        mean_a_ddot_x = np.mean(data_chunk['a_ddot_x'])

        mean_ay = np.mean(data_chunk['a_y'])
        mean_a_dot_y = np.mean(data_chunk['a_dot_y'])
        mean_a_ddot_y = np.mean(data_chunk['a_ddot_y'])

        mean_az = np.mean(data_chunk['a_z'])
        mean_a_dot_z = np.mean(data_chunk['a_dot_z'])
        mean_a_ddot_z = np.mean(data_chunk['a_ddot_z'])

        # populates the feature_vectors with the above features
        feature_vectors[didx, idx * feature_per_axis * axis:(idx + 1) * feature_per_axis * axis] = [
            mean_ax, mean_a_dot_x,  # mean_a_ddot_x,
            mean_ay, mean_a_dot_y,  # mean_a_ddot_y,
            mean_az, mean_a_dot_z]  # mean_a_ddot_z]

# labels_vector[labels_vector != 3] = 1
# labels_vector[labels_vector == 3] = 0

# same as above, but used to create 'the best' training set I've had so far - this is for testing

if (RELOAD_GOOD_PARAMETERS == True):
    for didx, dataset in enumerate(good_trained_patterns + good_counter_examples):  # change back to name w/o good_
        curr_data = dataset.copy()
        curr_data[x_plot] -= curr_data[x_plot].iloc[0]

        boundaries = np.linspace(0, len(curr_data), nr_boundaries + 1)  # 6 means partition time into 5 groups
        for idx in range(len(boundaries) - 1):
            data_chunk = curr_data[int(boundaries[idx]):int(boundaries[idx + 1])]

            mean_ax = np.mean(data_chunk['a_x'])
            mean_a_dot_x = np.mean(data_chunk['a_dot_x'])
            mean_a_ddot_x = np.mean(data_chunk['a_ddot_x'])

            mean_ay = np.mean(data_chunk['a_y'])
            mean_a_dot_y = np.mean(data_chunk['a_dot_y'])
            mean_a_ddot_y = np.mean(data_chunk['a_ddot_y'])

            mean_az = np.mean(data_chunk['a_z'])
            mean_a_dot_z = np.mean(data_chunk['a_dot_z'])
            mean_a_ddot_z = np.mean(data_chunk['a_ddot_z'])

            good_feature_vectors[didx, idx * feature_per_axis * axis:(idx + 1) * feature_per_axis * axis] = [
                mean_ax, mean_a_dot_x,  # mean_a_ddot_x,
                mean_ay, mean_a_dot_y,  # mean_a_ddot_y,
                mean_az, mean_a_dot_z]  # mean_a_ddot_z]

# these were used for normalizing the data - not used anymore
min_feat_vals = feature_vectors.min(axis=0)
peak_to_peak_vals = np.ptp(feature_vectors, axis=0)

if (RELOAD_GOOD_PARAMETERS == True):
    min_feat_vals = good_feature_vectors.min(axis=0)
    peak_to_peak_vals = np.ptp(good_feature_vectors, axis=0)

# feature_vectors = (feature_vectors - min_feat_vals)/peak_to_peak_vals

# Creation of the SVM - clf.fit generates the relevant parameters for classification
clf = SVC(kernel='rbf', C=0.001)
clf.fit(feature_vectors, labels_vector)
if (RELOAD_GOOD_PARAMETERS == False):
    clf.fit(feature_vectors, labels_vector)
else:
    clf.fit(good_feature_vectors, labels_vector)
# clf.predict(trained_patterns + counter_examples)
# print(good_feature_vectors.shape, feature_vectors.shape, labels_vector.shape)
# print('Accuracy is: ', clf.score(feature_vectors, labels_vector))

# for C in np.logspace(-3,3,7):
# 	clf = SVC(kernel='rbf', C = C)
# 	clf.fit(good_feature_vectors, labels_vector)
# 	#clf.predict(trained_patterns + counter_examples)
# 	print(good_feature_vectors.shape, feature_vectors.shape, labels_vector.shape)
# 	print('Accuracy is: ', clf.score(feature_vectors, labels_vector))

# 	clf = SVC(kernel='linear', C = C)
# 	clf.fit(good_feature_vectors, labels_vector)
# 	#clf.predict(trained_patterns + counter_examples)
# 	print('Accuracy is: ', clf.score(feature_vectors, labels_vector))

# 	for order in [1,2,3,4,5]:
# 		clf = SVC(kernel='poly', C = C, degree=order)
# 		clf.fit(good_feature_vectors, labels_vector)
# 		#clf.predict(trained_patterns + counter_examples)
# 		print('Accuracy is: ', clf.score(feature_vectors, labels_vector))

# 	clf = SVC(kernel='sigmoid', C = C)
# 	clf.fit(good_feature_vectors, labels_vector)
# 	#clf.predict(trained_patterns + counter_examples)
# 	print('Accuracy is: ', clf.score(feature_vectors, labels_vector))

# 	print('C = ',C)
# 	print('****************')


# Writes all relevant parameters and vectors to the chip - uses struck.pack and sends majority of data over as floats, some ints
print("++++++++ Writing ++++++++++")

ser.write(struct.pack('<i', clf.support_vectors_.shape[0]))

for i in range(0, clf.support_vectors_.shape[1]):
    ser.write(struct.pack('<f', min_feat_vals[i]))
    print("**** " + " The " + str(i) + "th entry that was written is " + str(min_feat_vals[i]))
ser.flush()

for i in range(0, clf.support_vectors_.shape[1]):
    ser.write(struct.pack('<f', peak_to_peak_vals[i]))
    print("**** " + " The " + str(i) + "th entry that was written is " + str(peak_to_peak_vals[i]))
ser.flush()

for i in range(0, clf.intercept_.shape[0]):
    ser.write(struct.pack('<f', clf.intercept_[i]))
    print("**** " + " The " + str(i) + "th entry that was written is " + str(clf.intercept_[i]))
ser.flush()

for r in range(0, clf.support_vectors_.shape[0]):
    for c in range(0, clf.support_vectors_.shape[1]):
        ser.write(struct.pack('<f', clf.support_vectors_[r][c]))
        print("**** [" + str(r) + "][" + str(c) + "] = " + str(clf.support_vectors_[r][c]))
ser.flush()

for r in range(0, clf.dual_coef_.shape[0]):
    for c in range(0, clf.dual_coef_.shape[1]):
        ser.write(struct.pack('<f', clf.dual_coef_[r][c]))
        print("**** [" + str(r) + "][" + str(c) + "] = " + str(clf.dual_coef_[r][c]))
ser.flush()

for r in range(0, clf.n_support_.shape[0]):
    ser.write(struct.pack('<i', clf.n_support_[r]))
    print("**** [" + str(r) + "] = " + str(clf.n_support_[r]))

# print("FEATURE VECTOR\n")
# print(feature_vectors)
# print("LABELS VECTORS\n")
# print(labels_vector)
# print("INTERCEPTS\n")
# print(clf.intercept_)
# print("SUPPORT VECTORS\n")
# print(clf.support_vectors_)
# print("DUAL COEFS\n")
# print(clf.dual_coef_)
# print("N SUPPORT\n")
# print(clf.n_support_)

# print(feature_vectors.shape)
# print(labels_vector.shape)
# print(min_feat_vals.shape)
# print(peak_to_peak_vals.shape)
# print(clf.intercept_.shape)
# print(clf.support_vectors_.shape)
# print(clf.dual_coef_.shape)
# print(clf.n_support_.shape)

# Final communication protocol so signify end of training

print("TRANSMISSION COMPLETE ... wait for Arduino's Ack")

time.sleep(5)

waitForAck = True

while waitForAck:
    try:
        if ser.in_waiting == 0:
            continue

        line = ser.readline().strip().decode('ascii')

    except:
        continue

    print(line)

    if line.startswith('SYS:TRAINING_END_ACK'):
        waitForAck = False
        break

time.sleep(10)

ser.close()

# (6,)
# (20, 45)
# (3, 20)

# test that SVM works - test on half data and other half
# MIGHT HAVE TO SEND TO ARDUINO CLASS INFORMATION - SO IT KNOWS WHICH PREDICTION VALUE CORRESPONDS TO WHICH ACTION

# was used for plotting features - test
if False:
    fig, axes = plt.subplots(9, 1, sharex=True)
    axes[0].plot(raw_data[x_plot], raw_data['a_x'])
    axes[0].set_ylabel('$a_x$')
    axes[1].plot(raw_data[x_plot], raw_data['a_y'])
    axes[1].set_ylabel('$a_y$')
    axes[2].plot(raw_data[x_plot], raw_data['a_z'])
    axes[2].set_ylabel('$a_z$')

    axes[3].plot(raw_data[x_plot], raw_data['a_dot_x'])
    axes[3].set_ylabel('$\dot{a}_x$')
    axes[4].plot(raw_data[x_plot], raw_data['a_dot_y'])
    axes[4].set_ylabel('$\dot{a}_y$')
    axes[5].plot(raw_data[x_plot], raw_data['a_dot_z'])
    axes[5].set_ylabel('$\dot{a}_z$')

    axes[6].plot(raw_data[x_plot], raw_data['a_ddot_x'])
    axes[6].set_ylabel('$\ddot{a}_x$')
    axes[7].plot(raw_data[x_plot], raw_data['a_ddot_y'])
    axes[7].set_ylabel('$\ddot{a}_y$')
    axes[8].plot(raw_data[x_plot], raw_data['a_ddot_z'])
    axes[8].set_ylabel('$\ddot{a}_z$')

    plt.show()