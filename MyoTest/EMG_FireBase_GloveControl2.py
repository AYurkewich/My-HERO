import matplotlib
from matplotlib import pyplot as plt
from collections import deque
from threading import Lock, Thread

import myo
import numpy as np
from firebase import firebase
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy import signal
from sympy import *
import sys


matplotlib.use('Qt4Agg')
firebase = firebase.FirebaseApplication('https://hero-d6297.firebaseio.com/')
firebase.put('', 'MyoCommand', 'open')


class EmgCollector(myo.DeviceListener):
  """
  Collects EMG data in a queue with *n* maximum number of elements.
  """

  def __init__(self, n):
    self.n = n
    self.lock = Lock()
    self.emg_data_queue = deque(maxlen=n)
    self.n = n
    self.lock = Lock()
    self.emg_data_queue = deque(maxlen=n)
    self.emg_data_filt_queue = deque(maxlen=n)
    self.ev_emg1 = 0.0
    self.ev_emg1_avg = 0.0
    self.ev_emg4 = 0.0
    self.ev_emg4_avg = 0.0
    self.ev_emg_avg = 0.0
    self.closeRobot = 0
    self.resetCounter = 0.0
    self.waitForRelax = 0
    self.waitForRelaxMax = 0
    self.relaxBeforeFLex = 0
    self.acceleration = None
    self.orientation = None

    self.ev_emg0_array = np.zeros(50)
    self.ev_emg1_array = np.zeros(50)
    self.ev_emg2_array = np.zeros(50)
    self.ev_emg3_array = np.zeros(50)
    self.ev_emg4_array = np.zeros(50)
    self.ev_emg5_array = np.zeros(50)
    self.ev_emg6_array = np.zeros(50)
    self.ev_emg7_array = np.zeros(50)
    self.sumEMG0_array = np.zeros(400)
    self.sumEMG1_array = np.zeros(400)
    self.sumEMG2_array = np.zeros(400)
    self.sumEMG3_array = np.zeros(400)
    self.sumEMG4_array = np.zeros(400)
    self.sumEMG5_array = np.zeros(400)
    self.sumEMG6_array = np.zeros(400)
    self.sumEMG7_array = np.zeros(400)
    self.delayCounterEMG = 200
    # self.ev_emg5_array = np.zeros(50)
    # self.ev_acc_array = np.zeros(50)
    # self.sumEMG_array = np.zeros(50)
    # self.sumACC_array = np.zeros(50)
    self.closed = 1
    self.sumEMGvalue = 0
    # self.delayCounter = 200
    # self.moveDelayCounter = 50
    # self.trigSumACC = 80
    self.trainCounter = 0
    self.emgElectrode = 0
    self.thresholdRelax = 0
    self.thresholdGrip = 0
    self.new = np.zeros(8)
    self.ev_emg_array = np.zeros(50)


  def on_orientation(self, event):
     with self.lock:
       #self.acceleration = event.acceleration
       self.orientation = event.orientation
       #print(abs(self.acceleration[0]) + abs(self.acceleration[1]) + abs(self.acceleration[2])) #[0] forward, [1] side, [2] up
       #print(abs(self.orientation[0]) + abs(self.orientation[1]) + abs(self.orientation[2])) #[0] forward, [1] side, [2] up

  def get_emg_data(self):
    with self.lock:
      return list(self.emg_data_queue)

  # myo.DeviceListener

  def on_connected(self, event):
    event.device.stream_emg(True)

  def on_emg(self, event):
    with self.lock:
        ##three factors to tune - relax threshold, grip threshold, emgElectrode -> multipleElectrodes?
      self.trainCounter = self.trainCounter+1
      if self.trainCounter == 1:
        print("RELAX your Hand and Arm on the table")
      if self.trainCounter == 1200:
        prctRelax = 30
        self.relax = np.array([np.percentile(self.sumEMG0_array, prctRelax), np.percentile(self.sumEMG1_array, prctRelax), np.percentile(self.sumEMG2_array, prctRelax), np.percentile(self.sumEMG3_array, prctRelax), np.percentile(self.sumEMG4_array, prctRelax), np.percentile(self.sumEMG5_array, prctRelax), np.percentile(self.sumEMG6_array, prctRelax), np.percentile(self.sumEMG7_array, prctRelax)])
        prctRelax2 = 90
        self.relaxHigh = np.array([np.percentile(self.sumEMG0_array, prctRelax2), np.percentile(self.sumEMG1_array, prctRelax2), np.percentile(self.sumEMG2_array, prctRelax2), np.percentile(self.sumEMG3_array, prctRelax2), np.percentile(self.sumEMG4_array, prctRelax2), np.percentile(self.sumEMG5_array, prctRelax2), np.percentile(self.sumEMG6_array, prctRelax2), np.percentile(self.sumEMG7_array, prctRelax2)])
        self.relaxStd = np.array([np.std(self.sumEMG0_array), np.std(self.sumEMG1_array), np.std(self.sumEMG2_array), np.std(self.sumEMG3_array), np.std(self.sumEMG4_array), np.std(self.sumEMG5_array), np.std(self.sumEMG6_array), np.std(self.sumEMG7_array)])

        # print("0:", np.percentile(self.sumEMG0_array, 25), np.percentile(self.sumEMG0_array, 95))
        # print("1:", np.percentile(self.sumEMG1_array, 25), np.percentile(self.sumEMG1_array, 95))
        # print("2:", np.percentile(self.sumEMG2_array, 25), np.percentile(self.sumEMG2_array, 95))
        # print("3:", np.percentile(self.sumEMG3_array, 25), np.percentile(self.sumEMG3_array, 95))
        # print("4:", np.percentile(self.sumEMG4_array, 25), np.percentile(self.sumEMG4_array, 95))
        # print("5:", np.percentile(self.sumEMG5_array, 25), np.percentile(self.sumEMG5_array, 95))
        # print("6:", np.percentile(self.sumEMG6_array, 25), np.percentile(self.sumEMG6_array, 95))
        # print("7:", np.percentile(self.sumEMG7_array, 25), np.percentile(self.sumEMG7_array, 95))

      if self.trainCounter == 1201:
        print("LIGHTLY grip the water bottle")

      if self.trainCounter == 2400:
        prctLight = 25
        self.light = np.array([np.percentile(self.sumEMG0_array, prctLight), np.percentile(self.sumEMG1_array, prctLight), np.percentile(self.sumEMG2_array, prctLight), np.percentile(self.sumEMG3_array, prctLight), np.percentile(self.sumEMG4_array, prctLight), np.percentile(self.sumEMG5_array, prctLight), np.percentile(self.sumEMG6_array, prctLight), np.percentile(self.sumEMG7_array, prctLight)])
        prctLight2 = 75
        self.lightHigh = np.array([np.percentile(self.sumEMG0_array, prctLight2), np.percentile(self.sumEMG1_array, prctLight2), np.percentile(self.sumEMG2_array, prctLight2), np.percentile(self.sumEMG3_array, prctLight2), np.percentile(self.sumEMG4_array, prctLight2), np.percentile(self.sumEMG5_array, prctLight2), np.percentile(self.sumEMG6_array, prctLight2), np.percentile(self.sumEMG7_array, prctLight2)])

        # print("0:", np.percentile(self.sumEMG0_array, 5), np.percentile(self.sumEMG0_array, 25))
        # print("1:", np.percentile(self.sumEMG1_array, 5), np.percentile(self.sumEMG1_array, 25))
        # print("2:", np.percentile(self.sumEMG2_array, 5), np.percentile(self.sumEMG2_array, 25))
        # print("3:", np.percentile(self.sumEMG3_array, 5), np.percentile(self.sumEMG3_array, 25))
        # print("4:", np.percentile(self.sumEMG4_array, 5), np.percentile(self.sumEMG4_array, 25))
        # print("5:", np.percentile(self.sumEMG5_array, 5), np.percentile(self.sumEMG5_array, 25))
        # print("6:", np.percentile(self.sumEMG6_array, 5), np.percentile(self.sumEMG6_array, 25))
        # print("7:", np.percentile(self.sumEMG7_array, 5), np.percentile(self.sumEMG7_array, 25))

      if self.trainCounter == 2401:
        print("TIGHTLY grip the water bottle")

      if self.trainCounter == 3200000:
        prctSqueeze = 50
        self.squeeze = np.array([np.percentile(self.sumEMG0_array, prctSqueeze), np.percentile(self.sumEMG1_array, prctSqueeze), np.percentile(self.sumEMG2_array, prctSqueeze), np.percentile(self.sumEMG3_array, prctSqueeze), np.percentile(self.sumEMG4_array, prctSqueeze), np.percentile(self.sumEMG5_array, prctSqueeze), np.percentile(self.sumEMG6_array, prctSqueeze), np.percentile(self.sumEMG7_array, prctSqueeze)])
        i = 0
        for x in self.squeeze:
          i = i+1
          if self.squeeze[i-1] < 500:
            self.relaxStd[i-1] = 1000

        #self.emgElectrode = np.argmin(self.relaxStd)
        self.emgElectrode = 3 #0
        #print(self.relaxStd)
        # self.thresholdLowArray = np.subtract(self.lightHigh, self.relaxHigh) #Change to percentages later
        # self.emgElectrode = np.argmax(self.thresholdLowArray)
        #self.thresholdRelax = self.relax[self.emgElectrode]
        self.thresholdRelax = 100 #70
        #self.thresholdGrip = self.thresholdRelax + 120 #self.squeeze[self.emgElectrode]
        self.thresholdGrip = 300 #150
        print("emgElectrode, thresholdRelax, thresholdGrip")
        print(self.emgElectrode, self.thresholdRelax, self.thresholdGrip)

        # print("0:", np.percentile(self.sumEMG0_array, 5), np.percentile(self.sumEMG0_array, 25))
        # print("1:", np.percentile(self.sumEMG1_array, 5), np.percentile(self.sumEMG1_array, 25))
        # print("2:", np.percentile(self.sumEMG2_array, 5), np.percentile(self.sumEMG2_array, 25))
        # print("3:", np.percentile(self.sumEMG3_array, 5), np.percentile(self.sumEMG3_array, 25))
        # print("4:", np.percentile(self.sumEMG4_array, 5), np.percentile(self.sumEMG4_array, 25))
        # print("5:", np.percentile(self.sumEMG5_array, 5), np.percentile(self.sumEMG5_array, 25))
        # print("6:", np.percentile(self.sumEMG6_array, 5), np.percentile(self.sumEMG6_array, 25))
        # print("7:", np.percentile(self.sumEMG7_array, 5), np.percentile(self.sumEMG7_array, 25))

      if self.trainCounter == 3201:
        print("Training Complete")

      if self.trainCounter > 3201000:
        self.ev_emg_array = np.roll(self.ev_emg_array, -1)
        self.ev_emg_array[len(self.ev_emg_array) - 1] = abs(event.emg[self.emgElectrode])  # emg0 >350<600, #emg1 >700<900, emg2>150<350, #emg3 >800<1000, #emg3 >700<900, #emg4 >250<500, #emg5 >0<250, #emg6 >100<250, #emg7 >100<300
        self.sumEMGvalue = np.sum(self.ev_emg_array)
        self.new = [self.sumEMGvalue, 0, 0, 0, 0, 0, 0, 0]

      if self.trainCounter > 4000000:
        if self.sumEMGvalue > self.thresholdGrip and self.closed == 0:
          firebase.put('', 'MyoCommand', 'close')
          print("grip")
          self.closed = 1
        elif self.sumEMGvalue <= self.thresholdRelax and self.closed == 1:
          firebase.put('', 'MyoCommand', 'open')
          print("extend")
          self.closed = 0

      if self.trainCounter < 3200000:
        self.ev_emg0_array = np.roll(self.ev_emg0_array, -1)
        self.ev_emg0_array[len(self.ev_emg0_array) - 1] = abs(event.emg[0])
        self.sumEMG0_array = np.roll(self.sumEMG0_array, -1)
        self.sumEMG0_array[len(self.sumEMG0_array) - 1] = np.sum(self.ev_emg0_array)

        # self.ev_emg1_array = np.roll(self.ev_emg1_array, -1)
        # self.ev_emg1_array[len(self.ev_emg1_array)-1] = abs(event.emg[1])
        # self.sumEMG1_array = np.roll(self.sumEMG1_array, -1)
        # self.sumEMG1_array[len(self.sumEMG1_array) - 1] = np.sum(self.ev_emg1_array)
        #
        # self.ev_emg2_array = np.roll(self.ev_emg2_array, -1)
        # self.ev_emg2_array[len(self.ev_emg2_array)-1] = abs(event.emg[2])
        # self.sumEMG2_array[len(self.sumEMG2_array) - 1] = np.sum(self.ev_emg2_array)
        # # print(self.sumEMG_array[len(self.sumEMG_array) - 1])

        self.ev_emg3_array = np.roll(self.ev_emg3_array, -1)
        self.ev_emg3_array[len(self.ev_emg3_array)-1] = abs(event.emg[3])
        self.sumEMG3_array = np.roll(self.sumEMG3_array, -1)
        self.sumEMG3_array[len(self.sumEMG3_array) - 1] = np.sum(self.ev_emg3_array)
        # print(self.sumEMG_array[len(self.sumEMG_array) - 1])

        self.ev_emg4_array = np.roll(self.ev_emg4_array, -1)
        self.ev_emg4_array[len(self.ev_emg4_array)-1] = abs(event.emg[4])
        self.sumEMG4_array = np.roll(self.sumEMG4_array, -1)
        self.sumEMG4_array[len(self.sumEMG4_array) - 1] = np.sum(self.ev_emg4_array)
        # print(self.sumEMG_array[len(self.sumEMG_array) - 1])
        #
        # self.ev_emg5_array = np.roll(self.ev_emg5_array, -1)
        # self.ev_emg5_array[len(self.ev_emg5_array)-1] = abs(event.emg[5])
        # self.sumEMG5_array = np.roll(self.sumEMG5_array, -1)
        # self.sumEMG5_array[len(self.sumEMG5_array) - 1] = np.sum(self.ev_emg5_array)
        # # print(self.sumEMG_array[len(self.sumEMG_array) - 1])
        #
        # self.ev_emg6_array = np.roll(self.ev_emg6_array, -1)
        # self.ev_emg6_array[len(self.ev_emg6_array)-1] = abs(event.emg[6])
        # self.sumEMG6_array = np.roll(self.sumEMG6_array, -1)
        # self.sumEMG6_array[len(self.sumEMG6_array) - 1] = np.sum(self.ev_emg6_array)
        # # print(self.sumEMG_array[len(self.sumEMG_array) - 1])
        #
        # self.ev_emg7_array = np.roll(self.ev_emg7_array, -1)
        # self.ev_emg7_array[len(self.ev_emg7_array)-1] = abs(event.emg[7])
        # self.sumEMG7_array = np.roll(self.sumEMG7_array, -1)
        # self.sumEMG7_array[len(self.sumEMG7_array) - 1] = np.sum(self.ev_emg7_array)
        # # print(self.sumEMG_array[len(self.sumEMG_array) - 1])

        self.new = [self.sumEMG0_array[49], self.sumEMG1_array[49], self.sumEMG2_array[49], self.sumEMG3_array[49], self.sumEMG4_array[49], self.sumEMG5_array[49], self.sumEMG6_array[49], self.sumEMG7_array[49]]


      self.emg_data_queue.append((event.timestamp, self.new))

            # if self.closed == 0:
            #    #firebase.put('', 'MyoCommand', 'close')
            #    print("grip")
            #    self.closed = 1
            # else:
            #    #firebase.put('', 'MyoCommand', 'open')
            #    print("extend")
            #    self.closed = 0
            # self.delayCounterEMG = 0

        # self.ev_emg1_array = np.roll(self.ev_emg1_array, -1)
        # self.ev_emg1_array[len(self.ev_emg1_array)-1] = abs(event.emg[1]) #emg0 >350<600, #emg1 >700<900, emg2>150<350, #emg3 >800<1000, #emg3 >700<900, #emg4 >250<500, #emg5 >0<250, #emg6 >100<250, #emg7 >100<300
        # self.sumEMG1_array = np.roll(self.sumEMG1_array, -1)
        # self.sumEMG1_array[len(self.sumEMG1_array) - 1] = np.sum(self.ev_emg1_array)
        # # print(self.sumEMG_array[len(self.sumEMG_array) - 1])
        #
        # # self.ev_emg2_array = np.roll(self.ev_emg2_array, -1)
        # # self.ev_emg2_array[len(self.ev_emg2_array)-1] = abs(event.emg[2]) #emg0 >350<600, #emg1 >700<900, emg2>150<350, #emg3 >800<1000, #emg3 >700<900, #emg4 >250<500, #emg5 >0<250, #emg6 >100<250, #emg7 >100<300
        # # self.sumEMG2_array = np.roll(self.sumEMG2_array, -1)
        # # self.sumEMG2_array[len(self.sumEMG2_array) - 1] = np.sum(self.ev_emg2_array)
        # # # print(self.sumEMG_array[len(self.sumEMG_array) - 1])
        #
        # self.ev_emg3_array = np.roll(self.ev_emg3_array, -1)
        # self.ev_emg3_array[len(self.ev_emg3_array)-1] = abs(event.emg[3]) #emg0 >350<600, #emg1 >700<900, emg2>150<350, #emg3 >800<1000, #emg3 >700<900, #emg4 >250<500, #emg5 >0<250, #emg6 >100<250, #emg7 >100<300
        # self.sumEMG3_array = np.roll(self.sumEMG3_array, -1)
        # self.sumEMG3_array[len(self.sumEMG3_array) - 1] = np.sum(self.ev_emg3_array)
        # # print(self.sumEMG_array[len(self.sumEMG_array) - 1])
        #
        # self.ev_emg4_array = np.roll(self.ev_emg4_array, -1)
        # self.ev_emg4_array[len(self.ev_emg4_array)-1] = abs(event.emg[4]) #emg0 >350<600, #emg1 >700<900, emg2>150<350, #emg3 >800<1000, #emg3 >700<900, #emg4 >250<500, #emg5 >0<250, #emg6 >100<250, #emg7 >100<300
        # self.sumEMG4_array = np.roll(self.sumEMG4_array, -1)
        # self.sumEMG4_array[len(self.sumEMG4_array) - 1] = np.sum(self.ev_emg4_array)
        # # print(self.sumEMG_array[len(self.sumEMG_array) - 1])
        #
        # self.ev_emg5_array = np.roll(self.ev_emg5_array, -1)
        # self.ev_emg5_array[len(self.ev_emg5_array)-1] = abs(event.emg[5]) #emg0 >350<600, #emg1 >700<900, emg2>150<350, #emg3 >800<1000, #emg3 >700<900, #emg4 >250<500, #emg5 >0<250, #emg6 >100<250, #emg7 >100<300
        # self.sumEMG5_array = np.roll(self.sumEMG5_array, -1)
        # self.sumEMG5_array[len(self.sumEMG5_array) - 1] = np.sum(self.ev_emg5_array)
        # # print(self.sumEMG_array[len(self.sumEMG_array) - 1])
        #
        # self.ev_emg6_array = np.roll(self.ev_emg6_array, -1)
        # self.ev_emg6_array[len(self.ev_emg6_array)-1] = abs(event.emg[6]) #emg0 >350<600, #emg1 >700<900, emg2>150<350, #emg3 >800<1000, #emg3 >700<900, #emg4 >250<500, #emg5 >0<250, #emg6 >100<250, #emg7 >100<300
        # self.sumEMG6_array = np.roll(self.sumEMG6_array, -1)
        # self.sumEMG6_array[len(self.sumEMG6_array) - 1] = np.sum(self.ev_emg6_array)
        # # print(self.sumEMG_array[len(self.sumEMG_array) - 1])
        #
        # self.ev_emg7_array = np.roll(self.ev_emg7_array, -1)
        # self.ev_emg7_array[len(self.ev_emg7_array)-1] = abs(event.emg[7]) #emg0 >350<600, #emg1 >700<900, emg2>150<350, #emg3 >800<1000, #emg3 >700<900, #emg4 >250<500, #emg5 >0<250, #emg6 >100<250, #emg7 >100<300
        # self.sumEMG7_array = np.roll(self.sumEMG7_array, -1)
        # self.sumEMG7_array[len(self.sumEMG7_array) - 1] = np.sum(self.ev_emg7_array)
        # # print(self.sumEMG_array[len(self.sumEMG_array) - 1])
        #
        # if self.delayCounterEMG <= 199:
        #     self.delayCounterEMG = self.delayCounterEMG + 1
        #
        # if self.delayCounterEMG>199 and self.sumEMG0_array[49]<600 and 700<self.sumEMG1_array[49] and self.sumEMG2_array[49]<350 and 700<self.sumEMG3_array[49] and self.sumEMG4_array[49]<500 and self.sumEMG5_array[49]<250 and self.sumEMG6_array[49]<250 and self.sumEMG7_array[49]<300:
        #     if self.closed == 0:
        #        #firebase.put('', 'MyoCommand', 'close')
        #        print("grip")
        #        self.closed = 1
        #     else:
        #        #firebase.put('', 'MyoCommand', 'open')
        #        print("extend")
        #        self.closed = 0
        #     self.delayCounterEMG = 0
      ################################
      # store EMG signals
      # self.ev_emg1_array = np.roll(self.ev_emg1_array, -1)
      # self.ev_emg1_array[len(self.ev_emg1_array)-1] = abs(event.emg[3])
      #
      # #filter EMG signals
      # #use sum instead of med - less noisy for same lag and array size
      # self.sumEMG_array = np.roll(self.sumEMG_array, -1)
      # self.sumEMG_array[len(self.sumEMG_array) - 1] = np.sum(self.ev_emg1_array)
      #
      # #gradientEMG = np.gradient(self.sumEMG_array)
      # #print(gradientEMG[len(self.sumEMG_array)-1])
      #
      # #find the change "gradient' between sets of EMG signals
      # diffEMG= np.sum(np.sum(self.sumEMG_array[34:49] - self.sumEMG_array[0:15]))
      # trigSumEMG = 0
      # #EMG sensitivity tuning
      # if diffEMG > 4000:
      #   trigSumEMG = 80
      #
      # # store Gyro signals
      # self.ev_acc_array = np.roll(self.ev_acc_array, -1)
      # self.ev_acc_array[len(self.ev_acc_array) - 1] = abs(self.orientation[0]) + abs(self.orientation[1]) + abs(self.orientation[2])
      #
      # # filter Gyro signals
      # # use sum instead of med - less noisy for same lag and array size
      # self.sumACC_array = np.roll(self.sumACC_array, -1)
      # self.sumACC_array[len(self.sumACC_array) - 1] = np.sum(self.ev_acc_array)
      #
      # # find the change "gradient' between sets of EMG signals
      # diffACC = np.sum(np.sum(self.sumACC_array[46:49] - self.sumACC_array[34:37]))
      # #print(diffACC)
      #
      # #ACC sensitivity tuning
      # if diffACC < -2 or diffACC > 2:
      #   self.trigSumACC = 80
      #   self.moveDelayCounter = 0
      # elif self.moveDelayCounter >= 50:
      #   self.trigSumACC = 0
      #   self.moveDelayCounter = 0
      #
      # if self.moveDelayCounter < 50:
      #   self.moveDelayCounter = self.moveDelayCounter + 1
      #
      # if self.delayCounter < 200:
      #   self.delayCounter = self.delayCounter + 1
      #
      # if self.delayCounter >= 200:
      #   if trigSumEMG == 80 and self.trigSumACC == 0:
      #     self.delayCounter = 0
      #     if self.closed == 0:
      #       firebase.put('', 'MyoCommand', 'close')
      #       print("grip")
      #       self.closed = 1
      #     else:
      #       firebase.put('', 'MyoCommand', 'open')
      #       print("extend")
      #       self.closed = 0
      #
      # #grip EMG is very similar to liftObject EMG beacuse you grip to stabilize the object (no object arm lift can be differentiated)
      # #my extensionEMG is different from my grip EMG but many stroke survivors wont be
      # #what is you need to trigger EMG after object stops moving (relax, grip, relax)
      #
        # new = [self.sumEMG0_array[len(self.sumEMG0_array)-1]]
        # self.emg_data_queue.append((event.timestamp, new))
      # #self.emg_data_queue.append((event.timestamp, event.emg))
#####################
      # x = Symbol('x')
      # y = x ** 2 + 1
      # yprime = y.diff(x)
      # f = lambdify(x, yprime, 'numpy')
      # f(np.ones(5))


      # control mode 7: Derivative flexor EMG amplitude Grip+Relax triggers grip and then triggers release
      # Hypothesis1: people activate muscles to grip but maintain a very low activation to hold objects
      # Hypothesis2: people activate muscles to extend but maintain a very low activation to keep hand extended
      # Works! 100 = 500ms, 700 threshold works for grasp
      # System Works! Myo > Bluetooth > Python (computer) > Firebase (wifi) > App (tablet) > Glove
      # Issue Resolved: triggers open and close if long muscle contraction -> checks for relax before moving
      # Issue: Resolve Monday: grip and relax to grasp is awkward
      # Issue: Resolve Monday: EMG signal is dampened when fingers in extension: Study??
      # Issue: Resolve Monday: Haven't tested with stroke/andrei/illya
      # Issue: Manually tuning variables and logic?
      # Issue: No moving average array or derivative
      # Issue: Resolve Friday: No feedback from therapists: continuous signal, force control, selectively train extensors, visual feedback, gamify
      # Issue: Grip + Relex for flexion and extension reduced false triggers but high thresholds needed not to move during arm motion
      # Issue: finger flexor data is mostly indepednent of arm motion, arm motion causes noise in all signals and some impulse which makes threshold for grasp high
      # emg_channel_grip = 3  # MYO light on dorsal (hairy) side of forearm
      # amplitude_threshold_high = 2000  # + self.ev_emg4   # 700 for Aaron #2000 works with arm motion for water bottle but not much less
      # amplitude_threshold_low = 1500  # 400 for Aaron
      # sampling_duration = 100  # 200Hz EMG sampling, 100 = 500ms
      # max_wait_for_relax = 4  # 4 = 2 seconds

      #Kalman filter - makes code lag - does not filter high frequency data out - maybe works in matlab but dofficult to choose variables here
      # f = KalmanFilter(dim_x=2, dim_z=1)
      # f.x = np.array([2., 0.])
      # f.F = np.array([[1., 1.], [0., 1.]])
      # f.H = np.array([[1., 0.]])
      # f.P = np.array([[1000., 0.], [0., 1000.]])
      # f.R = 100
      # f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=1)
      # z = abs(event.emg[emg_channel_grip])
      # f.predict()
      # f.update(z)
      # # plot(abs(event.emg[emg_channel_grip]))
      # # plot(f.x)
      #new= [float(abs(event.emg[emg_channel_grip])), output]
      #self.emg_data_queue.append((event.timestamp, new))

      # self.ev_emg1 = self.ev_emg1 + abs(event.emg[emg_channel_grip])  # grip derivative
      # # self.ev_emg4 = self.ev_emg4 + abs(event.emg[6])  # bicep derivative
      # self.resetCounter = self.resetCounter + 1
      # if self.resetCounter == sampling_duration:  # amplitude threshold
      #   self.resetCounter = 0
      #   #print(self.ev_emg1)  #uncomment for training to determine thresholds
      #   # print(self.ev_emg4)
      #   if self.ev_emg1 < amplitude_threshold_low and self.waitForRelax == 1 and self.waitForRelaxMax < max_wait_for_relax and self.closeRobot == 0:
      #     print('close')
      #     #firebase.put('', 'MyoCommand', 'close')
      #     self.closeRobot = 1
      #     self.waitForRelax = 0
      #     self.waitForRelaxMax = 0
      #   elif self.ev_emg1 < amplitude_threshold_low and self.waitForRelax == 1 and self.waitForRelaxMax < max_wait_for_relax and self.closeRobot == 1:
      #     print('open')
      #     #firebase.put('', 'MyoCommand', 'open')
      #     self.closeRobot = 0
      #     self.waitForRelax = 0
      #     self.waitForRelaxMax = 0
      #   elif self.ev_emg1 < amplitude_threshold_low:
      #     self.waitForRelaxMax = 0
      #     self.waitForRelax = 0
      #   elif self.ev_emg1 > amplitude_threshold_high and self.closeRobot == 0:
      #     print('wait C')
      #     self.waitForRelax = 1
      #     self.waitForRelaxMax = self.waitForRelaxMax + 1
      #   elif self.ev_emg1 > amplitude_threshold_high and self.closeRobot == 1:
      #     print('wait O')
      #     self.waitForRelax = 1
      #     self.waitForRelaxMax = self.waitForRelaxMax + 1
      #   self.hi=self.ev_emg1
      #   self.ev_emg1 = 0
      #   # self.ev_emg4 = 0



        #new = [abs(event.emg[emg_channel_grip]), f.x[0], f.x[1]]
        # new = [abs(event.emg[emg_channel_grip])]
        # self.emg_data_filt_queue.append((event.timestamp, new))
        # new2= self.emg_data_filt_queue
        # emg_data = np.array([x[1] for x in new2]).T
        # for g, data in zip(self.graphs, emg_data):
        #   if len(data) < self.n:
        #     # Fill the left side with zeroes.
        #     data = np.concatenate([np.zeros(self.n - len(data)), data])
        #   g.set_ydata(data)
        #plt.draw()


class Plot(object):

  def __init__(self, listener):
    self.n = listener.n
    self.listener = listener
    self.fig = plt.figure(figsize=(10, 10))
    self.axes = [self.fig.add_subplot('81' + str(i)) for i in range(1, 9)]
    [(ax.set_ylim([0, 300])) for ax in self.axes]
    [(ax.set_xlim([0, 511])) for ax in self.axes]
    [(ax.set_yticks([0, 50, 100, 150, 200, 250, 300])) for ax in self.axes]
    self.graphs = [ax.plot(np.arange(self.n), np.zeros(self.n))[0] for ax in self.axes]
    plt.ion()

  def update_plot(self):
    emg_data = self.listener.get_emg_data()
    emg_data = np.array([x[1] for x in emg_data]).T
    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    for g, data in zip(self.graphs, emg_data):
      if len(data) < self.n:
        # Fill the left side with zeroes.
        data = np.concatenate([np.zeros(self.n - len(data)), data])
      g.set_ydata(data)
    plt.get_current_fig_manager().window.setGeometry(500, 100, 900, 800)
    plt.draw()

  def main(self):
    while True:
      self.update_plot()
      plt.pause(1.0 / 100)


def main():
  myo.init()
  hub = myo.Hub()
  listener = EmgCollector(512)

  with hub.run_in_background(listener.on_event):
    Plot(listener).main()


if __name__ == '__main__':
  main()