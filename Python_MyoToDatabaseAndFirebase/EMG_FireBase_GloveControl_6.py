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
import pandas as pd


matplotlib.use('Qt4Agg')
firebase = firebase.FirebaseApplication('https://hero-d6297.firebaseio.com/')
firebase.put('', 'MyoCommand', 'relax')


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
    self.delayCounterEMG = 200
    self.ev_emg_store = np.zeros([8, 50])
    self.sumEMG = np.zeros(8)
    self.emgCounter = 0
    self.closed = 1
    self.counterEmgData = 0

    self.calib = 0 #NEW
    self.relax = np.zeros(8)
    self.arm = np.zeros(8)
    self.hand = np.zeros(8)
    self.handArm = np.zeros(8)
    self.armRelax = np.zeros(8)


    # CALIBRATION:
    # #myo on RIGHT arm
    # self.handEmgElectrode = 1  # myo on RIGHT arm
    # self.handEmgRelax = 60
    # self.handEmgGrip = 170
    # self.armEmgElectrode = 5  # myo on RIGHT arm
    # self.armEmgRelax = 52
    # self.f = open('test15.csv', "a")

    # #myo on LEFT arm
    self.handEmgElectrode = 0  # myo on LEFT arm
    self.handEmgRelax = 100
    self.handEmgGrip = 250
    self.armEmgElectrode = 2  # myo on LEFT arm
    self.armEmgRelax = 300
    self.f = open('test_AY1.csv', "a")  # P32_Glove.csv', "a")  # Uncomment to SAVE data

    self.f.write("TIMESTAMP" + ", " + "EMG0" + ", " + "EMG1" + ", "
      + "EMG2" + ", " + "EMG3" + ", " + "EMG4" + ", "
      + "EMG5" + ", " + "EMG6" + ", " + "EMG7" + ", "
      + "ACCELERATION0" + ", " + "ACCELERATION1" + ", " + "ACCELERATION2" + ", "
      + "ORIENTATION0" + ", " + "ORIENTATION1" + ", " + "ORIENTATION2" + ", "
      + "GLOVEEXTENDEDIF0" + ","
      + "CALIBRATION: HAND ELECTRODE" + "," + "HAND RELAXED" + ", " + "HAND GRIPPING" + ", "
      + "ARM ELECTRODE" + ", " + "ARM RELAXED" + "\n")

  def on_orientation(self, event):
     with self.lock:
        self.acceleration = [event.acceleration[0], event.acceleration[1], event.acceleration[2]]
        self.orientation = [event.orientation[0], event.orientation[1], event.orientation[2]]

  def get_emg_data(self):
    with self.lock:
      return list(self.emg_data_queue)

  # myo.DeviceListener

  def on_connected(self, event):
    event.device.stream_emg(True)

  def on_emg(self, event):
    with self.lock:
        self.emgCounter = 0
        self.calib = 0  # NEW

        #store emg data, average filter emg data
        for row in self.ev_emg_store:  #could use iter if you want it quicker (and messy)
          self.ev_emg_store[self.emgCounter] = np.roll(self.ev_emg_store[self.emgCounter], -1)
          self.ev_emg_store[self.emgCounter, -1] = abs(event.emg[self.emgCounter])
          self.sumEMG[self.emgCounter] = np.sum(self.ev_emg_store[self.emgCounter])
          self.emgCounter = self.emgCounter + 1

        #save data, self.closed == 1 == grip, sorry for being verbose ;)
        self.f.write(str(event.timestamp) + ", " + str(event.emg[0]) + ", " + str(event.emg[1]) + ", "
            + str(event.emg[2]) + ", " + str(event.emg[3]) + ", " + str(event.emg[4]) + ", "
            + str(event.emg[5]) + ", " + str(event.emg[6]) + ", " + str(event.emg[7]) + ", "
            + str(self.acceleration[0]) + ", " + str(self.acceleration[1]) + ", " + str(self.acceleration[2]) + ", "
            + str(self.orientation[0]) + ", " + str(self.orientation[1]) + ", " + str(self.orientation[2]) + ", "
            + str(self.closed) + "\n")

        if self.counterEmgData == 500:  #NEW
          print('RELAX ARM AND HAND')

        if 1500 < self.counterEmgData < 2001:  #NEW
          for row in self.sumEMG:
            self.relax[self.calib] = self.relax[self.calib] + self.sumEMG[self.calib]/500
            self.calib = self.calib + 1

        if self.counterEmgData == 2500:  #NEW
          print('LIFT ARM, RELAX HAND')

        if 3500 < self.counterEmgData < 4001:  #NEW
          for row in self.sumEMG:
            self.arm[self.calib] = self.arm[self.calib] + self.sumEMG[self.calib]/500
            self.calib = self.calib + 1


        if self.counterEmgData == 4500:  # NEW
          print('LIFT ARM, MAKE FIST')

        if 5500 < self.counterEmgData < 6001:  # NEW
          for row in self.sumEMG:
            self.hand[self.calib] = self.hand[self.calib] + self.sumEMG[self.calib]/500
            self.calib = self.calib + 1

        if self.counterEmgData == 6001:
          for row in self.sumEMG:
            self.handArm[self.calib] = self.hand[self.calib] - self.arm[self.calib]
            self.armRelax[self.calib] = self.arm[self.calib] - self.relax[self.calib]
            self.calib = self.calib + 1

          print("Relax: ", self.relax)
          print("Arm: ", self.arm)
          print("Hand: ", self.hand)

          self.handEmgElectrode = np.argmax(self.handArm)
          self.handEmgRelax = self.arm[self.handEmgElectrode] + 20  # self.relax[self.handEmgElectrode] + 10
          self.handEmgGrip = (self.hand[self.handEmgElectrode] - self.handEmgRelax)/2 + self.handEmgRelax
          self.armEmgElectrode = np.argmax(self.armRelax)
          self.armEmgRelax = self.relax[self.armEmgElectrode] + 30

          print("HAND Electrode, Relax, Grip: ", self.handEmgElectrode, ", ", self.handEmgRelax, ", ", self.handEmgGrip)
          print("ARM Electrode, Relax: ", self.armEmgElectrode, ", ", self.armEmgRelax)

          self.f.write(str(event.timestamp) + ", " + str(event.emg[0]) + ", " + str(event.emg[1]) + ", "
            + str(event.emg[2]) + ", " + str(event.emg[3]) + ", " + str(event.emg[4]) + ", "
            + str(event.emg[5]) + ", " + str(event.emg[6]) + ", " + str(event.emg[7]) + ", "
            + str(self.acceleration[0]) + ", " + str(self.acceleration[1]) + ", " + str(self.acceleration[2]) + ", "
            + str(self.orientation[0]) + ", " + str(self.orientation[1]) + ", " + str(self.orientation[2]) + ", "
            + str(self.closed) + ", "
            + str(self.handEmgElectrode) + ", " + str(self.handEmgRelax) + ", " + str(self.handEmgGrip) + ", "
            + str(self.armEmgElectrode) + ", " + str(self.armEmgRelax) + ", " + "CALIBRATED" + "\n")

        if self.counterEmgData > 6001:
          #send open or close command to firebase to control glove, be careful not to write commands after the put statement
          if self.sumEMG[self.handEmgElectrode] > self.handEmgGrip and self.closed == 0:
            print("grip")
            self.closed = 1
            firebase.put('', 'MyoCommand', 'close')
          elif self.sumEMG[self.armEmgElectrode] < self.armEmgRelax and self.sumEMG[self.handEmgElectrode] < self.handEmgRelax and self.closed == 1:
            print("extend")
            self.closed = 0
            firebase.put('', 'MyoCommand', 'open')
          #print semgEMG Values
          if self.counterEmgData % 50 == 0:
            print(self.sumEMG)

        self.counterEmgData = self.counterEmgData + 1

class Plot(object):

  def __init__(self, listener):
    pass
    # self.n = listener.n
    # self.listener = listener
    # self.fig = plt.figure(figsize=(10, 10))
    # self.axes = [self.fig.add_subplot('81' + str(i)) for i in range(1, 9)]
    # [(ax.set_ylim([0, 300])) for ax in self.axes]
    # [(ax.set_xlim([0, 511])) for ax in self.axes]
    # [(ax.set_yticks([0, 50, 100, 150, 200, 250, 300])) for ax in self.axes]
    # self.graphs = [ax.plot(np.arange(self.n), np.zeros(self.n))[0] for ax in self.axes]
    # plt.ion()


  def update_plot(self):
    pass
    # emg_data = self.listener.get_emg_data()
    # emg_data = np.array([x[1] for x in emg_data]).T
    # # figManager = plt.get_current_fig_manager()
    # # figManager.window.showMaximized()
    # for g, data in zip(self.graphs, emg_data):
    #   if len(data) < self.n:
    #     # Fill the left side with zeroes.
    #     data = np.concatenate([np.zeros(self.n - len(data)), data])
    #   g.set_ydata(data)
    # plt.get_current_fig_manager().window.setGeometry(500, 100, 900, 800)
    # plt.draw()


  def main(self):
    while True:
      #self.update_plot() #  uncomment
      plt.pause(1.0 / 100) # plt.pause is required (a pause in the main is required or CPU usage goes past 100%), 1/100 has less CPU usage than 1/1000, less may be more ideal?


def main():
  myo.init()
  hub = myo.Hub()
  listener = EmgCollector(512)

  with hub.run_in_background(listener.on_event):
    Plot(listener).main()


if __name__ == '__main__':
  main()