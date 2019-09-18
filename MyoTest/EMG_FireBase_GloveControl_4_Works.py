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
    self.delayCounterEMG = 200
    self.ev_emg_store = np.zeros([8, 50])
    self.sumEMG = np.zeros(8)
    self.emgCounter = 0
    self.closed = 1
    self.counterEmgData = 0

    # CALIBRATION:
    # #myo on RIGHT arm
    # self.handEmgElectrode = 1  # myo on RIGHT arm
    # self.handEmgRelax = 60
    # self.handEmgGrip = 170
    # self.armEmgElectrode = 5  # myo on RIGHT arm
    # self.armEmgRelax = 52
    # self.f = open('test15.csv', "a")

    # #myo on LEFT arm
    self.handEmgElectrode = 2  # myo on LEFT arm
    self.handEmgRelax = 70
    self.handEmgGrip = 180
    self.armEmgElectrode = 2  # myo on LEFT arm
    self.armEmgRelax = 100
    self.f = open('test_HP020_Glove2Manual.csv', "a")  # P32_Glove.csv', "a")  # Uncomment to SAVE data


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