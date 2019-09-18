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
    self.sumEMG0_array = np.zeros(50)
    self.sumEMG1_array = np.zeros(50)
    self.sumEMG2_array = np.zeros(50)
    self.sumEMG3_array = np.zeros(50)
    self.sumEMG4_array = np.zeros(50)
    self.sumEMG5_array = np.zeros(50)
    self.sumEMG6_array = np.zeros(50)
    self.sumEMG7_array = np.zeros(50) #50 to 400
    self.delayCounterEMG = 200
    # self.ev_emg5_array = np.zeros(50)
    # self.ev_acc_array = np.zeros(50)
    # self.sumEMG_array = np.zeros(50)
    # self.sumACC_array = np.zeros(50)
    self.closed = 0
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
    self.light = np.zeros(8)
    self.lift = np.zeros(8)
    self.relax = np.zeros(8)
    self.relaxStd = np.zeros(8)
    self.squeeze = np.zeros(8)
    self.ev_emg_store = np.zeros([8, 50])
    self.sumEMG = np.zeros(8)
    self.emgCounter = 0


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
        pass # print("REST ARM ON TABLE - RELAX HAND")
      
      if self.trainCounter == 1200:
        prctRelax = 50
        self.relax = [np.percentile(self.sumEMG0_array, prctRelax), np.percentile(self.sumEMG1_array, prctRelax),
           np.percentile(self.sumEMG2_array, prctRelax), np.percentile(self.sumEMG3_array, prctRelax),
           np.percentile(self.sumEMG4_array, prctRelax), np.percentile(self.sumEMG5_array, prctRelax),
           np.percentile(self.sumEMG6_array, prctRelax), np.percentile(self.sumEMG7_array, prctRelax)]
            
        self.relaxStd = [np.std(self.sumEMG0_array), np.std(self.sumEMG1_array),
           np.std(self.sumEMG2_array), np.std(self.sumEMG3_array),
           np.std(self.sumEMG4_array), np.std(self.sumEMG5_array),
           np.std(self.sumEMG6_array), np.std(self.sumEMG7_array)]
        
      if self.trainCounter == 1201:
        pass # print("LIFT ARM - RELAX HAND")

      if self.trainCounter == 2400:
        prctLift = 50
        self.lift = [np.percentile(self.sumEMG0_array, prctLift), np.percentile(self.sumEMG1_array, prctLift),
           np.percentile(self.sumEMG2_array, prctLift), np.percentile(self.sumEMG3_array, prctLift),
           np.percentile(self.sumEMG4_array, prctLift), np.percentile(self.sumEMG5_array, prctLift),
           np.percentile(self.sumEMG6_array, prctLift), np.percentile(self.sumEMG7_array, prctLift)]

      if self.trainCounter == 2401:
        pass # print("REST ARM ON TABLE - LIGHTLY FLEX HAND")

      if self.trainCounter == 3600:
        prctLight = 50
        self.light = [np.percentile(self.sumEMG0_array, prctLight), np.percentile(self.sumEMG1_array, prctLight),
           np.percentile(self.sumEMG2_array, prctLight), np.percentile(self.sumEMG3_array, prctLight),
           np.percentile(self.sumEMG4_array, prctLight), np.percentile(self.sumEMG5_array, prctLight),
           np.percentile(self.sumEMG6_array, prctLight), np.percentile(self.sumEMG7_array, prctLight)]

      if self.trainCounter == 3601:
        pass # print("REST ARM ON TABLE - FLEX HAND TIGHT")

      if self.trainCounter == 4800:
        prctSqueeze = 50
        self.squeeze = [np.percentile(self.sumEMG0_array, prctSqueeze), np.percentile(self.sumEMG1_array, prctSqueeze),
           np.percentile(self.sumEMG2_array, prctSqueeze), np.percentile(self.sumEMG3_array, prctSqueeze),
           np.percentile(self.sumEMG4_array, prctSqueeze), np.percentile(self.sumEMG5_array, prctSqueeze),
           np.percentile(self.sumEMG6_array, prctSqueeze), np.percentile(self.sumEMG7_array, prctSqueeze)]

        ## Find EMG Electrode, Relax Threshold and Flex Threshold:
        # Based on: (minimize relaxStd), (maximize squeeze - relax), (minimize lift - rest) -> Steady signal when hand at rest, Large signal change from hand at rest to squeeze, minimal difference between arm at rest and lifted

        squeezeCount = 0
        for squeezeValue in self.squeeze:
          if self.squeeze[squeezeCount] < float(300):
            self.lift[squeezeCount] = float(10000.0)
            # print("squeeze: ", squeezeCount)
          squeezeCount = squeezeCount+1

        relaxCount = 0
        for relaxValue in self.relaxStd:
          if self.relaxStd[relaxCount] > float(10):
            self.lift[relaxCount] = float(10000.0)
            # print("Std: ", relaxCount)
          relaxCount = relaxCount + 1

        self.emgElectrode = np.argmin(np.subtract(self.lift, self.relax))
        if self.lift[self.emgElectrode] >= float(10000.0):
          # print ("RETRAIN")
          self.trainCounter = 0

        self.thresholdRelax = self.relax[self.emgElectrode] + 10
        #self.thresholdRelax = np.percentile(self.relax[self.emgElectrode], 95)
        self.thresholdGrip = self.light[self.emgElectrode]
        # print("Training Complete")
        # print("emgElectrode, thresholdRelax, thresholdGrip")
        # print(self.emgElectrode, self.thresholdRelax, self.thresholdGrip)
        # print(self.relaxStd)

      if self.trainCounter > 4800:
        pass
        # self.ev_emg_array = np.roll(self.ev_emg_array, -1)
        # self.ev_emg_array[len(self.ev_emg_array) - 1] = abs(event.emg[self.emgElectrode])
        # self.sumEMGvalue = np.sum(self.ev_emg_array)
        # self.new = [self.sumEMGvalue, 0, 0, 0, 0, 0, 0, 0]

      if self.trainCounter > 500: # Change back to 5000
        if self.sumEMG[0] > 200 and self.closed == 0:
           firebase.put('', 'MyoCommand', 'close')
           print("grip")
           self.closed = 1
        elif self.sumEMG[4] < 80 and self.sumEMG[0] < 100 and self.closed == 1:
          firebase.put('', 'MyoCommand', 'open')
          print("extend")
          self.closed = 0

         ## BEST Current Code
        # if self.sumEMGvalue > self.thresholdGrip and self.closed == 0:
        #   firebase.put('', 'MyoCommand', 'close')
        #   print("grip")
        #   self.closed = 1
        # elif self.sumEMGvalue <= self.thresholdRelax and self.closed == 1:
        #   firebase.put('', 'MyoCommand', 'open')
        #   print("extend")
        #   self.closed = 0
        ####

      if self.trainCounter > 0:  # Change back to <=4800
        self.emgCounter = 0
        for row in self.ev_emg_store:  #could use iter if you want it quicker (and messy)
          self.ev_emg_store[self.emgCounter] = np.roll(self.ev_emg_store[self.emgCounter], -1)
          self.ev_emg_store[self.emgCounter, -1] = abs(event.emg[self.emgCounter])
          self.sumEMG[self.emgCounter] = np.sum(self.ev_emg_store[self.emgCounter])
          self.emgCounter = self.emgCounter + 1

        self.emg_data_queue.append((event.timestamp, self.sumEMG))

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