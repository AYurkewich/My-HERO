from matplotlib import pyplot as plt
from collections import deque
from threading import Lock, Thread
import serial
import struct

import myo
import numpy as np
from firebase import firebase
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

firebase = firebase.FirebaseApplication('https://hero-d6297.firebaseio.com/')


class EmgCollector(myo.DeviceListener):
  """
  Collects EMG data in a queue with *n* maximum number of elements.
  """

  def __init__(self, n):
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

    # self.fig = plt.figure()
    # #self.axes = [self.fig.add_subplot('81' + str(i)) for i in range(1, 9)]
    # self.axes = [self.fig.add_subplot('51' + str(i)) for i in range(1, 4)]
    # [(ax.set_ylim([-1, 60])) for ax in self.axes]
    # #[(ax.set_xlim([0, 200])) for ax in self.axes]
    # self.graphs = [ax.plot(np.arange(self.n), np.zeros(self.n))[0] for ax in self.axes]
    # plt.ion()

  def get_emg_data(self):
    with self.lock:
      return list(self.emg_data_queue)

  def get_emg_filt_data(self):
    with self.lock:
      return list(self.emg_data_filt_queue)

  def get_ev_emg_avg(self):
    with self.lock:
      return self.ev_emg_avg

  # myo.DeviceListener

  def on_connected(self, event):
    event.device.stream_emg(True)

  # def on_orientation(self, event):
  #   with self.lock:

  # def on_orientation(self, event):
  #   with self.lock:
  #     self.acceleration = event.acceleration
  #     #print(self.acceleration)


  def on_emg(self, event):
    with self.lock:
      self.emg_data_queue.append((event.timestamp, event.emg))


      # arduino code stored in CableTendonGlove-Left-2018-12-04-Manual_Auto_Buttons
      # # control mode 1: High EMG amplitude triggers grip, Low EMG amplitude triggers release
      # # Does not work: People relax hand once holding an object
      # self.ev_emg_avg = 0.9*self.ev_emg_avg + 0.1*abs(event.emg[1])
      # #print(self.ev_emg_avg)
      # if self.ev_emg_avg > 6.0:
      #   #firebase.put('', 'MyoCommand', 'close')
      #   print('close')
      # else:
      #   #firebase.put('', 'MyoCommand', 'open')
      #   print('open')

      # # control mode 2: EMG integral modulates grip and release
      # # Hypothesis1: people activate muscles to grip but maintain a very low activation to hold objects
      # # Hypothesis2: people activate muscles to extend but maintain a very low activation to keep hand extended
      # # Integration could stably control grip force using: self.ev_emg_avg + 0.005*abs(event.emg[1])
      # # Integration could stably control extension IF users have good extensor control: Do They? STUDY? Sliding mode control
      # # Annoying drift from long term hold or extend - need to isolate and reset extension and flexion, can't use subtraction because very slow due to drift
      # # Need to know Myo orientation
      # # Does not work: Residual signals in integration drift, over 30 seconds residuals are larger than pulse muscle activation
      # self.ev_emg1 = self.ev_emg1 + 0.005 * abs(event.emg[1]) # grip
      # self.ev_emg4 = self.ev_emg4 + 0.005 * abs(event.emg[3]) # extend
      # #self.ev_emg_avg = self.ev_emg1 - self.ev_emg4
      # if self.ev_emg1 < 0.0:
      #    self.ev_emg1 = 0
      # if self.ev_emg4 < 0.0:
      #    self.ev_emg4 = 0
      # print(self.ev_emg1, self.ev_emg4)
      # if self.ev_emg1 > 100.0 and self.closeRobot == 0:
      #    #firebase.put('', 'MyoCommand', 'open')
      #    self.ev_emg4 = 0
      #    self.closeRobot = 1
      #    #print('close')
      # elif self.ev_emg4 > 100.0 and self.closeRobot == 1:
      #   self.ev_emg1 = 0
      #   self.closeRobot = 0
      #   #print('open')

      # # control mode 3: High flexor EMG amplitude triggers grip, High extensor EMG amplitude triggers release
      # # Hypothesis1: people activate muscles to grip but maintain a very low activation to hold objects
      # # Hypothesis2: people activate muscles to extend but maintain a very low activation to keep hand extended
      # # Difference signal is sensitive to drift - need to scale very accurately to maintain zero or else response VERY slow - if drift is low you could use threshold and quick reset
      # # Somewhat works: fine tune scaling, constant resetting and great flexion and extension muscle activation required - grip only, derivative or moving average may be better
      # self.ev_emg1 = self.ev_emg1 + 0.005 * abs(event.emg[1]) # grip
      # self.ev_emg4 = self.ev_emg4 + 0.004 * abs(event.emg[3]) # extend
      # self.ev_emg_avg = self.ev_emg1 - self.ev_emg4
      # #print(self.ev_emg_avg)
      # self.resetCounter = self.resetCounter + 1
      # if self.resetCounter == 500:
      #   self.resetCounter = 0
      #   self.ev_emg1 = 0
      #   self.ev_emg4 = 0
      # if self.ev_emg_avg > 4.0:
      #   self.closeRobot = 1
      #   print('close')
      # elif self.ev_emg_avg < -4.0:
      #   self.closeRobot = 0
      #   print('open')
      # elif self.closeRobot == 1:
      #   print('close')
      # elif self.closeRobot == 0:
      #   print('open')

      # control mode 4: Derivative flexor EMG amplitude triggers grip and then triggers release
      # Hypothesis1: people activate muscles to grip but maintain a very low activation to hold objects
      # Hypothesis2: people activate muscles to extend but maintain a very low activation to keep hand extended
      # Works! 100 = 500ms, 700 threshold works for grasp
      # Issue: triggers open and close too quick - need to check relax derivative
      # Variables: amplitude threshold, max duration, emg channel
      # self.ev_emg1 = self.ev_emg1 + abs(event.emg[1]) # grip derivative
      # self.resetCounter = self.resetCounter + 1
      # if self.resetCounter == 100:
      #   self.resetCounter = 0
      #   if self.ev_emg1>700 and self.closeRobot == 0:
      #     print('close')
      #     self.closeRobot = 1
      #   elif self.ev_emg1 > 700 and self.closeRobot == 1:
      #     print('open')
      #     self.closeRobot = 0
      #   self.ev_emg1 = 0

      # # control mode 5: Derivative flexor EMG amplitude Grip+Relax triggers grip and then triggers release
      # # Hypothesis1: people activate muscles to grip but maintain a very low activation to hold objects
      # # Hypothesis2: people activate muscles to extend but maintain a very low activation to keep hand extended
      # # Works! 100 = 500ms, 700 threshold works for grasp
      # # System Works! Myo > Bluetooth > Python (computer) > Firebase (wifi) > App (tablet) > Glove
      # # Issue Resolved: triggers open and close if long muscle contraction -> checks for relax before moving
      # # Issue: Resolve Monday: grip and relax to grasp is awkward
      # # Issue: Resolve Monday: EMG signal is dampened when fingers in extension: Study??
      # # Issue: Resolve Monday: Haven't tested with stroke/andrei/illya
      # # Issue: Manually tuning variables and logic?
      # # Issue: No moving average array or derivative
      # # Issue: Resolve Friday: No feedback from therapists: continuous signal, force control, selectively train extensors, visual feedback, gamify
      # # Issue: Grip + Relex for flexion and extension reduced false triggers but high thresholds needed not to move during arm motion
      # # Issue: finger flexor data is mostly indepednent of arm motion, arm motion causes noise in all signals and some impulse which makes threshold for grasp high
      # emg_channel_grip = 3  # MYO light on dorsal (hairy) side of forearm
      # amplitude_threshold_high = 2000 # + self.ev_emg4   # 700 for Aaron #2000 works with arm motion for water bottle but not much less
      # amplitude_threshold_low = 1500  # 400 for Aaron
      # sampling_duration = 100  # 200Hz EMG sampling, 100 = 500ms
      # max_wait_for_relax = 4  # 4 = 2 seconds
      #
      # self.ev_emg1 = self.ev_emg1 + abs(event.emg[emg_channel_grip])  # grip derivative
      # #self.ev_emg4 = self.ev_emg4 + abs(event.emg[6])  # bicep derivative
      # self.resetCounter = self.resetCounter + 1
      # if self.resetCounter == sampling_duration:  # amplitude threshold
      #   self.resetCounter = 0
      #   #print(self.ev_emg1)  #uncomment for training to determine thresholds
      #   #print(self.ev_emg4)
      #   if self.ev_emg1 < amplitude_threshold_low and self.waitForRelax == 1 and self.waitForRelaxMax < max_wait_for_relax and self.closeRobot == 0:
      #     print('close')
      #     firebase.put('', 'MyoCommand', 'close')
      #     self.closeRobot = 1
      #     self.waitForRelax = 0
      #     self.waitForRelaxMax = 0
      #   elif self.ev_emg1 < amplitude_threshold_low and self.waitForRelax == 1 and self.waitForRelaxMax < max_wait_for_relax and self.closeRobot == 1:
      #     print('open')
      #     firebase.put('', 'MyoCommand', 'open')
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
      #   self.ev_emg1 = 0
      #   #self.ev_emg4 = 0

      # # control mode 6: Derivative flexor EMG amplitude Grip triggers grip and then grip+relax triggers release
      # # Hypothesis1: people activate muscles to grip but maintain a very low activation to hold objects
      # # Hypothesis2: people activate muscles to extend but maintain a very low activation to keep hand extended
      # # Works! 100 = 500ms, 700 threshold works for grasp
      # # System Works! Myo > Bluetooth > Python (computer) > Firebase (wifi) > App (tablet) > Glove
      # # Issue Resolved: triggers open and close if long muscle contraction -> checks for relax before moving
      # # Issue: Resolve Monday: grip and relax to grasp is awkward
      # # Issue: Resolve Monday: EMG signal is dampened when fingers in extension: Study??
      # # Issue: Resolve Monday: Haven't tested with stroke/andrei/illya
      # # Issue: Manually tuning variables and logic?
      # # Issue: No moving average array or derivative
      # # Issue: Resolve Friday: No feedback from therapists: continuous signal, force control, selectively train extensors, visual feedback, gamify
      # # Issue: lifting your arm triggers flexion because of bicep; at rest glove should stay in extension
      # emg_channel_grip = 4  # MYO light on dorsal (hairy) side of forearm
      # amplitude_threshold_high = 500  # 700 for Aaron
      # amplitude_threshold_low = 400  # 400 for Aaron # never go below 200, rarely go below 300
      # sampling_duration = 100  # 200Hz EMG sampling, 100 = 500ms
      # max_wait_relax = 4  # 4 = 2 seconds
      #
      # self.ev_emg1 = self.ev_emg1 + abs(event.emg[emg_channel_grip])  # grip derivative
      # self.resetCounter = self.resetCounter + 1
      # if self.resetCounter == sampling_duration:  # amplitude threshold
      #   self.resetCounter = 0
      #   print(self.ev_emg1)  # uncomment for training to determine thresholds
      #   if self.ev_emg1 > amplitude_threshold_high and self.closeRobot == 0:
      #     print('close')
      #     firebase.put('', 'MyoCommand', 'close')
      #     self.closeRobot = 1
      #   elif self.ev_emg1 < amplitude_threshold_low and self.waitForRelax == 1 and self.waitForRelaxMax < max_wait_relax and self.closeRobot == 1:
      #     print('open')
      #     firebase.put('', 'MyoCommand', 'open')
      #     self.closeRobot = 0
      #     self.waitForRelax = 0
      #     self.waitForRelaxMax = 0
      #     self.relaxBeforeFLex = 0
      #   elif self.ev_emg1 < amplitude_threshold_low and self.closeRobot == 1:
      #     print('relax -> flex then relax to open')
      #     self.relaxBeforeFLex = 1
      #     self.waitForRelaxMax = 0
      #     self.waitForRelax = 0
      #   elif self.ev_emg1 > amplitude_threshold_high and self.relaxBeforeFLex == 1 and self.closeRobot == 1:
      #     print('relax and flex -> relax to open')
      #     self.waitForRelax = 1
      #     self.waitForRelaxMax = self.waitForRelaxMax + 1
      #   elif self.waitForRelaxMax >= max_wait_relax:
      #     print (' flex then relax quicker ')
      #   self.ev_emg1 = 0

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
      emg_channel_grip = 3  # MYO light on dorsal (hairy) side of forearm
      amplitude_threshold_high = 2000  # + self.ev_emg4   # 700 for Aaron #2000 works with arm motion for water bottle but not much less
      amplitude_threshold_low = 1500  # 400 for Aaron
      sampling_duration = 100  # 200Hz EMG sampling, 100 = 500ms
      max_wait_for_relax = 4  # 4 = 2 seconds

      self.ev_emg1 = self.ev_emg1 + abs(event.emg[emg_channel_grip])  # grip derivative
      # self.ev_emg4 = self.ev_emg4 + abs(event.emg[6])  # bicep derivative
      self.resetCounter = self.resetCounter + 1
      if self.resetCounter == sampling_duration:  # amplitude threshold
        self.resetCounter = 0
        print(self.ev_emg1)  #uncomment for training to determine thresholds
        # print(self.ev_emg4)
        if self.ev_emg1 < amplitude_threshold_low and self.waitForRelax == 1 and self.waitForRelaxMax < max_wait_for_relax and self.closeRobot == 0:
          print('close')
          firebase.put('', 'MyoCommand', 'close')
          self.closeRobot = 1
          self.waitForRelax = 0
          self.waitForRelaxMax = 0
        elif self.ev_emg1 < amplitude_threshold_low and self.waitForRelax == 1 and self.waitForRelaxMax < max_wait_for_relax and self.closeRobot == 1:
          print('open')
          firebase.put('', 'MyoCommand', 'open')
          self.closeRobot = 0
          self.waitForRelax = 0
          self.waitForRelaxMax = 0
        elif self.ev_emg1 < amplitude_threshold_low:
          self.waitForRelaxMax = 0
          self.waitForRelax = 0
        elif self.ev_emg1 > amplitude_threshold_high and self.closeRobot == 0:
          print('wait C')
          self.waitForRelax = 1
          self.waitForRelaxMax = self.waitForRelaxMax + 1
        elif self.ev_emg1 > amplitude_threshold_high and self.closeRobot == 1:
          print('wait O')
          self.waitForRelax = 1
          self.waitForRelaxMax = self.waitForRelaxMax + 1
        self.ev_emg1 = 0
        # self.ev_emg4 = 0

        # f = KalmanFilter(dim_x=2, dim_z=1)
        # f.x = np.array([2., 0.])
        # f.F = np.array([[1., 1.], [0., 1.]])
        # f.H = np.array([[1., 0.]])
        # f.P = np.array([[1000., 0.], [0., 1000.]])
        # f.R = 5
        # f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)
        # z = abs(event.emg[emg_channel_grip])
        # f.predict()
        # f.update(z)
        # # plot(abs(event.emg[emg_channel_grip]))
        # # plot(f.x)
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
    #self.fig = plt.figure()
    #self.axes = [self.fig.add_subplot('81' + str(i)) for i in range(1, 9)]
    # self.axes = [self.fig.add_subplot('51' + str(i)) for i in range(1, 4)]
    # [(ax.set_ylim([-1, 60])) for ax in self.axes]
    # #[(ax.set_xlim([0, 200])) for ax in self.axes]
    # self.graphs = [ax.plot(np.arange(self.n), np.zeros(self.n))[0] for ax in self.axes]
    # plt.ion()

  def update_plot(self):
    emg_data = self.listener.get_emg_data()
    # emg_data = self.listener.get_emg_filt_data()
    # emg_data = np.array([x[1] for x in emg_data]).T
    # for g, data in zip(self.graphs, emg_data):
    #   if len(data) < self.n:
    #     # Fill the left side with zeroes.
    #     data = np.concatenate([np.zeros(self.n - len(data)), data])
    #   g.set_ydata(data)
    # plt.draw()


  def main(self):
    while True:
      emg_test = abs(self.listener.get_ev_emg_avg())
      # print('yaas')
      #print(emg_test)
      # else:
      #    firebase.put('', 'MyoCommand', 'close')
      # #serial to arduino begin (don't forget ser up top)
      # ser.write(struct.pack('B', emg_test))  # write a string
      # while ser.in_waiting:  # Or: while ser.inWaiting():
      #   ser.read()
      # #serial to arduino end
        #print(ord(ser.read()))


  #comment out for faster code
      #self.update_plot()
      #plt.pause(1.0 / 30)


def main():
  myo.init()
  hub = myo.Hub()
  listener = EmgCollector(512)


  with hub.run_in_background(listener.on_event):
    Plot(listener).main()

if __name__ == '__main__':
  main()