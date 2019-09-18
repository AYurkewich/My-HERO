from matplotlib import pyplot as plt
from collections import deque
from threading import Lock, Thread
import serial
import struct

import myo
import numpy as np

# pusing BLE suite


class EmgCollector(myo.DeviceListener):
  """
  Collects EMG data in a queue with *n* maximum number of elements.
  """

  def __init__(self, n):
    self.n = n
    self.lock = Lock()
    self.emg_data_queue = deque(maxlen=n)
    self.ev_emg1 = 0

  def get_emg_data(self):
    with self.lock:
      return list(self.emg_data_queue)

  def get_ev_emg1(self):
    with self.lock:
      return self.ev_emg1

  # myo.DeviceListener

  def on_connected(self, event):
    event.device.stream_emg(True)


  def on_emg(self, event):
    with self.lock:
      self.emg_data_queue.append((event.timestamp, event.emg))
      self.ev_emg1 = event.emg[1]


class Plot(object):

  def __init__(self, listener):
    self.n = listener.n
    self.listener = listener
    self.fig = plt.figure()
    self.axes = [self.fig.add_subplot('81' + str(i)) for i in range(1, 9)]
    [(ax.set_ylim([-100, 100])) for ax in self.axes]
    self.graphs = [ax.plot(np.arange(self.n), np.zeros(self.n))[0] for ax in self.axes]
    plt.ion()

  def update_plot(self):
    emg_data = self.listener.get_emg_data()
    emg_data = np.array([x[1] for x in emg_data]).T
    for g, data in zip(self.graphs, emg_data):
      if len(data) < self.n:
        # Fill the left side with zeroes.
        data = np.concatenate([np.zeros(self.n - len(data)), data])
      g.set_ydata(data)
    plt.draw()


  def main(self):
    while True:
      emg_test = abs(self.listener.get_ev_emg1())
      # #serial to arduino begin (don't forget ser up top)
      # ser.write(struct.pack('B', emg_test))  # write a string
      # while ser.in_waiting:  # Or: while ser.inWaiting():
      #   ser.read()
      # #serial to arduino end
        #print(ord(ser.read()))

  #comment out for faster code
      self.update_plot()
      plt.pause(1.0 / 30)


def main():
  myo.init()
  hub = myo.Hub()
  listener = EmgCollector(512)


  with hub.run_in_background(listener.on_event):
    Plot(listener).main()


if __name__ == '__main__':
  main()