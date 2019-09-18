import serial
import sys
import glob

if sys.platform.startswith('win'):
    ports = ['COM%s' % (i + 1) for i in range(256)]
elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
    # this excludes your current terminal "/dev/tty"
    ports = glob.glob('/dev/tty[A-Za-z]*')
elif sys.platform.startswith('darwin'):
    ports = glob.glob('/dev/tty.*')
else:
    raise EnvironmentError('Unsupported platform')

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
print(ser.name)         # check which port was really used, should be like /dev/tty.usbmodem14101
ser.write(b'hello')     # write a string
response = ser.read()
print(response)
ser.close()             # close port
