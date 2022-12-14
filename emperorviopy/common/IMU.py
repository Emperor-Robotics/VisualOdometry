import serial
import time
from collections import namedtuple

IMURead = namedtuple(
    'IMURead', ['yaw', 'pitch', 'roll', 'accX', 'accY', 'accZ'])

# TODO: Need to be able to read latest line? not whats stuck in buffer?


class IMU:
    """
    IMU interfacing device. 
    Currently developed to read serial output from an arduino nano + MPU6050.
    But should work assuming correct port and baudrate are supplied and device serial output 
    is in the format expected:
        "ROLL PITCH YAW ACCEL_X ACCEL_Y ACCEL_Z"

    Calibration output: 
             aX     aY    aZ     gX     gY    gZ  
    Current: -850	297	1213	-332	397	-680
    """

    def __init__(self, serial_port: str = '/dev/ttyUSB0', baudrate: int = 38400, timeout=.1) -> None:
        self.ardu = serial.Serial(
            port=serial_port, baudrate=baudrate, timeout=timeout)
        self.ardu.reset_output_buffer()
        # arducode currently requires an initalization byte, can be anything. "press to start"
        self.initalize()

    def initalize(self):
        self.is_initalized = False
        print("Attempting intialization...")
        trycount = 100
        while not self.is_initalized:
            self.ardu.write(bytes('k', 'utf-8'))
            line = self.ardu.readline().decode()
            if len(line) > 2:
                if line[0].isnumeric() or line[1].isnumeric():
                    self.is_initalized = True
            trycount -= 1
            if trycount <= 0:
                print("Initialization failed!")
                return
        print("Initialization complete!")

    def read_IMU(self) -> IMURead:
        if not self.is_initalized:
            return -1
        lines = self.ardu.readline().decode()
        split = lines.split()
        if len(split) != 6:
            print("Bad read.")
            return -1
        return IMURead(*split)


if __name__ == '__main__':
    imu = IMU()
    while 1:
        print(imu.read_IMU())
