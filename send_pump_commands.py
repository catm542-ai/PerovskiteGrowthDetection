
import argparse
import serial
import serial.win32
import time

ap=argparse.ArgumentParser()
ap.add_argument("-n","--name",required=True,help="Name Entered By the User")
args=vars(ap.parse_args())
name=args["name"]
print(name)
name1=str(name)+"\r"

puerto = "COM4"
baudrate = 9600
ser = serial.Serial(port=puerto, baudrate=baudrate, stopbits=1)
#ser.write(b'/2V1000EA0R\r')
ser.write(name1.encode())
time.sleep(0)
ser.close()
