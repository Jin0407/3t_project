# !/usr/bin/python
# coding:utf-8 
"""
Created on Tue Mar 15 09:02:07 2022

@author: User
"""

import serial
import serial.tools.list_ports

#serialPort="COM0"  #串口
serialPort="/dev/ttyUSB0"  #串口
baudRate=9600  #波特率
s=serial.Serial(serialPort,baudRate)

greenon=[0xFE , 0x05 , 0x00 , 0x00 , 0xFF , 0x00 , 0x98 , 0x35]
greenoff=[0xFE , 0x05 , 0x00 , 0x00 , 0x00 , 0x00 , 0xD9 , 0xC5]
redon=[0xFE , 0x05 , 0x00 , 0x01 , 0xFF , 0x00 , 0xC9 , 0xF5]
redoff=[0xFE , 0x05 , 0x00 , 0x01 , 0x00 , 0x00 , 0x88, 0x05]

def Greenon():
    s.write(greenon)
    print("綠開")

def Greenoff():
    s.write(greenoff)
    print("綠關")
    
def Redon():
    s.write(redon)
    print("紅開")
    
def Redoff():
    s.write(redoff)
    print("紅關")
    
# Greenoff()
Redoff()
# Redon()