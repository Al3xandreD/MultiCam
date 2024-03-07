import os
import subprocess
# import libraries
from vidgear.gears import VideoGear
from vidgear.gears import NetGear


def get_current_lan_ssid():
    wlan_interfaces = subprocess.check_output("netsh wlan show interfaces")
    try:
        return 1, (wlan_interfaces.split(str.encode("\n"))[9].split(str.encode(": "))[1]).decode("utf-8")
    except Exception as e:
        return -1, "N/A"


class FrameStream:
    def __init__(self):
        stream = VideoGear(source='test.mp4').start()  # Open any video stream
        server = NetGear()  # Define netgear server with default settings


def send_to_stream(frame):
    server.send(frame)
