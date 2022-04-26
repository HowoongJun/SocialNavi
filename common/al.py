###
#
#       @Brief          al.py
#       @Details        Abstraction layer for SocialNavi
#       @Org            Robot Learning Lab(https://rllab.snu.ac.kr), Seoul National University
#       @Author         Howoong Jun (howoong.jun@rllab.snu.ac.kr)
#       @Date           Apr. 15, 2022
#       @Version        v0.1
#
###

from abc import *
from enum import IntEnum

class eSetCmd(IntEnum):
    eSetCmd_NONE = 1
    eSetCmd_WIDTH = 2
    eSetCmd_HEIGHT = 3
    eSetCmd_IMAGE = 4

class CSocialNaviCore(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        print("SocialNavi Core Constructor")
        
    @abstractmethod
    def __del__(self):
        print("SocialNavi Core Destructor")

    @abstractmethod
    def Open(self):
        print("SocialNavi Core Open")
    
    @abstractmethod
    def Close(self):
        print("SocialNavi Core Close")
    
    @abstractmethod
    def Write(self):
        print("SocialNavi Core Write")

    @abstractmethod
    def Read(self):
        print("SocialNavi Core Read")

    @abstractmethod
    def Control(self):
        print("SocialNavi Core Control")

    @abstractmethod
    def Reset(self):
        print("SocialNavi Core Reset")
        