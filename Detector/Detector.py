###
#
#       @Brief          Detector.py
#       @Details        Main framework for SocialNavi detector
#       @Org            Robot Learning Lab(https://rllab.snu.ac.kr), Seoul National University
#       @Author         Howoong Jun (howoong.jun@rllab.snu.ac.kr)
#       @Date           Apr. 15, 2022
#       @Version        v0.1
#
###

from common.Log import DebugPrint
from common.al import *
import torch, math
import numpy as np
from enum import IntEnum
from Detector.DetectedData import CDetectedData

class eSetCmd(IntEnum):
    eSetCmd_IMAGE = 1
    eSetCmd_CLASS = 2
    
class CDetector(CSocialNaviCore):
    def __init__(self):
        self.__oObjectDetect = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.__oObjectDetect.classes = np.array([0])
        self.__oImage = None
        self.__AVERAGE_HEIHGT = 1.735
        self.__FOVY = math.pi / 3

    def __del__(self):
        print("Destructor")

    def Open(self):
        print("Open")
    
    def Close(self):
        print("Close")

    def Write(self):
        print("Write")

    def Read(self):
        oResults = self.__oObjectDetect(self.__oImage)
        vPixMeterRatio = []
        vDetectedResults = []
        oParsedResults = oResults.pandas().xyxy[0]
        for elements in range(0, len(oParsedResults)):
            oDetected = CDetectedData()
            oDetected.data_class = oParsedResults['class'][elements]
            oDetected.confidence = oParsedResults['confidence'][elements]
            oDetected.xmin = oParsedResults['xmin'][elements]
            oDetected.ymin = oParsedResults['ymin'][elements]
            oDetected.xmax = oParsedResults['xmax'][elements]
            oDetected.ymax = oParsedResults['ymax'][elements]
            oDetected.obj_id = elements
            vPixMeterRatio.append(self.__AVERAGE_HEIHGT/(oDetected.ymax - oDetected.ymin))
            oDetected.depth = self.__AVERAGE_HEIHGT / (oDetected.ymax - oDetected.ymin) * self.__oImage.shape[0] / math.tan(self.__FOVY)
            vDetectedResults.append(oDetected)
        return vDetectedResults

    def Control(self, eSet:int, Value):
        eControlCmd = eSetCmd(eSet)
        if(eControlCmd == eSetCmd.eSetCmd_IMAGE):
            self.__oImage = np.array(Value)
        elif(eControlCmd == eSetCmd.eSetCmd_CLASS):
            self.__oObjectDetect.classes  = np.array(Value)

    def Reset(self):
        self.__oImage = None