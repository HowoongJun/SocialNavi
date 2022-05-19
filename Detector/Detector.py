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

DEPTH_OFFSET = 1.0 #m
LIDAR_LEFT_IDX = 182
LIDAR_RIGHT_IDX = 901
LIDAR_RESOLUTION = 0.25 #deg
LIDAR_MIN_VAL = 0.1

class eSetCmd(IntEnum):
    eSetCmd_IMAGE = 1
    eSetCmd_CLASS = 2
    eSetCmd_LIDAR = 3
    
class CDetector(CSocialNaviCore):
    def __init__(self):
        self.__oObjectDetect = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.__oObjectDetect.classes = np.array([0])
        self.__oImage = None
        self.__oLidar = None
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
        vLidarVal = self.__LidarProcess()
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
            fEstmDepth = self.__AVERAGE_HEIHGT / (oDetected.ymax - oDetected.ymin) * self.__oImage.shape[0] / math.tan(self.__FOVY)
            fDepth = self.__FindDepth(fEstmDepth, vLidarVal)
            if(fDepth is not None): fEstmDepth = fDepth
            oDetected.depth = fEstmDepth
            vDetectedResults.append(oDetected)
        return vDetectedResults

    def Control(self, eSet:int, Value):
        eControlCmd = eSetCmd(eSet)
        if(eControlCmd == eSetCmd.eSetCmd_IMAGE):
            self.__oImage = np.array(Value)
        elif(eControlCmd == eSetCmd.eSetCmd_CLASS):
            self.__oObjectDetect.classes  = np.array(Value)
        elif(eControlCmd == eSetCmd.eSetCmd_LIDAR):
            self.__oLidar = np.array(Value)

    def Reset(self):
        self.__oImage = None

    def __LidarProcess(self):
        if(len(self.__oLidar) != 1081):
            DebugPrint().error("Wrong LiDAR data!")
            return None
        vDepth = []
        for lidar_idx in range(LIDAR_LEFT_IDX, LIDAR_RIGHT_IDX):
            if(math.isinf(self.__oLidar[lidar_idx]) or self.__oLidar[lidar_idx] < LIDAR_MIN_VAL):
                vDepth.append(-1)
                continue
            fDepth = self.__oLidar[lidar_idx] * math.sin((lidar_idx - LIDAR_LEFT_IDX - 1) * LIDAR_RESOLUTION * math.pi / 180)
            vDepth.append(fDepth)
        return vDepth

    def __FindDepth(self, estm_depth, det_depth):
        vDepth = []
        uCount = 0
        for depth in det_depth:
            if(depth > estm_depth + DEPTH_OFFSET or depth < estm_depth - DEPTH_OFFSET):
                uCount = 0
                vDepth = []
                continue
            vDepth.append(depth)
            uCount += 1
            if(uCount > 5): break

        return sum(vDepth) / len(depth)