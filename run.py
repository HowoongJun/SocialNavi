###
#
#       @Brief          run.py
#       @Details        Run SocialNavi
#       @Org            Robot Learning Lab(https://rllab.snu.ac.kr), Seoul National University
#       @Author         Howoong Jun (howoong.jun@rllab.snu.ac.kr)
#       @Date           Apr. 15, 2022
#       @Version        v0.1
#
###

from Detector.Detector import CDetector as CDetector
from Detector.Detector import eSetCmd as eSetCmd
import argparse
import cv2, os
from common.Log import DebugPrint
from Tracker.Tracker import PedestrianTracker

parser = argparse.ArgumentParser(description="Social Navi Pedestrian Detection and Tracking")
parser.add_argument('--dir', type=str, dest='data_dir',
                    help="Data directory")
parser.add_argument('--track', type=str, default="jeongho", dest='tracking',
                    help="Tracking method (hogun/jeongho)")
args = parser.parse_args()

if __name__ == "__main__":
    strResultDir = os.path.join(args.data_dir, "results")
    strImgDir = os.path.join(args.data_dir, "image_raw")
    vImgFileList = os.listdir(strImgDir)
    vImgFileList.sort()

    oDetector = CDetector()
    oTracker = PedestrianTracker()

    for strImgPath in vImgFileList:
        DebugPrint().info("Processing " + strImgPath)
        oImg = cv2.imread(os.path.join(strImgDir, strImgPath))
        
        # Detection
        oDetector.Control(eSetCmd.eSetCmd_IMAGE, oImg)
        oResults = oDetector.Read()
        oDetector.Reset()
        
        vBBox = []
        # Tracking
        for vDetectedResult in oResults:
            vBBox.append([vDetectedResult.xmin, 
                          vDetectedResult.ymin, 
                          vDetectedResult.xmax, 
                          vDetectedResult.ymax])
        oTracker.track(vBBox)
        oImgRender = oTracker.visualize(oImg)
        cv2.imwrite(os.path.join(strResultDir, strImgPath), oImgRender)