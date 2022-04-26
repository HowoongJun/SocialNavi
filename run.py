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
from Tracker.Tracker_cv2 import CTrackerCv2

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
    if(args.tracking == 'hogun'):
        oTracker = CTrackerCv2()
    else:
        oTracker = PedestrianTracker()

    uPreNumInst = 0

    for strImgPath in vImgFileList:
        DebugPrint().info("Processing " + strImgPath)
        oImg = cv2.imread(os.path.join(strImgDir, strImgPath))
        
        # Detection
        oDetector.Control(eSetCmd.eSetCmd_IMAGE, oImg)
        oResults = oDetector.Read()
        oDetector.Reset()
        
        # Tracking
        oTracker.set_pre_num_inst(uPreNumInst)
        oTracker.track(oResults)
        oImgRender = oTracker.visualize(oImg)
        cv2.imwrite(os.path.join(strResultDir, strImgPath), oImgRender)
        uPreNumInst = oTracker.get_num_inst()