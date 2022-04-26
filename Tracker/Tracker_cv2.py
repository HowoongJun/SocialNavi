import copy
import cv2
import os
import json
import numpy as np

class CTrackerCv2():
    def __init__(self):

        tracker_type = {
            #'boost': cv2.TrackerBoosting_create,
            'mil': cv2.TrackerMIL_create,
            'kcf': cv2.TrackerKCF_create,
            'csrt': cv2.TrackerCSRT_create,
            'tld': cv2.TrackerTLD_create,
            'medianflow': cv2.TrackerMedianFlow_create,
            #'goturn': cv2.TrackerGOTURN_create,
            'mosse': cv2.TrackerMOSSE_create
        }
        method = 'csrt'
        self.trackers = cv2.MultiTracker_create()
        self.gen_tracker = tracker_type[method]

    def draw_bbox(self, rgb, bboxes, color):
        for bbox in bboxes:
            x, y, w, h = bbox
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            cv2.rectangle(rgb,(x,y),(x+w,y+h),color,2)
        return rgb

    def reset_tracker(self, rgb, bboxes):
        trackers = cv2.MultiTracker_create()
        for bbox in bboxes:
            tracker = self.gen_tracker()
            trackers.add(tracker, rgb, tuple(bbox))
        return trackers

    def update_tracker(self, trackers, rgb):
        success, boxes = trackers.update(rgb)
        return success, boxes
    
    def set_pre_num_inst(self, value):
        self.pre_num_inst = value

    def get_num_inst(self):
        return self.num_inst

    def track(self, results):
        self.bboxes = []
        # Tracking
        for vDetectedResult in results:
            self.bboxes.append([vDetectedResult.xmin, 
                          vDetectedResult.ymin, 
                          vDetectedResult.xmax, 
                          vDetectedResult.ymax])
    
        self.num_inst = len(results)
        
    def visualize(self, rgb):
        if self.num_inst != self.pre_num_inst:
            self.trackers = self.reset_tracker(rgb, self.bboxes)
            rgb = self.draw_bbox(rgb, self.bboxes, (0, 255, 0))
        else:
            success, self.bboxes = self.update_tracker(self.trackers, rgb)
            if success:
                rgb = self.draw_bbox(rgb, self.bboxes, (0, 0, 255))
            else:
                rgb = self.draw_bbox(rgb, self.bboxes, (255, 0, 0))
        return rgb