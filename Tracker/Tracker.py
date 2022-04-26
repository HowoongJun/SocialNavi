import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import json
import copy

def get_center_pos(pos):
    x1, y1, x2, y2 = pos
    ctr_x = (x1+x2)/2
    ctr_y = (y1+y2)/2
    return [ctr_x, ctr_y]

def get_dist(pos1, pos2):
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

class Pedestrian():
    def __init__(self, ped_id, init_pos):
        self.pos = init_pos
        self.ctr_pos = get_center_pos(init_pos)
        self.exist_t = 0 # or 1
        self.undetected_t = 0
        self.id = ped_id
    
    def update_pos(self, pos):
        self.pos = pos
        self.ctr_pos = get_center_pos(pos)

class PedestrianTracker():
    def __init__(self):

        # pedestrians currently being tracked
        self.peds_tracked = {}
        # pedestrians currently being tracked, but failed to be detected
        self.peds_tracked_unmatched = []

        # next id number to be used for a new ped.
        self.curr_max_id = 0
        # the number of pedestrians currently tracked
        self.curr_ped_num = 0
        # the ids of pedestrians currently tracked
        self.curr_ped_ids = []
        # dummy value used in distance matrix
        self.dummy_val = 1e6

        # maximum pixel distance for matching two boxes
        self.DIST_THRESHOLD = 200.0
        # maximum length of time step for not being detected
        self.NOISE_THRESHOLD = 3

    def add_new_ped(self, init_pos):
        """
        adding a newly detected pedestrian
        """
        self.peds_tracked.update({
            self.curr_max_id: Pedestrian(self.curr_max_id, init_pos)
            })
        self.curr_ped_ids.append(self.curr_max_id)
        self.curr_max_id += 1
        self.curr_ped_num += 1
        

    def update_matched_ped(self, ped_id, new_pos):
        """
        update the positions of pedestrians who are being successfuly tracked.
        """
        if ped_id not in self.curr_ped_ids:
            print("wrong udpate")
            raise ValueError
        self.peds_tracked[ped_id].update_pos(new_pos)
        self.peds_tracked[ped_id].exist_t += 1

    def update_unmatched_ped(self, ped_id):
        """
        update the status of pedestrians who failed to be matched with 
        newly detected pedestrians.
        """
        self.peds_tracked[ped_id].undetected_t += 1
        self.peds_tracked[ped_id].exist_t += 1
        if self.peds_tracked[ped_id].undetected_t == 1:
            self.peds_tracked_unmatched.append(ped_id)

    def remove_ped(self, ped_id):
        """
        remove the pedestrians who do not exist in the camera frame anymore.
        """
        self.peds_tracked.pop(ped_id)
        if ped_id in self.peds_tracked_unmatched:
            self.peds_tracked_unmatched.remove(ped_id)
        self.curr_ped_num -= 1
        self.curr_ped_ids.remove(ped_id)

    def match_peds(self, detected_ped_poses):
        """
        match the newly detected pedestrians with the past detected pedestrians
        using Hungarian matching algorithm.
        """
        dist_matrix = self.construct_dist_matrix(detected_ped_poses)
        row_ids, col_ids = linear_sum_assignment(cost_matrix=dist_matrix)
        matched_pairs = []
        peds_tracked_unmatched = []
        peds_new_unmatched = []
        for r, c in zip(row_ids, col_ids):
            if r >= self.curr_ped_num:
                peds_new_unmatched.append(c)
                continue

            ped_id = self.curr_ped_ids[r]
            if dist_matrix[r,c] != self.dummy_val:
                matched_pairs.append([ped_id,c])
            else:
                if c >= len(detected_ped_poses): 
                    peds_tracked_unmatched.append(ped_id)
                    continue
                
                peds_tracked_unmatched.append(ped_id)
                peds_new_unmatched.append(c)
        return matched_pairs, peds_tracked_unmatched, peds_new_unmatched

    def construct_dist_matrix(self, detected_ped_poses):
        """
        construct a cost matrix for Hungarian algorithm.
        distance metric = euclidean dist.
        """
        num_detected = len(detected_ped_poses)
        max_num = max(self.curr_ped_num, num_detected)
        dist_matrix = np.full((max_num, max_num), self.dummy_val)
        for i, pos in enumerate(detected_ped_poses):
            detected_ctr_pos = get_center_pos(pos)
            for j in range(self.curr_ped_num):
                ped_id = self.curr_ped_ids[j]
                ped = self.peds_tracked[ped_id]
                dist = get_dist(ped.ctr_pos, detected_ctr_pos)
                dist_matrix[j, i] = dist
        return dist_matrix


    def track(self, detected_peds):
        """
        main function.
        input: list of the box coordinates of detected pedestrians.
        """
        if self.curr_ped_num == 0:
            for ped in detected_peds:
                self.add_new_ped(ped)
            return

        matched_pairs, peds_tracked_unmatched, peds_new_unmatched = self.match_peds(detected_peds)
        for pair in matched_pairs:
            last_pos = self.peds_tracked[pair[0]].ctr_pos
            detected_pos = get_center_pos(detected_peds[pair[1]])
            dist = get_dist(last_pos, detected_pos)
            if dist > self.DIST_THRESHOLD:
                self.update_unmatched_ped(pair[0])
                self.add_new_ped(detected_peds[pair[1]])
            else:
                self.update_matched_ped(pair[0], detected_peds[pair[1]])
            
        for ped_id in peds_tracked_unmatched:
            if self.peds_tracked[ped_id].undetected_t > self.NOISE_THRESHOLD:
                self.remove_ped(ped_id)
            else:
                self.update_unmatched_ped(ped_id)
        
        for c in peds_new_unmatched:
            self.add_new_ped(detected_peds[c])

    def visualize(self, img):
        """
        visualize the id of each pedestrians and their corresponding positions.
        """
        size_ratio = 1
        ori_h, ori_w, ch = img.shape
        img = cv2.resize(img, dsize=(ori_w*size_ratio, ori_h*size_ratio), interpolation=cv2.INTER_AREA)
        
        font_size = 1.1*size_ratio
        font_width = 3*size_ratio
        mg = 10*size_ratio 
        for ped_id in self.curr_ped_ids:
            ped = self.peds_tracked[ped_id]
            h, w = ped.ctr_pos
            int_h, int_w = int(h)*size_ratio, int(w)*size_ratio
            img[int_w-mg:int_w+mg, int_h-mg:int_h+mg] = [255,0,0]

            font_color = (0,0,40)
            cv2.putText(img, "P.%02d" % ped_id, (int_h, int_w), cv2.FONT_HERSHEY_SIMPLEX,
                       font_size, font_color, font_width, cv2.LINE_AA)
        return img
        