# AI495 Final Project
# Authors: Charles Cheng and Sushma Chandra

# Import libraries
import numpy as np # pip install numpy 
import cv2 as cv # pip install opencv-python
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TKAgg')

class FallDetector:

    def __init__(self, filename, fps, aspect_ratio):
        self.filename = filename
        self.fps = 15
        self.resolution = (aspect_ratio[0]*32, aspect_ratio[1]*32)
        self.img_size = self.resolution[0]*self.resolution[1]
        self.motion_mask = np.zeros(self.resolution, dtype=np.uint8)
        self.color_video, self.gray_video, self.thres_video = [], [], []
        self.read_video(filename, captured_fps=fps, blur_size=(7,7))#, frame_limit=500)
        self.id_counter = 0
        self.obj_hist = {}
        self.motion_objs_stamped = []
        self.fast_objs = []
        self.show_all=False
        self.show_fast=True
        self.show_mask=False

    def read_video(self, filename, captured_fps=30, blur_size=5, frame_limit=10000):
        print("Processing", filename, "...")
        cap = cv.VideoCapture(filename)
        # video_length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.color_video = []
        # self.color_video, self.gray_video, self.thres_video = [], [], [] # TODO: fix array size 
        curr_frame = frame_count = 0
        stride = captured_fps//self.fps
        while cap.isOpened() and curr_frame < frame_limit:
            if (curr_frame % stride) == 0:
                ret, frame = cap.read()
                if not ret:
                    break
                print("Frame: ", frame_count, end='\r')
                frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                # frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                self.color_video.append(np.array(cv.resize(cv.flip(frame_hsv, -1), \
                                                           (self.resolution[1], self.resolution[0])), dtype=np.float32))
                # self.gray_video.append(np.array(cv.GaussianBlur(cv.resize(cv.flip(frame_gray, -1), \
                #                                                           (self.resolution[1], self.resolution[0])), \
                #                                                 (blur_size, blur_size), 0, 0), np.float32))
                # self.thres_video.append(np.array(cv.threshold(cv.GaussianBlur(cv.resize(cv.flip(frame_gray, -1), \
                #                                                                         (self.resolution[1], self.resolution[0])), \
                #                                                               (blur_size, blur_size), 0, 0), \
                #                                               0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]))
                frame_count += 1
            else:
                ret = cap.grab()
                if not ret:
                    break
            curr_frame += 1
        cap.release()
        print(frame_count, "frames read")

    def detect_falling(self, channel=0, threshold=0.2, open_size=5, close_size=7, min_area=0.1, wH=0.33, wS=0.33, wV=0.33, min_speed=10.0, similarity=0.04, post_morph=False):
        for frame_idx in range(len(self.color_video)-1):
            time = frame_idx/self.fps
            print("Time (s):", time, end='\r')
            static_motion_mask = self.detect_motion(self.color_video[0][:,:,channel], self.color_video[frame_idx+1][:,:,channel], threshold, open_size, close_size)
            dyn_motion_mask = self.detect_motion(self.color_video[frame_idx][:,:,channel], self.color_video[frame_idx+1][:,:,channel], threshold, open_size, close_size)
            self.motion_mask = np.logical_and(static_motion_mask, dyn_motion_mask).astype(np.uint8)
            if post_morph:
                self.motion_mask = self.closing(self.opening(self.motion_mask, kernel_size=(open_size, open_size)), kernel_size=(close_size, close_size))
            self.motion_objs = cv.connectedComponentsWithStats(self.motion_mask, 4, cv.CV_32S)[2][1:]
            self.motion_filter(min_area)
            self.fast_objs = []
            self.motion_objs_stamped = []
            self.update_obj_hist(frame_idx+1, time, wH, wS, wV, min_speed, similarity)
            self.show_motion(frame_idx+1)
        cv.destroyAllWindows()

    def detect_motion(self, prev_frame, curr_frame, threshold, open_size, close_size):
        mask = (np.abs(curr_frame-prev_frame)/255.0 > threshold).astype(np.uint8)
        return self.closing(self.opening(mask, kernel_size=(open_size, open_size)), kernel_size=(close_size, close_size))

    def closing(self, img, kernel_size):
        kernel = np.ones(kernel_size, np.uint8)
        return cv.erode(cv.dilate(img, kernel, iterations=1), kernel, iterations=1)

    def opening(self, img, kernel_size):
        kernel = np.ones(kernel_size, np.uint8)
        return cv.dilate(cv.erode(img, kernel, iterations=1), kernel, iterations=1)

    def motion_filter(self, min_area):
        self.motion_objs = [obj for obj in self.motion_objs if obj[2]*obj[3] > min_area*self.img_size]

    def update_obj_hist(self, frame_idx, time, wH, wS, wV, min_speed, similarity):
        unique = {}
        for obj in self.motion_objs:
            raw_score = self.similiarity_score(frame_idx, obj, wH, wS, wV)
            score = round(self.round_nearest(raw_score, similarity), 2)
            if score not in unique:
                if score in self.obj_hist:
                    new_obj = (time, self.obj_hist[score][-1][1], obj, score)
                    self.obj_hist[score].append(new_obj)
                    self.speed_filter(self.obj_hist[score][-2], new_obj, min_speed)
                else:
                    self.obj_hist[score] = [(time, self.id_counter, obj, score)]
                    self.id_counter += 1
                unique[score] = True
            self.motion_objs_stamped.append((time, 0, obj, score))

    def calculate_average(self, frame_idx, tracked_obj, channel):
        sum = 0
        area = 0
        x = np.linspace(0, tracked_obj[3], tracked_obj[3], endpoint=False)
        wx = self.normal_dist(x, np.mean(x), np.std(x)*1.5)
        y = np.linspace(0, tracked_obj[2], tracked_obj[2], endpoint=False)
        wy = self.normal_dist(y, np.mean(y), np.std(y)*1.5)
        for row_pixel in range(tracked_obj[3]):
            for col_pixel in range(tracked_obj[2]):
                wxy = wx[row_pixel]*wy[col_pixel]
                area += wxy
                sum += wxy*self.color_video[frame_idx][tracked_obj[1]+row_pixel, tracked_obj[0]+col_pixel, channel]
        return sum/area
    
    def similiarity_score(self, frame_idx, tracked_obj, wH, wS, wV):
        return wH*self.calculate_average(frame_idx, tracked_obj, 0)/179.0+wS*self.calculate_average(frame_idx, tracked_obj, 1)/255.0+wV*self.calculate_average(frame_idx, tracked_obj, 2)/255.0

    def speed_filter(self, prev_obj, curr_obj, min_speed):
        prev_centroid = np.array([prev_obj[2][1]+prev_obj[2][3]/2, prev_obj[2][0]+prev_obj[2][2]/2])
        curr_centroids = np.array([curr_obj[2][1]+curr_obj[2][3]/2, curr_obj[2][0]+curr_obj[2][2]/2])
        if np.linalg.norm(curr_centroids-prev_centroid)/(curr_obj[0]-prev_obj[0]) > min_speed:
            self.fast_objs.append(curr_obj)

    def show_motion(self, frame_idx):
        img = self.color_video[frame_idx].astype(np.uint8)
        if self.show_all:
            for obj in self.motion_objs_stamped:
                cv.putText(img, 'Score ' + str(obj[3]), (obj[2][0], obj[2][1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, 2)
                cv.rectangle(img, (obj[2][0], obj[2][1]), (obj[2][0]+obj[2][2]-1, obj[2][1]+obj[2][3]-1), (0, 255, 255), 1) # red
        if self.show_fast:
            for obj in self.fast_objs:
                cv.putText(img, 'Falling!', (obj[2][0], obj[2][1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (63,255,255), 1, 2)
                cv.rectangle(img, (obj[2][0], obj[2][1]), (obj[2][0]+obj[2][2]-1, obj[2][1]+obj[2][3]-1), (63, 255, 255), 1) # green
        if self.show_mask:
            fg = cv.bitwise_or(img, img, mask=self.motion_mask)
            cv.imshow('motion mask', cv.cvtColor(fg, cv.COLOR_HSV2BGR))
        cv.imshow('object detection', cv.cvtColor(img, cv.COLOR_HSV2BGR))
        cv.waitKey(0)
     
    def round_nearest(self, num, a):
        return round(num/a)*a

    def normal_dist(self, x, avg, std):
        ndist = np.pi*std*np.exp(-0.5*((x-avg)/std)**2)
        return ndist/max(ndist)

if __name__ == "__main__":

    falling_objs = FallDetector("falling_objects.mov", 30, (16, 9))
    falling_objs.show_mask = True
    falling_objs.show_all = True
    falling_objs.detect_falling(channel=2, threshold=0.12, open_size=3, close_size=9, min_area=0.0005, wH=0.15, wS=0.25, wV=0.6, min_speed=200.0, similarity=0.04, post_morph=True)

    # falling_simult = FallDetector("falling_simult.mov", 30, (16, 9))
    # falling_simult.detect_falling(channel=2, threshold=0.15, open_size=3, close_size=9, min_area=0.005, wH=0.2, wS=0.3, wV=0.5, min_speed=200.0, similarity=0.04, post_morph=True)
    
    # nonfalling_motion = FallDetector("nonfalling_motion.mov", 30, (16, 9))
    # nonfalling_motion.detect_falling(channel=1, threshold=0.1, open_size=3, close_size=9, min_area=0.005, wH=0.2, wS=0.3, wV=0.5, min_speed=300.0, similarity=0.05, post_morph=False)

# TODO: try blurring 
