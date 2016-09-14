'''
****** Laser Gesture Recognition (v.1.0) ******
Author: Ashish Derhgawen
Website: http://ashishrd.blogspot.com
E-mail: ashish.derhgawen@gmail.com
=====================================

Keys:
-----
    q     - exit
    b     - show/hide binary image
    r     - record new gesture
    e     - enable/disable gesture recognition
'''

import cv2
import numpy as np
import math
#import matplotlib.pyplot as plt
from dtw import dtw

class App(object):
    def __init__(self, video_src): # Initialization code
      self.camera = cv2.VideoCapture(video_src)
      cv2.namedWindow('Cam')
      self.kernel = np.ones((3,3),np.uint8)
      self.color1 = np.array([0,255,0]) # Green
      self.color2 = np.array([0,255,255]) # Yellow
      self.color3 = np.array([0,0,255]) # Red
      self.tracking = False
      self.pts  = [] # For stroring laser path
      self.pause_counter = 0 # Counts frames when laser is not detected
      self.record = False
      self.gestures = []
      self.show_binary = False
      self.enabled = False #  Gesture recognition disabled by default

      self.h_min = 0
      self.h_max = 255
      self.s_min = 0
      self.s_max = 93
      self.v_min = 225
      self.v_max = 255

      cv2.createTrackbar('h_min', 'Cam', self.h_min, 255,self.h_min_slider)
      cv2.createTrackbar('h_max', 'Cam', self.h_max, 255,self.h_max_slider)
      cv2.createTrackbar('s_min', 'Cam', self.s_min, 255,self.s_min_slider)
      cv2.createTrackbar('s_max', 'Cam', self.s_max, 255,self.s_max_slider)
      cv2.createTrackbar('v_min', 'Cam', self.v_min, 255,self.v_min_slider)
      cv2.createTrackbar('v_max', 'Cam', self.v_max, 255,self.v_max_slider)


    def h_min_slider(self, val):
        self.h_min = val

    def h_max_slider(self, val):
        self.h_max = val

    def s_min_slider(self, val):
        self.s_min = val

    def s_max_slider(self, val):
        self.s_max = val

    def v_min_slider(self, val):
        self.v_min = val

    def v_max_slider(self, val):
        self.v_max = val

    def normList(self, L, normalizeTo=1):
        # normalize values of a list to make its max = normalizeTo

        vMax = max(L)
        return [ x/(vMax*1.0)*normalizeTo for x in L]

    # Captures a single image from the camera and returns it in PIL format
    def get_image(self):
        retval, im = self.camera.read()
        return im

    def recognize_gesture(self,img):
      # Find centroid of gesture
        num = len(self.pts)
        x = [p[0] for p in self.pts]
        y = [p[1] for p in self.pts]
        centroid = (sum(x) / num, sum(y) / num)

        # Calculate the distance of each point from the centroid
        dst = []
        for pt in self.pts:
            distance = math.sqrt((pt[0]-centroid[0])**2 + (pt[1]-centroid[1])**2)
            dst.append(distance)
            cv2.circle(img,(pt[0],pt[1]),2,self.color3,2)

        cv2.line(img,(centroid[0]-10,centroid[1]),(centroid[0]+10,centroid[1]),self.color2,2)
        cv2.line(img,(centroid[0],centroid[1]-10),(centroid[0],centroid[1]+10),self.color2,2)

        # Normalize distances
        dst = self.normList(dst)

        if self.record == True:
            gesture_name = raw_input('Enter name for gesture:')
            self.gestures.append([gesture_name, dst])
            print '- Recorded -\n'
            self.record = False
        else:
            gestures = [g[1] for g in self.gestures]
            c = 0
            min = 100
            best_match = ''

            # Calculate similarity using Dynamic Time Warping (DTW)
            for gesture in gestures:
                dist,cost,path = dtw(dst,gesture)
                if dist < min:
                    best_match = self.gestures[c][0], dist
                    min = dist
                c = c + 1

            print best_match
            print '------------------'


        #plt.plot(dst)
        #plt.show()
        return img


    def run(self):
        while True:
            img = self.get_image()
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

            lower = np.array([self.h_min,self.s_min,self.v_min])
            upper = np.array([self.h_max,self.s_max,self.v_max])
            binary = cv2.inRange(hsv,lower,upper)
            binary = cv2.dilate(binary,self.kernel,iterations = 1)
            binary = cv2.erode(binary,self.kernel,iterations = 1)

            if self.show_binary:
                cv2.imshow('binary',binary)

            if self.enabled:
                contours = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]

                # Find largest contour
                max_area = 10
                best_cnt = None
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > max_area:
                        max_area = area
                        best_cnt = cnt

                # Find centroid of best_cnt and draw circle around it
                if best_cnt != None:
                    M = cv2.moments(best_cnt)
                    cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                    cv2.circle(img,(cx,cy),15,self.color1,2)
                    self.tracking = True
                    self.pts.append([cx,cy])

                    self.pause_counter = 0
                else:
                    if self.tracking == True:
                        self.pause_counter = self.pause_counter + 1
                        if self.pause_counter > 15 and len(self.pts) > 10:
                            gesture = self.recognize_gesture(img)
                            self.pts = []
                            self.tracking = False
                            cv2.imshow('Gesture',img)
                            self.pause_counter = 0


            cv2.putText(img, "Recognition enabled: " + str(self.enabled), (5,20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,255,255))
            cv2.putText(img, "Show binary: " + str(self.show_binary), (5,40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,255,255))
            if self.record:
                cv2.putText(img, "** RECORDING **", (img.shape[:2][1]-150,20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,255))

            cv2.imshow('Cam',img);

            ch = cv2.waitKey(1) & 0xFF

            if ch == ord('q'):
                break
            elif ch == ord('r'):
                self.record = True
                print 'Recording....\n'
            elif ch == ord('b'):
                self.show_binary = not self.show_binary
                if self.show_binary == False:
                    cv2.destroyWindow('binary')
            elif ch == ord('e'):
                self.enabled = not self.enabled
                print 'Gesture recognition enabled:',self.enabled

        cv2.destroyAllWindows()
        del(self.camera)

if __name__ == '__main__':
    video_src = 0
    App(0).run()
