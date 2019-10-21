#!/usr/bin/python3

import numpy as np;
from HeadPoseEstimator import HeadPoseEstimator;

class AntiSpoofing(object):

  def __init__(self, model_path = "models"):

    self.db = dict();
    self.estimator = HeadPoseEstimator(model_path);

  def reset(self):

    self.db = dict();

  def update(self, img, face_rect, iid):
    
    _, _, euler_angle = self.estimator.estimate(img, face_rect);
    if iid not in self.db:
      self.db[iid] = {"pitch": list(), "yaw": list(), "roll": list()};
    self.db[iid]["pitch"].append(euler_angle[0]);
    self.db[iid]["yaw"].append(euler_angle[1]);
    self.db[iid]["roll"].append(euler_angle[2]);

  def by_nod(self, iid):

    if iid not in self.db:
      self.db[iid] = {"pitch": list(), "yaw": list(), "roll": list()};
    pitch = np.array(self.db[iid]["pitch"]);
    return np.std(pitch) > 6;

  def by_shake(self, iid):

    if iid not in self.db:
      self.db[iid] = {"pitch": list(), "yaw": list(), "roll": list()};
    yaw = np.array(self.db[iid]["yaw"]);
    return np.std(yaw) > 26;

if __name__ == "__main__":

  import sys;
  import cv2;
  from MTCNN import Detector;
  if len(sys.argv) != 2:
    print("Usage: "+ sys.argv[0] + " <video>");
    exit(1);
  detector = Detector();
  antispoofing = AntiSpoofing();
  cap = cv2.VideoCapture(sys.argv[1]);
  if cap is None:
    print("invalid video!");
    exit(1);
  while True:
    ret, img = cap.read();
    img = cv2.resize(img, (640,360));
    if ret == False: break;
    rectangles = detector.detect(img);
    if rectangles.shape[0]:
      antispoofing.update(img, rectangles[0], 0);
    else:
      print("no face detected!");
    cv2.imshow('show', img);
    print("nod = " + ("true" if antispoofing.by_nod(0) else "false") + " shake = " + ("true" if antispoofing.by_shake(0) else "false"));
    cv2.waitKey(1);

