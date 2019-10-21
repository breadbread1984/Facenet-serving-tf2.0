#!/usr/bin/python3

from os import listdir;
from os.path import join, exists, isdir;
import pickle;
import numpy as np;
import cv2;
import tensorflow as tf;
from MTCNN import Detector;
from FaceNet import Encoder;

class Recognizer(object):

    def __init__(self, model_path = 'models'):

        self.detector = Detector(model_path);
        self.encoder = Encoder(model_path);
        self.knn = None;
        self.ids = None;

    def load(self):

        self.knn = cv2.ml.KNearest_create();
        self.knn.load('knn.xml');
        with open('ids.pkl', 'rb') as f:
            self.ids = pickle.loads(f.read());

    def loadFaceDB(self, directory = 'facedb', margin = 10):

        if False == exists(directory) or False == isdir(directory):
            print("invalid directory");
            return False;
        self.ids = dict();
        count = 0;
        features = tf.zeros((0,128), dtype = tf.float32);
        labels = tf.zeros((0,1), dtype = tf.int32);
        for f in listdir(directory):
            if isdir(join(directory,f)):
                imgs = list();
                label = count;
                self.ids[label] = f;
                count += 1;
                # only visit directory under given directory
                for img in listdir(join(directory,f)):
                    if False == isdir(join(directory,f,img)):
                        # visit every image under directory
                        image = cv2.imread(join(directory,f,img));
                        if image is None:
                            print("can't open file " + join(directory,f,img));
                            continue;
                        rectangles = self.detector.detect(image);
                        if rectangles.shape[0] != 1:
                            print("can't detect single face in image " + join(directory,f,img));
                            continue;
                        print("processing image " + join(directory, f, img));
                        # crop square from facial area
                        upperleft = rectangles[0,0:2];
                        downright = rectangles[0,2:4];
                        wh = downright - upperleft;
                        length = tf.math.reduce_max(wh, axis = -1).numpy();
                        center = (upperleft + downright) // 2;
                        upperleft = center - tf.constant([length,length], dtype = tf.float32) // 2 - margin // 2;
                        downright = upperleft + tf.constant([length,length], dtype = tf.float32) + margin // 2;
                        face = image[int(upperleft[1]):int(downright[1]),int(upperleft[0]):int(downright[0]),:];
                        imgs.append(face);
                feature = self.encoder.encode(imgs);
                label = tf.tile(tf.constant([[label]], dtype = tf.int32), (feature.shape[0],1));
                features = tf.concat([features,feature],axis = 0);
                labels = tf.concat([labels, label], axis = 0);
        features = features.numpy(); # features.shape = (n, 8631)
        labels = labels.numpy(); # labels.shape = (n, 1)
        # train KD-tree
        self.knn = cv2.ml.KNearest_create();
        self.knn.setIsClassifier(True);
        #self.knn.setAlgorithmType(cv2.BRUTE_FORCE);
        self.knn.setDefaultK(1);
        self.knn.train(features, cv2.ml.ROW_SAMPLE, labels);
        self.knn.save('knn.xml');
        # save ids
        with open('ids.pkl', 'wb') as f:
            f.write(pickle.dumps(self.ids));
        return True;

    def recognize(self, image, margin = 10):

        assert image is not None;
        if self.knn is None or self.ids is None:
            print('call load() or loadFaceDB() before this method!');
            return;
        rectangles = self.detector.detect(image);
        upperleft = rectangles[...,0:2];
        downright = rectangles[...,2:4];
        wh = downright - upperleft;
        length = tf.math.reduce_max(wh, axis = -1);
        center = (upperleft + downright) // 2;
        upperleft = center - tf.stack([length,length], axis = -1) // 2 - margin // 2;
        downright = upperleft + tf.stack([length, length], axis = -1) + margin // 2;
        upperleft = tf.reverse(upperleft, axis = [1]); # in h,w order
        downright = tf.reverse(downright, axis = [1]); # in h,w order
        boxes = tf.concat([upperleft, downright], axis = -1) / tf.cast(tf.tile(image.shape[0:2], (2,)), dtype = tf.float32);
        image = tf.expand_dims(tf.cast(image, dtype = tf.float32), axis = 0);
        faces = tf.image.crop_and_resize(image, boxes, tf.zeros((boxes.shape[0],), dtype = tf.int32),(224,224));
        faces = [face.numpy() for face in faces];
        features = self.encoder.encode(faces);
        ret, results, neighbours, dist = self.knn.findNearest(features.numpy(), k = 3);
        retval = list();
        for i in range(features.shape[0]):
            # for each input feature
            rect = rectangles[i];
            labels, counts = np.unique(neighbours[i].astype('uint32'), return_counts = True);
            label = labels[np.argmax(counts)];
            if np.min(dist[i,neighbours[i] == label]) <= 0.9: retval.append((rect,self.ids[label]));
            else: retval.append((rect,""));
        return retval;

    def visualize(self, img, targets):

        for target in targets:
            rect, label = target;
            cv2.rectangle(img, tuple(rect[0:2]), tuple(rect[2:4]), (0,0,255,), 2);
            cv2.putText(img, (label if label != "" else "???"), tuple(rect[0:2]), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0,255,0), 3, 8);
        return img;

if __name__ == "__main__":

    assert tf.executing_eagerly() == True;
    import sys;
    if len(sys.argv) != 2:
        print("Usage: " + sys.argv[0] + " <video>");
        exit(1);
    recognizer = Recognizer();
    recognizer.loadFaceDB();
    cap = cv2.VideoCapture(sys.argv[1]);
    if cap is None:
        print("invalid video!");
        exit(0);
    while True:
        ret, img = cap.read();
        if ret == False: break;
        targets = recognizer.recognize(img);
        img = recognizer.visualize(img, targets);
        cv2.imshow('detection', img);
        #k = cv2.waitKey(int(cap.get(cv2.CAP_PROP_FPS)));
        k = cv2.waitKey(1);
        if k == 'q': break;


