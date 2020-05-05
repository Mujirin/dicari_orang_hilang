from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
#import sys
import numpy as np
import mxnet as mx
import os


from scipy import misc
import random
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector
from skimage import transform as trans
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mxnet.contrib.onnx.onnx2mx.import_model import import_model



## coding: utf-8
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
#from sklearn.cluster import KMeans
#import cv2
#import sys
#import numpy as np
#import mxnet as mx
#import os
#import pickle

from os import listdir
from os.path import isfile, join
#from scipy import misc
#import random
#import sklearn
#from sklearn.decomposition import PCA
#from time import sleep
#from easydict import EasyDict as edict
#from mtcnn_detector import MtcnnDetector
#from skimage import transform as trans
#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
#from mxnet.contrib.onnx.onnx2mx.import_model import import_model
#

def get_model(ctx, model):
    image_size = (112, 112)
    # Import ONNX model
    sym, arg_params, aux_params = import_model(model)
    # Define and binds parameters to the network
    model = mx.mod.Module(symbol=sym,
                          context=ctx,
                          label_names=None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


def preprocess(img, bbox=None, landmark=None, **kwargs):
    M = None
    image_size = []
    str_image_size = kwargs.get('image_size', '')
    # Assert input shape
    if len(str_image_size) > 0:
        image_size = [int(x) for x in str_image_size.split(',')]
        if len(image_size) == 1:
            image_size = [image_size[0], image_size[0]]
        assert len(image_size) == 2
        assert image_size[0] == 112
        assert image_size[0] == 112 or image_size[1] == 96

    # Do alignment using landmark points
    if landmark is not None:
        assert len(image_size) == 2
        src = np.array([
          [30.2946, 51.6963],
          [65.5318, 51.5014],
          [48.0252, 71.7366],
          [33.5493, 92.3655],
          [62.7299, 92.2041]], dtype=np.float32)
        if image_size[1] == 112:
            src[:, 0] += 8.0
        dst = landmark.astype(np.float32)
        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]
        assert len(image_size) == 2
        warped = cv2.warpAffine(img, M, (image_size[1],
                                         image_size[0]), borderValue=0.0)
        return warped

    # If no landmark points available,
    # do alignment using bounding box.
    # If no bounding box available use center crop
    if M is None:
        if bbox is None:
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1]*0.0625)
            det[1] = int(img.shape[0]*0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
        bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret


def pre_process(img, bbox=None, landmark=None, **kwargs):
    M = None
    image_size = []
    str_image_size = kwargs.get('image_size', '')
    if len(str_image_size) > 0:
        image_size = [int(x) for x in str_image_size.split(',')]
        if len(image_size) == 1:
            image_size = [image_size[0], image_size[0]]
        assert len(image_size) == 2
        assert image_size[0] == 112
        assert image_size[0] == 112 or image_size[1] == 96
    if landmark is not None:
        assert len(image_size) == 2
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        if image_size[1] == 112:
            src[:, 0] += 8.0
        dst = landmark.astype(np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2, :]

    if M is None:
        if bbox is None:  # use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1] * 0.0625)
            det[1] = int(img.shape[0] * 0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
        bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
    else:  # do align using landmark
        assert len(image_size) == 2
        warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]),
                                borderValue=0.0)
        return warped


def get_input(detector, face_img):
    # Pass input images through face detector
    ret = detector.detect_face(face_img, det_type=0)
    if ret is None:
        return None
    bbox, points = ret
    if bbox.shape[0] == 0:
        return None
    bbox = bbox[0, 0:4]
    points = points[0, :].reshape((2, 5)).T
    # Call preprocess() to generate aligned images
    nimg = preprocess(face_img, bbox, points, image_size='112,112')
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(nimg, (2, 0, 1))
    return aligned


def get_feature(model, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    embedding = model.get_outputs()[0].asnumpy()
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding


def post_process_dummy(img1, img2):
    # Determine and set context
    if len(mx.test_utils.list_gpus()) == 0:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(0)
    # Configure face detector
    det_threshold = [0.6, 0.7, 0.8]
    mtcnn_path = os.path.join(os.path.dirname('__file__'), 'mtcnn-model')
    detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx,
                             num_worker=1, accurate_landmark=True,
                             threshold=det_threshold)

    # Path to ONNX model
    model_name = 'resnet100.onnx'

    # Load ONNX model
    model = get_model(ctx, model_name)

    # Display first image
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.show()

    # Preprocess first image
    pre1 = get_input(detector, img1)
    # Display preprocessed image
    plt.imshow(np.transpose(pre1, (1, 2, 0)))
    plt.show()
    # Get embedding of first image
    out1 = get_feature(model, pre1)

    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.show()

    # Preprocess second image
    pre2 = get_input(detector, img2)
    # Display preprocessed image
    plt.imshow(np.transpose(pre2, (1, 2, 0)))
    plt.show()
    # Get embedding of second image
    out2 = get_feature(model, pre2)

    # Compute squared distance between embeddings
    dist = np.sum(np.square(out1-out2))
    # Compute cosine similarity between embedddings
    sim = np.dot(out1, out2.T)
    # Print predictions
    print('Distance = %f' % (dist))
    print('Similarity = %f' % (sim))
    return {"Distance": dist, "Similarity": sim}


def getModel(model_name='resnet100.onnx'):
        # Determine and set context
    if len(mx.test_utils.list_gpus()) == 0:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(0)
    # Configure face detector
    det_threshold = [0.6, 0.7, 0.8]
    mtcnn_path = os.path.join(os.path.dirname('__file__'), 'mtcnn-model')
    detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx,
                             num_worker=1, accurate_landmark=True,
                             threshold=det_threshold)

    # Path to ONNX model
#    model_name = 'resnet100.onnx'

    # Load ONNX model
    model = get_model(ctx, model_name)
    return model, detector
#img1 = cv2.imread('evan1.jpg')
#img2 = cv2.imread('bambang2.jpg')
##img1 = cv2.imread('bambang1.jpg')
#
## img1 = cv2.imread('c.jpg')
## img2 = cv2.imread('c2.jpg')
#post_process_dummy(img1, img2)


def getSimDist(img1, img2, model, detector):
    # Display first image
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.show()

    # Preprocess first image
    pre1 = get_input(detector, img1)
    # Display preprocessed image
    plt.imshow(np.transpose(pre1, (1, 2, 0)))
    plt.show()
    # Get embedding of first image
    out1 = get_feature(model, pre1)

    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.show()

    # Preprocess second image
    pre2 = get_input(detector, img2)
    # Display preprocessed image
    plt.imshow(np.transpose(pre2, (1, 2, 0)))
    plt.show()
    # Get embedding of second image
    out2 = get_feature(model, pre2)

    # Compute squared distance between embeddings
    dist = np.sum(np.square(out1-out2))
    # Compute cosine similarity between embedddings
    sim = np.dot(out1, out2.T)
    # Print predictions
    print('Distance = %f' % (dist))
    print('Similarity = %f' % (sim))
    return dist, sim

#path_gambar = '/Users/thomas/Documents/UI/LF/cluster/gambar'
path_gambar = '/Users/thomas/Documents/UI/LF/cluster/gambar_cluster_1'
path_group = [path_gambar]
path_picture = []
for i in range(len(path_group)):
    path_i = [
            f for f in listdir(path_group[i])if isfile(
                    join(path_group[i], f))]
    path_picture = path_picture + path_i
img1 = cv2.imread('evan1.jpg')
# model
#model_name = 'resnet100.onnx'
model, detector = getModel(model_name='resnet100.onnx')

dist_group = []
sim_group = []
picture_name = []
for i in range(len(path_picture)):
    if path_picture[i].endswith('.png'):
        img_path = cv2.imread('gambar_cluster_1/' + path_picture[i])
        dist, sim = getSimDist(img1, img_path, model, detector)
        picture_name.append(path_picture[i])
        dist_group.append(dist)
        sim_group.append(sim)



import pandas as pd
df = pd.DataFrame()
df['picture'] = picture_name
df['similarity'] = sim_group
df['distance'] = dist_group




#img1 = cv2.imread('evan1.jpg')
#img2 = cv2.imread('bambang2.jpg')
#model, detector = model(model_name='resnet100.onnx')

#dist, sim = getSimDist(img1, img2, model, detector)
