import tensorflow as tf
import os
import pathlib
import numpy as np
import zipfile
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from plot_object_detection_saved_model import IMAGE_PATHS

# 경로세팅
PATH_TO_LABELS = 'C:\\Users\\5-18\\Documents\\TensorFlow\\models-master\\research\\object_detection\\data\\mscoco_label_map.pbtxt'
CATEGORY_INDEX = label_map_util.create_categories_from_labelmap(PATH_TO_LABELS)
PATH_TO_MODEL_DIR = download_model(MODEL_NAME, MODEL_DATE)

def load_model(model_dir) :
    model_full_dir = model_dir + "/saved_model"  
    # Load saved model and build the detection function
    detection_model = tf.saved_model.load(model_full_dir)
    return detection_model

detection_model = load_model(PATH_TO_MODEL_DIR)

print(detection_model.signatures['serving_default'].inputs)
print(detection_model.signatures['serving_default'].output_dtypes)
print(detection_model.signatures['serving_default'].output_shapes)

# 우리가 가지고 있는 이미지 경로에서 이미지를 가져오는 코드
PATH_TO_IMAGE_DIR = pathlib.Path('data')
IMAGE_PATHS = list(PATH_TO_IMAGE_DIR.glob('*.jpg'))

print(IMAGE_PATHS)