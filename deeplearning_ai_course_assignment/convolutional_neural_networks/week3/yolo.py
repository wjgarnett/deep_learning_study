# coding: utf-8
import os
import numpy as np
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
import tensorflow as tf
from yolo_utils import *
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
import scipy.misc
from matplotlib.pyplot import imshow


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.6):
    """对yolo网络检测结果过滤

    Args:
        box_confidence: 有目标的概率，shape(19, 19, 5, 1)
        boxes: 目标位置信息，(19, 19, 5, 4)
        box_class_probs: 目标类别信息，共80类，(19, 19, 5, 80)
        threshold:

    Returns:
        score: 目标属于该类别概率
        boxes: 目标位置信息
        classes: 目标类别信息
    """

    box_scores = box_confidence * box_class_probs

    box_class = K.argmax(box_scores, axis=-1)
    box_class_score = K.max(box_scores, axis=-1)

    mask = box_class_score > threshold
    scores = tf.boolean_mask(box_class_score, mask)
    boxes = tf.boolean_mask(boxes, mask)
    classes = tf.boolean_mask(box_class, mask)

    return scores, boxes, classes


def iou(box1, box2):
    """计算IOU

    Args:
        box1: (x1, y1, x2, y2)
        box2:

    Returns:
        iou: IOU结果
    """

    x1_max = max(box1[0], box2[0])
    x2_min = min(box1[2], box2[2])
    y1_max = max(box1[1], box2[1])
    y2_min = min(box1[3], box2[3])
    inter_area = (x2_min - x1_max) * (y2_min - y1_max)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - 2*inter_area

    iou = inter_area / (1.0 * union_area)

    return iou


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):

    max_boxes_tensor = K.variable(max_boxes, dtype='int32')  # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))  # initialize variable max_boxes_tensor

    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    ### START CODE HERE ### (≈ 1 line)
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)
    ### END CODE HERE ###

    # Use K.gather() to select only nms_indices from scores, boxes and classes
    ### START CODE HERE ### (≈ 3 lines)
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)
    ### END CODE HERE ###

    return scores, boxes, classes


def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=0.6, iou_threshold=0.5):

    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=score_threshold)

    boxes = scale_boxes(boxes, image_shape)

    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes


def predict(sess, image_file):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.

    Arguments:
    sess -- your tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image stored in the "images" folder.

    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes

    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes.
    """

    # Preprocess your image
    image, image_data = preprocess_image("images/" + image_file, model_image_size=(608, 608))

    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    # You'll need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
    ### START CODE HERE ### (≈ 1 line)
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],
                                                  feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})
    ### END CODE HERE ###

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=90)
    # Display the results in the notebook
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(output_image)

    return out_scores, out_boxes, out_classes


if __name__ == '__main__':
    sess = K.get_session()
    class_names = read_classes("model_data/coco_classes.txt")
    anchors = read_anchors("model_data/yolo_anchors.txt")
    image_shape = (720., 1280.)
    yolo_model = load_model("model_data/yolo.h5")

    yolo_model.summary()
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

    out_scores, out_boxes, out_classes = predict(sess, "test.jpg")
