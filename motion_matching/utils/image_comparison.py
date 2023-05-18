import argparse
import cv2
import mediapipe as mp
import numpy as np
from openvino.runtime import Core

import torch
import torchvision
import itertools

from motion_matching.utils.pose import Pose


def human_mediapipe_detection(image_path, pose):
    # print('human: ', image_path)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    a = Pose()

    image_height, image_width, _ = image.shape
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        landmark = results.pose_landmarks.landmark
        left_eye = landmark[2]
        right_eye = landmark[5]
        head = [(left_eye.x + right_eye.x) / 2 * image_width,
                (left_eye.y + right_eye.y) / 2 * image_height]

        right_hand = [landmark[16].x * image_width,
                      landmark[16].y * image_height]
        left_hand = [landmark[15].x * image_width,
                     landmark[15].y * image_height]
        right_foot = [landmark[28].x * image_width,
                      landmark[28].y * image_height]
        left_foot = [landmark[27].x * image_width,
                     landmark[27].y * image_height]

        y_trunk_lmid = (landmark[11].y + landmark[23].y) / 2
        y_trunk_llower = (y_trunk_lmid + landmark[23].y) / 2
        y_trunk_rmid = (landmark[12].y + landmark[24].y) / 2
        y_trunk_rlower = (y_trunk_rmid + landmark[24].y) / 2

        y_trunk = (y_trunk_llower + y_trunk_rlower) / 2 * image_height
        # y_trunk = (y_trunk_rmid + y_trunk_lmid) / 2 * image_height
        trunk = [(landmark[23].x + landmark[24].x) / 2 * image_width, y_trunk]

        keypoints = [head, trunk, right_hand, left_hand, right_foot, left_foot]
        visualize(image, keypoints=np.array([keypoints]))

        name = image_path.split('/')[-1].split('.')[0]
        new_name = image_path.replace(name, 'result/' + name + '_result')
        cv2.imwrite(new_name, image)

        merged = list(itertools.chain.from_iterable(keypoints))

        # reinitialize image_points
        image_points = []
        # Normalization of the points
        input_new_coords = np.asarray(a.roi(merged)).reshape(6, 2)
        image_points.append(input_new_coords)
        return image_points


def load_robot_rcnn_model(path):
    # Load the network in OpenVINO Runtime.
    ie = Core()
    model_ir = ie.read_model(model='weight/keypoint_rcnn.xml')
    compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU")

    return compiled_model_ir


def robot_rcnn_detection(image_path, model):
    device = torch.device('cpu')
    a = Pose()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    orig_frame = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float)
    image = image.unsqueeze(0).to(device)

    res_ir = model([image])

    boxes_index = model.output(0)
    labels_index = model.output(1)
    scores_index = model.output(2)
    keypoints_index = model.output(3)
    keypoints_scores_index = model.output(4)

    scores = res_ir[scores_index]
    # Indexes of boxes with scores > 0.7
    high_scores_idxs = np.where(scores > 0.55)[0].tolist()

    boxes_1 = torch.from_numpy(res_ir[boxes_index])
    scores_1 = torch.from_numpy(res_ir[scores_index])
    keypoints_1 = torch.from_numpy(res_ir[keypoints_index])

    post_nms_idxs = torchvision.ops.nms(boxes_1[high_scores_idxs], scores_1[high_scores_idxs], 0.3).cpu(
    ).numpy()  # Indexes of boxes left after applying NMS (iou_threshold=0.3)

    keypoints = []
    for kps in keypoints_1[high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        keypoints.append([list(map(int, kp[:2])) for kp in kps])
    # print(keypoints)
    bboxes = []
    for bbox in boxes_1[high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        bboxes.append(list(map(int, bbox.tolist())))

    visualize(orig_frame, bboxes, keypoints)
    name = image_path.split('/')[-1].split('.')[0]
    new_name = image_path.replace(name, 'result/' + name + '_result')
    cv2.imwrite(new_name, orig_frame)

    merged = list(itertools.chain.from_iterable(keypoints[0]))

    # reinitialize image_points
    image_points = []
    # Normalization of the points
    input_new_coords = np.asarray(a.roi(merged)).reshape(6, 2)
    image_points.append(input_new_coords)
    return image_points


def visualize(image, bboxes=None, keypoints=None):
    # keypoints_classes_ids2names = {0: 'Head', 1: 'Trunk', 2: 'RH', 3: 'LH', 4: 'RF', 5: 'LF'}
    # keypoints_classes_ids2names = {0: 'NOSE', 1: 'LEFT_EYE', 2: 'RIGHT_EYE', 3: 'LEFT_EAR', 4: 'RIGHT_EAR', 5: 'LEFT_SHOULDER', 6:'RIGHT_SHOULDER', 7:'LEFT_ELBOW', 8:'RIGHT_ELBOW', 9:'LEFT_WRIST', 10:'RIGHT_WRIST', 11:'LEFT_HIP', 12:'RIGHT_HIP' ,13:'LEFT_KNEE', 14:'RIGHT_KNEE',15:'LEFT_ANKLE', 16:'RIGHT_ANKLE'}

    if bboxes is not None:
        for bbox in bboxes:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image = cv2.rectangle(image, start_point,
                                  end_point, (0, 255, 0), 2)

    if keypoints is not None:
        for kps in keypoints:
            for idx, kp in enumerate(kps):
                kp = tuple(int(i) for i in kp)

                # cv2.circle(image, center_coordinates, radius, color, thickness)
                image = cv2.circle(image, kp, 4, (255, 0, 0), -1)
                # cv2.putText(image, text, org, font, fontScale, color, thickness, lineType, bottomLeftOrigin)
                # image = cv2.putText(image, " " + keypoints_classes_ids2names[idx], kp, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
