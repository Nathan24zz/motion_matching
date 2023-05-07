from enum import Enum


class JointId(Enum):
    # head motors
    NECK_YAW = 19
    NECK_PITCH = 20

    # left arm motors
    LEFT_SHOULDER_PITCH = 2
    LEFT_SHOULDER_ROLL = 4
    LEFT_ELBOW = 6

    # right arm motors
    RIGHT_SHOULDER_PITCH = 1
    RIGHT_SHOULDER_ROLL = 3
    RIGHT_ELBOW = 5

    # left leg motors
    LEFT_HIP_YAW = 8
    LEFT_HIP_ROLL = 10
    LEFT_HIP_PITCH = 12
    LEFT_KNEE = 14
    LEFT_ANKLE_PITCH = 16
    LEFT_ANKLE_ROLL = 18

    # right leg motors
    RIGHT_HIP_YAW = 7
    RIGHT_HIP_ROLL = 9
    RIGHT_HIP_PITCH = 11
    RIGHT_KNEE = 13
    RIGHT_ANKLE_PITCH = 15
    RIGHT_ANKLE_ROLL = 17


joint_id_by_name = {
    JointId.NECK_YAW.value: "neck_yaw",
    JointId.NECK_PITCH.value: "neck_pitch",
    JointId.LEFT_SHOULDER_PITCH.value: "left_shoulder_pitch",
    JointId.LEFT_SHOULDER_ROLL.value: "left_shoulder_roll",
    JointId.LEFT_ELBOW.value: "left_elbow",
    JointId.RIGHT_SHOULDER_PITCH.value: "right_shoulder_pitch",
    JointId.RIGHT_SHOULDER_ROLL.value: "right_shoulder_roll",
    JointId.RIGHT_ELBOW.value: "right_elbow",
    JointId.LEFT_HIP_YAW.value: "left_hip_yaw",
    JointId.LEFT_HIP_ROLL.value: "left_hip_roll",
    JointId.LEFT_HIP_PITCH.value: "left_hip_pitch",
    JointId.LEFT_KNEE.value: "left_knee",
    JointId.LEFT_ANKLE_ROLL.value: "left_ankle_roll",
    JointId.LEFT_ANKLE_PITCH.value: "left_ankle_pitch",
    JointId.RIGHT_HIP_YAW.value: "right_hip_yaw",
    JointId.RIGHT_HIP_ROLL.value: "right_hip_roll",
    JointId.RIGHT_HIP_PITCH.value: "right_hip_pitch",
    JointId.RIGHT_KNEE.value: "right_knee",
    JointId.RIGHT_ANKLE_ROLL.value: "right_ankle_roll",
    JointId.RIGHT_ANKLE_PITCH.value: "right_ankle_pitch"
}
