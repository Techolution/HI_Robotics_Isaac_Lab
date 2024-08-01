import math
import numpy as np

# Constants
WRIST_LENGTH = 83
PALM_LENGTH = 178
FINGER_SEGMENT_1_LENGTH = 118.5
FINGER_SEGMENT_2_LENGTH = 103
MIN_FINGER_DIFF = 0
MAX_FINGER_DIFF = 120


def calculate_width(left_point, right_point):
    x1, y1, z1 = left_point
    x2, y2, z2 = right_point
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def calculate_palm_tcp(wrist_tcp, pan_angle):
    return (
        wrist_tcp[0] + PALM_LENGTH * np.sin(pan_angle),
        wrist_tcp[1],
        wrist_tcp[2] + PALM_LENGTH * np.cos(pan_angle),
    )


def calculate_finger_tcp(palm_tcp, pan_angle, temp_angle):
    return (
        palm_tcp[0] + np.sin(pan_angle) * (FINGER_SEGMENT_1_LENGTH * np.sin(temp_angle) + FINGER_SEGMENT_2_LENGTH),
        palm_tcp[1] + FINGER_SEGMENT_1_LENGTH * np.cos(temp_angle),
        palm_tcp[2] + np.cos(pan_angle) * (FINGER_SEGMENT_1_LENGTH * np.sin(temp_angle) + FINGER_SEGMENT_2_LENGTH),
    )


def ai_hand_ikin(target_pose, object_rotation, left_point, right_point, pick_method="vertical_pick"):
    """
    Calculate the inverse kinematics for a robotic hand to pick up an object.

    Parameters:
    target_pose (tuple): Target position (x, y, z) and rotation angles (A, B, C).
    object_rotation (tuple): Rotation angles (rx, ry, rz) of the object.
    left_point (tuple): Coordinates (x, y, z) of the left point of the object.
    right_point (tuple): Coordinates (x, y, z) of the right point of the object.
    pick_method (str): Method of picking the object, either "vertical_pick" or "horizontal_pick".

    Returns:
    dict: Dictionary containing the calculated angles and positions.
    """
    x_target, y_target, z_target, A_target, B_target, C_target = target_pose
    rx_object, ry_object, rz_object = object_rotation

    width = calculate_width(left_point, right_point)
    wrist_tcp = (0, 0, WRIST_LENGTH)

    if pick_method == "vertical_pick":
        pan_angle = np.deg2rad(45)
    elif pick_method == "horizontal_pick":
        pan_angle = np.deg2rad(-45)
    else:
        raise ValueError("Invalid pick method. Choose either 'vertical_pick' or 'horizontal_pick'.")

    palm_tcp = calculate_palm_tcp(wrist_tcp, pan_angle)
    temp_angle = np.arccos(((width - MIN_FINGER_DIFF) / 2) / FINGER_SEGMENT_1_LENGTH)

    finger_left_1_angle = -(90 - np.rad2deg(temp_angle))
    finger_left_2_angle = finger_left_1_angle
    finger_right_1_angle = -finger_left_2_angle
    finger_right_2_angle = finger_right_1_angle

    finger_tcp = calculate_finger_tcp(palm_tcp, pan_angle, temp_angle)

    return {
        "finger_left_1_angle": finger_left_1_angle,
        "finger_left_2_angle": finger_left_2_angle,
        "finger_right_1_angle": finger_right_1_angle,
        "finger_right_2_angle": finger_right_2_angle,
        "finger_tcp": finger_tcp,
    }
