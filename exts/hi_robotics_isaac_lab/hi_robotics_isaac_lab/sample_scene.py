"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="This script creates a sample scene for the AIHand plus DOOSAN Cobot.")

parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create")
parser.add_argument("--env_spacing", type=float, default=5.0, help="Distance each environment should be spaced apart")

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import numpy as np
import time
import torch

import requests
from sympy import euler

from hi_robotics_isaac_lab.assets.ai_hand_cobot import AIHAND_COBOT_CFG, AIHAND_COBOT_HIGH_PD_CFG
from hi_robotics_isaac_lab.assets.ai_hand_ikin import ai_hand_ikin

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.controllers.joint_impedance import JointImpedanceController, JointImpedanceControllerCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR


@configclass
class AIHandSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(rot=[0.70711, 0, 0, 0.70711]),
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    stand = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Stand",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.25, 0, 0.25]),
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand.usd"),
    )

    soup_can = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SoupCan",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.25, 0, 0.5], rot=[0.5, -0.5, -0.5, 0.5]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
        ),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # AIHand + Cobot
    aihand_cobot: ArticulationCfg = AIHAND_COBOT_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Cobot_AIHand")
    aihand_cobot.init_state.pos = [-0.6, 0.0, 0.0]


def degree_to_radian(degree_vector: torch.Tensor) -> torch.Tensor:
    """
    Converts a degree vector to a radian vector.

    Args:
        degree_vector (torch.Tensor): Vector of angles in degrees

    Returns:
        torch.Tensor: Vector of angles in radians
    """
    return degree_vector * np.pi / 180.0


def cobot_and_ai_hand_joint_angles(joint_angles: list[float], device: str) -> torch.Tensor:
    """
    Converts a list of joint angles to a tensor of joint angles for the AIHand + Cobot.

    Format: [joint1, joint2, joint3, joint4, joint5, joint6, W1, FR_1, FL_1, FR_2, FL_2]

    Args:
        joint_angles (list[float]): List of joint angles for the AIHand + Cobot

    Returns:
        torch.Tensor: Tensor of joint angles for the AIHand + Cobot
    """
    return degree_to_radian(torch.tensor(joint_angles)).to(device)


def compute_move_close_to_object(pose: list[float], cobot_api_link: str, endpoint: str, device: str) -> torch.Tensor:
    """
    Compute the joint angles given the desired 6d pose of the Cobot

    Args:
        pose (list[float]): Desired 6d pose of the Cobot [x, y, z, a, b, c]

    Returns:
        torch.Tensor: Joint angles for the Cobot
    """

    payload = {
        "X": pose[0],
        "Y": pose[1],
        "Z": pose[2],
        "A": pose[3],
        "B": pose[4],
        "C": pose[5],
    }

    response = requests.post("https://" + cobot_api_link + endpoint, json=payload)

    return torch.tensor(response.json()["joint_angles"]).to(
        device
    )  # Not sure what the receiving format is so edit this @Saketh


def get_target_position() -> torch.Tensor:
    """
    Get the target position for the AIHand + Cobot.

    Returns:
        torch.Tensor: Target position for the AIHand + Cobot
    """
    ...


def quaternion_to_euler(w, x, y, z):
    """Convert quaternion to Euler angles (roll, pitch, yaw)."""
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pi_over_2 = torch.tensor(math.pi / 2, device=sinp.device)
    pitch = torch.where(torch.abs(sinp) >= 1, torch.copysign(pi_over_2, sinp), torch.asin(sinp))

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def get_positions_orientations(object) -> torch.Tensor:
    """Return the positions and orientations (in Euler angles) of the objects."""
    positions = object.data.body_pos_w
    orientations = object.data.body_quat_w

    # Convert quaternions to Euler angles for all instances and bodies
    euler_orientations = torch.zeros_like(orientations[..., :3])
    for i in range(orientations.shape[0]):
        for j in range(orientations.shape[1]):
            w, x, y, z = orientations[i, j]
            roll, pitch, yaw = quaternion_to_euler(w, x, y, z)
            euler_orientations[i, j] = torch.tensor([roll, pitch, yaw])

    pose = torch.cat((positions, euler_orientations), dim=-1)

    return pose


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    velocity: float = 0.5,
    seconds_to_reset: float | None = None,
):
    """Runs the simulation loop."""

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    robot = scene["aihand_cobot"]
    soup_can: RigidObject = scene["soup_can"]

    # Buffers
    current_joint_pos = torch.zeros_like(robot.data.default_joint_pos).to(scene.device)
    new_joint_pos = torch.zeros_like(robot.data.default_joint_pos).to(scene.device)
    target_joint_pos = torch.zeros_like(robot.data.default_joint_pos).to(scene.device)
    step_pos = torch.zeros_like(robot.data.default_joint_pos).to(scene.device)
    delta_pos = torch.zeros_like(robot.data.default_joint_pos).to(scene.device)

    # Define the new home position joint angles
    target_joint_pos[:] = cobot_and_ai_hand_joint_angles(
        [0.0, 0.0, 90.0, 0.0, 90.0, 0.0, 0.0, 90.0, -90.0, 90.0, -90.0], device=scene.device
    )

    reset_time = time.time() + seconds_to_reset if seconds_to_reset is not None else None

    # Simulate physics
    while simulation_app.is_running():

        if reset_time is not None and time.time() > reset_time:
            count = 0

            soup_state = soup_can.data.default_root_state.clone()
            soup_can.write_root_state_to_sim(soup_state)

            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_state_to_sim(root_state)
            # set joint positions
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()

            reset_time = time.time() + seconds_to_reset if seconds_to_reset is not None else None

        # Get the current joint positions
        current_joint_pos[:] = robot.data.joint_pos

        soup_can_6d_pose = get_positions_orientations(soup_can)

        device = current_joint_pos.device

        delta_pos[:] = target_joint_pos - current_joint_pos
        step_pos[:] = delta_pos * velocity
        new_joint_pos[:] = current_joint_pos + step_pos

        # Create new joint positions if the robot has reached the target
        if torch.allclose(new_joint_pos, target_joint_pos, atol=0.01):
            print("[INFO]: Target joint positions reached. Creating new target joint positions.")
            # target_joint_pos[:] = get_target_position()

        # Apply the interpolated joint positions to the robot
        robot.set_joint_position_target(new_joint_pos)
        # Write data to sim
        scene.write_data_to_sim()

        # Perform step
        sim.step()

        count += 1

        # Update buffers
        scene.update(sim_dt)


def main():
    num_of_envs = args_cli.num_envs
    env_spacing = args_cli.env_spacing
    print("====================================================")
    print(f"[INFO]: Creating {num_of_envs} environments with spacing of {env_spacing} units.")
    print("====================================================")

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set main camera
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    scene_cfg = AIHandSceneCfg(num_envs=num_of_envs, env_spacing=env_spacing)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()

    simulation_app.close()
