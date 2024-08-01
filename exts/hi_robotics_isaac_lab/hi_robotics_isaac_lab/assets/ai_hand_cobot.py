import os

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg

AIHAND_COBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{os.path.dirname(__file__)}/cobot_aihand/cobot_aihand.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "W1": 0.0,
            "FR_1": 0.0,
            "FR_2": 0.0,
            "FL_1": 0.0,
            "FL_2": 0.0,
        }
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            velocity_limit={
                "joint[1-3]": 3.1416,
                "joint[4-6]": 6.2832,
            },
            effort_limit={
                "joint[1-2]": 194.0,
                "joint[3]": 163.0,
                "joint[4-6]": 50.0,
            },
            stiffness={
                "joint[1-3]": 100.0,
                "joint[4-6]": 50.0,
            },
            damping={
                "joint[1-3]": 1.0,
                "joint[4-6]": 0.5,
            },
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["W1"],
            effort_limit=50.0,
            velocity_limit=6.2832,
            stiffness=100.0,
            damping=1.0,
        ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["FR_[1-2]", "FL_[1-2]"],
            effort_limit=30.0,
            velocity_limit=50.0,
            stiffness=50.0,
            damping=1.0,
        ),
    },
)

AIHAND_COBOT_HIGH_PD_CFG = AIHAND_COBOT_CFG.copy()
AIHAND_COBOT_HIGH_PD_CFG.actuators["arm"].stiffness = {
    "joint[1-3]": 400.0,
    "joint[4-6]": 200.0,
}

AIHAND_COBOT_HIGH_PD_CFG.actuators["arm"].damping = {
    "joint[1-3]": 80.0,
    "joint[4-6]": 80.0,
}

AIHAND_COBOT_HIGH_PD_CFG.actuators["wrist"].stiffness = 400.0
AIHAND_COBOT_HIGH_PD_CFG.actuators["wrist"].damping = 80.0

AIHAND_COBOT_HIGH_PD_CFG.actuators["fingers"].stiffness = 200.0
AIHAND_COBOT_HIGH_PD_CFG.actuators["fingers"].damping = 40.0
