# custom_go1_robot.py
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass


@configclass
class CustomGo1RobotCfg(ArticulationCfg):
    """Custom configuration for Unitree Go1 robot using your USD file."""

    # USD file path
    prim_path = "/World/envs/env_.*/Robot"
    usd_path = "/home/sripu715/Downloads/go1_sensor.usd"

    # Spawn configuration
    spawn = sim_utils.UsdFileCfg(
        usd_path=usd_path,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.0,
            stabilization_threshold=0.0,
        ),
    )

    # Initial state
    init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": 0.67,
            ".*_calf_joint": -1.3,
        },
        joint_vel={".*": 0.0},
    )

    # For Isaac Lab, use the controller configuration if available
    # Remove the soft_joint_pos_ctrl line if it's not supported

    # # Soft joint position control (PD controller)
    # soft_joint_pos_ctrl = ArticulationCfg.SoftJointPositionControllerCfg(
    #     stiffness=30.0,
    #     damping=1.0,
    #     usd_physics=True,
    # )