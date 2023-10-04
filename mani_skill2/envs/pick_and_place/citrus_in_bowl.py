from collections import OrderedDict
from copy import copy
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import sapien.core as sapien

from math import pi
from transforms3d.euler import euler2quat, quat2euler

from mani_skill2.envs.sapien_env import Action, ActionType

from mani_skill2 import ASSET_DIR, format_path
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import check_actor_static, vectorize_pose

from .base_env import StationaryManipulationEnv
from .pick_single import PickSingleYCBEnv
from .stack_cube import UniformSampler

@register_env("AToB-v0", max_episode_steps=1000)
class AToBEnv(StationaryManipulationEnv, PickSingleYCBEnv):
    OBJ_A_ID = '014_lemon'
    OBJ_B_ID = '024_bowl'

    def __init__(
        self,
        asset_root: str = None,
        model_json: str = None,
        obj_init_rot_z=True,
        obj_init_rot=0,
        goal_thresh=0.025,
        **kwargs,
    ):
        if asset_root is None:
            asset_root = self.DEFAULT_ASSET_ROOT
        self.asset_root = Path(format_path(asset_root))

        # TODO: model_json stores bounding box info about the objects
        # need to create myself or can reuse? What's the difference between
        # raw and pick? Scaling? Rot?
        if model_json is None:
            model_json = self.DEFAULT_MODEL_JSON
        model_json = self.asset_root / format_path(model_json)

        if not model_json.exists():
            raise FileNotFoundError(
                f"{model_json} is not found."
                "Please download the corresponding assets:"
                "`python -m mani_skill2.utils.download_asset ${ENV_ID}`."
            )

        self.model_scale = None
        self.model_bbox_size = None

        self.obj_init_rot_z = obj_init_rot_z
        self.obj_init_rot = obj_init_rot
        self.goal_thresh = goal_thresh

        self._check_assets()
        super().__init__(**kwargs)

    def _get_default_scene_config(self):
        scene_config = super()._get_default_scene_config()
        scene_config.enable_pcm = True
        return scene_config

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)

        self._load_objects()
        self.obj_a.set_damping(0.1, 0.1)
        self.obj_b.set_damping(0.1, 0.1)

    def _load_objects(self):
        # TODO: implement. Cannot reuse _load_model from PickSingleYCBEnv, as
        # we need to load two objects and thus can't store them a self.obj
        raise NotImplementedError

    def _initialize_actors(self):
        raise NotImplementedError
        xy = self._episode_rng.uniform(-0.1, 0.1, [2])
        region = [[-0.1, -0.2], [0.1, 0.2]]
        sampler = UniformSampler(region, self._episode_rng)
        radius = np.linalg.norm(self.box_half_size[:2]) + 0.001
        cubeA_xy = xy + sampler.sample(radius, 100)
        cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)

        cubeA_quat = euler2quat(0, 0, self._episode_rng.uniform(0, 2 * np.pi))
        cubeB_quat = euler2quat(0, 0, self._episode_rng.uniform(0, 2 * np.pi))
        z = self.box_half_size[2]
        cubeA_pose = sapien.Pose([cubeA_xy[0], cubeA_xy[1], z], cubeA_quat)
        cubeB_pose = sapien.Pose([cubeB_xy[0], cubeB_xy[1], z], cubeB_quat)

        self.cubeA.set_pose(cubeA_pose)
        self.cubeB.set_pose(cubeB_pose)

    def _get_obs_extra(self):
       raise NotImplementedError
       obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
        )
        if self._obs_mode in ["state", "state_dict", "state_dict+image"]:
            obs.update(
                cubeA_pose=vectorize_pose(self.cubeA.pose),
                cubeB_pose=vectorize_pose(self.cubeB.pose),
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.tcp.pose.p,
                tcp_to_cubeB_pos=self.cubeB.pose.p - self.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
            )
        return obs

    def _get_solution_sequence(self):
        raise NotImplementedError
        goal_a2w = copy(self.cubeA.pose)
        goal_b2w = copy(self.cubeB.pose)

        # Translate move_goal from world frame to robot root link frame, see
        # https://github.com/haosulab/SAPIEN/blob/ab1d9a9fa1428484a918e61185ae9df2beb7cb30/docs/source/tutorial/motion_planning/plan_a_path.rst#L37
        root2w = self.agent.robot.get_root_pose()
        w2root = root2w.inv()

        root2move_goal_a = w2root.transform(goal_a2w)
        root2move_goal_b = w2root.transform(goal_b2w)

        # get z rotation of cubes to match gripper rotation
        a_quat = root2move_goal_a.q
        a_euler = quat2euler(a_quat)
        # NOTE: get the closest rotation to pick the cube. With the + pi/4, we
        # get the closest rotation to the gripper [-45, 45) degrees. Without it
        # the angle would be [0, 90) degrees and unimodal.
        a_angle_z = (a_euler[2] + pi/4) % (pi/2) - (pi/4)  # mod by 90 degrees
        a_euler = (-pi, 0, a_angle_z)  # rotate 180 degrees around x axis
        a_rot = euler2quat(*a_euler)

        b_quat = root2move_goal_b.q
        b_euler = quat2euler(b_quat)
        b_angle_z = (b_euler[2] + pi/4) % (pi/2) - (pi/4)  # mod by 90 degrees
        b_euler = (-pi, 0, b_angle_z)  # rotate 180 degrees around x axis
        b_rot = euler2quat(*b_euler)

        z_offset = np.array([0, 0, self.box_half_size[2]])

        # Transform to np.ndarray
        move_goal_above_a = np.concatenate(
            [root2move_goal_a.p + z_offset * 2, a_rot])
        move_goal_a = np.concatenate([root2move_goal_a.p, a_rot])
        move_goal_above_b = np.concatenate(
            [root2move_goal_b.p + z_offset * 4, b_rot])
        move_goal_on_b = np.concatenate(
            [root2move_goal_b.p + 2 * z_offset, b_rot])

        seq = [
            Action(ActionType.MOVE_TO, goal=move_goal_above_a),
            Action(ActionType.MOVE_TO, goal=move_goal_a),
            Action(ActionType.CLOSE_GRIPPER),
            Action(ActionType.MOVE_TO, goal=move_goal_above_a),
            Action(ActionType.MOVE_TO, goal=move_goal_above_b),
            Action(ActionType.MOVE_TO, goal=move_goal_on_b),
            Action(ActionType.OPEN_GRIPPER),
        ]

        return seq

    def _check_cubeA_on_cubeB(self):
        raise NotImplementedError
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        offset = pos_A - pos_B
        xy_flag = (
            np.linalg.norm(offset[:2]) <= np.linalg.norm(self.box_half_size[:2]) + 0.005
        )
        z_flag = np.abs(offset[2] - self.box_half_size[2] * 2) <= 0.005
        return bool(xy_flag and z_flag)

    def evaluate(self, **kwargs):
        raise NotImplementedError
        is_cubeA_on_cubeB = self._check_cubeA_on_cubeB()
        is_cubeA_static = check_actor_static(self.cubeA)
        is_cubaA_grasped = self.agent.check_grasp(self.cubeA)
        success = is_cubeA_on_cubeB and is_cubeA_static and (not is_cubaA_grasped)

        return {
            "is_cubaA_grasped": is_cubaA_grasped,
            "is_cubeA_on_cubeB": is_cubeA_on_cubeB,
            "is_cubeA_static": is_cubeA_static,
            # "cubeA_vel": np.linalg.norm(self.cubeA.velocity),
            # "cubeA_ang_vel": np.linalg.norm(self.cubeA.angular_velocity),
            "success": success,
        }

    def compute_dense_reward(self, info, **kwargs):
       raise NotImplementedError
