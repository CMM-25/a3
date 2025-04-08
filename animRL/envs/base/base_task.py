from isaacgym.torch_utils import to_torch, torch_rand_float, get_axis_params, quat_rotate_inverse
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import sys
import torch
import os

import matplotlib.pyplot as plt
from collections import deque

from animRL import ROOT_DIR
from animRL.utils.helpers import class_to_dict
from animRL.utils.math import quat_apply_yaw
from animRL.rewards.rewards import REWARDS


# Base class for RL tasks
class BaseTask:

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        self.headless = headless
        self.is_playing = False
        self.init_done = False

        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        if sim_device_type == 'cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'
        self.graphics_device_id = self.sim_device_id
        if self.headless:
            self.graphics_device_id = -1

        # parse env config
        self._parse_cfg()

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id,
                                       self.graphics_device_id,
                                       self.physics_engine,
                                       self.sim_params)
        if self.cfg.terrain.mesh_type == 'plane':
            self._create_ground_plane()
        else:
            raise Exception

        self._create_envs()
        self.gym.prepare_sim(self.sim)

        self.enable_viewer_sync = True
        self.overview = self.cfg.viewer.overview
        self.focus_env = self.cfg.viewer.ref_env
        self.viewer = None
        self.debug_viz = False

        self._init_buffers()
        self._set_camera_recording()

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.cfg.viewer.enable_viewer:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.overview:
                self._set_camera(self.cfg.viewer.overview_pos, self.cfg.viewer.overview_lookat)
            else:
                ref_pos = [self.root_states[self.focus_env, 0].item() + self.cfg.viewer.ref_pos_b[0],
                           self.root_states[self.focus_env, 1].item() + self.cfg.viewer.ref_pos_b[1],
                           self.cfg.viewer.ref_pos_b[2]]
                ref_lookat = [self.root_states[self.focus_env, 0].item(),
                              self.root_states[self.focus_env, 1].item(),
                              0.2]
                self._set_camera(ref_pos, ref_lookat)
            self.viewer_set = True

            # keyboard events
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_O, "toggle_overview")

            # if running with a viewer, and we are running play function
            if not self.headless and self.cfg.env.play:
                self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "key_space")
                self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "key_r")
                self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_NUMPAD_ADD, "key_plus")
                self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_NUMPAD_SUBTRACT, "key_minus")

    def _set_camera_recording(self):
        camera_props = gymapi.CameraProperties()
        camera_props.horizontal_fov = self.cfg.viewer.camera_horizontal_fov
        camera_props.width = self.cfg.viewer.camera_width
        camera_props.height = self.cfg.viewer.camera_height
        self.camera_props = camera_props
        self.image_env = self.cfg.viewer.camera_env
        self.camera_sensors = self.gym.create_camera_sensor(self.envs[self.image_env], self.camera_props)
        self.camera_image = np.zeros((self.camera_props.height, self.camera_props.width, 4), dtype=np.uint8)

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def compute_observations(self):
        raise NotImplementedError

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.critic_obs_buf

    def render(self, sync_frame_time=True):
        # fetch results
        if self.device != 'cpu':
            self.gym.fetch_results(self.sim, True)

        if self.cfg.viewer.record_camera_imgs:
            ref_pos = [self.root_states[self.image_env, 0].item() + self.cfg.viewer.camera_pos_b[0],
                       self.root_states[self.image_env, 1].item() + self.cfg.viewer.camera_pos_b[1],
                       self.env_origins[self.focus_env, 2] + self.cfg.viewer.camera_pos_b[2]]
            ref_lookat = [self.root_states[self.image_env, 0].item() + self.cfg.viewer.ref_lookat[0],
                          self.root_states[self.image_env, 1].item() + self.cfg.viewer.ref_lookat[1],
                          self.env_origins[self.focus_env, 2] + self.cfg.viewer.ref_lookat[2]]
            cam_pos = gymapi.Vec3(ref_pos[0], ref_pos[1], ref_pos[2])
            cam_target = gymapi.Vec3(ref_lookat[0], ref_lookat[1], ref_lookat[2])

            self.gym.set_camera_location(self.camera_sensors, self.envs[self.image_env], cam_pos, cam_target)

        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "toggle_overview" and evt.value > 0:
                    self.overview = not self.overview
                    self.viewer_set = False

            # step graphics
            if self.enable_viewer_sync:
                if self.cfg.env.play and not self.overview:
                    ref_pos = [self.root_states[self.focus_env, 0].item() + self.cfg.viewer.ref_pos_b[0],
                               self.root_states[self.focus_env, 1].item() + self.cfg.viewer.ref_pos_b[1],
                               self.cfg.viewer.ref_pos_b[2]]
                    ref_lookat = [self.root_states[self.focus_env, 0].item() + self.cfg.viewer.ref_lookat[0],
                                  self.root_states[self.focus_env, 1].item() + self.cfg.viewer.ref_lookat[1],
                                  self.cfg.viewer.ref_lookat[2]]
                    self._set_camera(ref_pos, ref_lookat)
                else:
                    if not self.viewer_set:
                        if self.overview:
                            self._set_camera(self.cfg.viewer.overview_pos, self.cfg.viewer.overview_lookat)
                        else:
                            ref_pos = [
                                self.root_states[self.focus_env, 0].item() + self.cfg.viewer.ref_pos_b[0],
                                self.root_states[self.focus_env, 1].item() + self.cfg.viewer.ref_pos_b[1],
                                self.cfg.viewer.ref_pos_b[2]]
                            ref_lookat = [self.root_states[self.focus_env, 0].item() + self.cfg.viewer.ref_lookat[0],
                                          self.root_states[self.focus_env, 1].item() + self.cfg.viewer.ref_lookat[1],
                                          self.cfg.viewer.ref_lookat[2]]
                            self._set_camera(ref_pos, ref_lookat)
                        self.viewer_set = True
            else:
                self.gym.poll_viewer_events(self.viewer)

        if self.cfg.viewer.record_camera_imgs or (self.viewer and self.enable_viewer_sync):
            self.gym.step_graphics(self.sim)

            if self.cfg.viewer.record_camera_imgs:
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)
                self.camera_image = self.gym.get_camera_image(self.sim, self.envs[self.image_env], self.camera_sensors,
                                                              gymapi.IMAGE_COLOR).reshape((self.camera_props.height,
                                                                                           self.camera_props.width, 4))
                self.gym.end_access_image_tensors(self.sim)

            if self.viewer and self.enable_viewer_sync:
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)

    # ------------------------------------------------------------------------------------------------------------------

    def _parse_cfg(self):
        self.num_envs = self.cfg.env.num_envs
        self.num_obs = self.cfg.env.num_observations
        self.num_privileged_obs = self.cfg.env.num_privileged_obs
        self.num_actions = self.cfg.env.num_actions
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.max_episode_length = self.cfg.env.episode_length

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        asset_path = self.cfg.asset.file.format(ROOT_DIR=ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self._set_default_dof_pos()
        ee_names = [s for s in body_names if any(key in s for key in self.cfg.asset.ee_offsets.keys())]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + \
                               self.cfg.init_state.rot + \
                               self.cfg.init_state.lin_vel + \
                               self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim,
                                             gymapi.Vec3(0., 0., 0.),
                                             gymapi.Vec3(0., 0., 0.),
                                             int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            # pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i,
                                                 self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.ee_indices = torch.zeros(len(ee_names), dtype=torch.long, device=self.device)
        self.ee_offsets = torch.zeros(len(ee_names), 3, dtype=torch.float, device=self.device)
        for i in range(len(ee_names)):
            self.ee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], ee_names[i])
            self.ee_offsets[i, :] = to_torch(self.cfg.asset.ee_offsets[ee_names[i]])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])

    def _get_env_origins(self):
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.  # flat terrain

    def _process_rigid_shape_props(self, props, env_id):
        # randomize friction
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1),
                                                    device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_rigid_body_props(self, props):
        # randomize mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props

    def _process_dof_props(self, props, env_id):
        if env_id == 0:
            self.dof_pos_soft_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device)
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_soft_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_soft_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                # Note: position mode needs stiffness and damping to be set,
                # in velocity mode stiffness should be zero,
                # in torque mode both should be zero
                if props["driveMode"][i] == 1:  # position mode
                    props["stiffness"][i] = self.p_gains[i]
                    props["damping"][i] = self.d_gains[i]
                elif props["driveMode"][i] == 2:  # velocity mode
                    props["damping"][i] = self.d_gains[i]
                elif props["driveMode"][i] == 3:  # torque mode
                    props["damping"][i] = 0.01  # slightly positive for better stability

        return props

    def _set_default_dof_pos(self):
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _init_buffers(self):
        # basic buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.critic_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device,
                                              dtype=torch.float)
        else:
            self.critic_obs_buf = None

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.base_quat = self.root_states[:, 3:7]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # contact_forces shape: num_envs, num_bodies, xyz axis
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)

        # constant vectors
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))

        # some buffers
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)

        # joint targets
        self.joint_targets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, )
        self.last_joint_targets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)
        self.joint_targets_rate = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.joint_targets_rate_scaler = np.sqrt(self.num_actions) * self.dt

        # torque
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)

        # end effector
        self.num_ee = len(self.cfg.asset.ee_offsets)

        self.extras = {}
        self.common_step_counter = 0

    def _compute_torques(self, joint_targets):
        # pd controller
        control_type = self.cfg.control.control_type

        p_gains = self.p_gains
        d_gains = self.d_gains

        torques = self.torque_limits
        if control_type == "P":
            if self.cfg.asset.default_dof_drive_mode == 3:
                torques = p_gains * (joint_targets + self.default_dof_pos - self.dof_pos) - d_gains * self.dof_vel
        elif control_type == "T":
            torques = joint_targets
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _prepare_reward_function(self):
        self.reward_terms = class_to_dict(self.cfg.rewards.terms)

        # buffers for sum of rewards
        self.episode_term_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for name in self.reward_terms.keys()}
        self.rew_terms_buf = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for name in self.reward_terms.keys()}
        self.episode_rew_sums = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def compute_reward(self):
        self.rew_buf[:] = 1.0
        for reward_name, reward_cfg in self.reward_terms.items():
            reward_function = getattr(REWARDS, 'reward_' + reward_name)
            reward_sigma = reward_cfg[0]
            reward_tolerance = reward_cfg[1]
            term_reward = reward_function(self.data, reward_sigma, reward_tolerance)
            self.rew_terms_buf[reward_name] = term_reward
            self.episode_term_sums[reward_name] += term_reward
            self.rew_buf *= term_reward
        self.episode_rew_sums += self.rew_buf

    def process_keystroke(self):
        # check for keyboard events
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.value > 0:
                if evt.action == "key_space":
                    self.is_playing = not self.is_playing
                if evt.action == "key_r":
                    self.reset_triggered = True
                if evt.action == "toggle_overview":
                    self.overview = not self.overview

    def draw_ghost(self, env_id, root_state, dof_pos):
        self.dof_pos[env_id] = dof_pos
        self.dof_vel[env_id] = 0.0
        env_ids_int32 = torch.asarray([env_id], dtype=torch.int32, device=self.device)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.root_states[env_id, :7] = root_state[:7]
        self.root_states[env_id, 7:13] = 0.0
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _set_camera(self, position, lookat):
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, self.envs[self.focus_env], cam_pos, cam_target)

    def plotter_init(self, y_getters):
        num_sub_plots = len(y_getters)
        max_data_points = 200
        self.plotter_fig, self.plotter_axs = plt.subplots(num_sub_plots)
        self.plotter_x_buffer = deque(maxlen=max_data_points)
        self.plotter_y_buffer = []
        self.plotter_lines = []
        for subplot_id in range(num_sub_plots):
            curr_plotter_ax = self.plotter_axs[subplot_id] if num_sub_plots > 1 else self.plotter_axs
            data, label, legend = y_getters[subplot_id]()
            curr_plotter_ax.set_ylabel(label)
            subplot_lines = []
            subplot_y_buffer = []
            for i in range(len(data)):
                plotter_ln, = curr_plotter_ax.plot([], [], '-')
                if len(legend) == len(data):
                    plotter_ln.set_label(legend[i])
                subplot_lines.append(plotter_ln)
                subplot_y_buffer.append(deque(maxlen=max_data_points))
            curr_plotter_ax.legend(bbox_to_anchor=(-0.1, 1), ncol=3, loc='lower left')
            self.plotter_lines.append(subplot_lines)
            self.plotter_y_buffer.append(subplot_y_buffer)
        return self.plotter_lines

    def plotter_update(self, frame, x_getter, y_getters):
        if not self.is_playing:
            return self.plotter_lines

        self.plotter_x_buffer.append(x_getter())
        for (subplot_id, subplot_y_getter) in enumerate(y_getters):
            line_id = 0
            y_data, label, legend = subplot_y_getter()
            for data_value in y_data:
                self.plotter_y_buffer[subplot_id][line_id].append(data_value)
                self.plotter_lines[subplot_id][line_id].set_data(self.plotter_x_buffer,
                                                                 self.plotter_y_buffer[subplot_id][line_id])
                line_id += 1

            curr_plotter_ax = self.plotter_axs[subplot_id] if len(y_getters) > 1 else self.plotter_axs
            curr_plotter_ax.relim()
            curr_plotter_ax.autoscale_view()
        return self.plotter_lines

    def getplt_rewards(self):
        reward_terms = []
        legend = []
        for reward_name, reward_cfg in self.reward_terms.items():
            legend.append(reward_name)
            reward_function = getattr(REWARDS, 'reward_' + reward_name)
            reward_sigma = reward_cfg[0]
            reward_tolerance = reward_cfg[1]
            reward_terms.append(
                reward_function(self.data, reward_sigma, reward_tolerance).detach().cpu()[self.focus_env])
        label = "rewards"
        return np.array(reward_terms), label, legend
