
import pybullet as p
import pybullet_data
import time
import numpy as np
import os

class RoughTerrainEnv:
    def __init__(
        self,
        terrain_npy_path="jezero_128x128_1m.npy",
        texture_path="jezero_texture_128x128.png",
        gui=True,
        max_steps=500,
        log_data=False,
        height_window_size=16,
    ):


        self.gui = gui
        self.max_steps = max_steps
        self.log_data = log_data
        self.height_window_size = height_window_size

        # 1. Connect to PyBullet
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # 2. Terrain Parameters
        self.terrain_size = 128
        self.pixel_size_m = 1.0096633268711
        self.terrain_scale_xy = self.pixel_size_m

        # 3. Load Heightmap
        assert os.path.exists(terrain_npy_path), f"Missing {terrain_npy_path}"
        heights = np.load(terrain_npy_path).astype(np.float32)
        assert heights.shape == (self.terrain_size, self.terrain_size)
        self.heights = heights

        # Fix terrain height range 
        hmin, hmax = float(np.nanmin(self.heights)), float(np.nanmax(self.heights))
        hrange = hmax - hmin
        print(f"[heightmap] min={hmin:.3f} max={hmax:.3f} range={hrange:.3f}")
        if hrange < 0.5:
            print("[heightmap] small range; applying vertical exaggeration")
            self.heights *= 0.3

        hmin_f, hmax_f = float(np.nanmin(self.heights)), float(np.nanmax(self.heights))
        self.z_offset = (hmin_f + hmax_f) / 2.0
        print(f"[heightmap] Final min={hmin_f:.3f} max={hmax_f:.3f}")

        # Flatten for PyBullet
        height_data = self.heights.reshape(-1).tolist()

        # 4. Create Terrain 
        print("Creating collision shape...")
        terrain_shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[self.terrain_scale_xy, self.terrain_scale_xy, 1.0],
            heightfieldData=height_data,
            heightfieldTextureScaling=self.terrain_size - 1,
            numHeightfieldRows=self.terrain_size,
            numHeightfieldColumns=self.terrain_size,
        )

        print("Creating terrain body...")
        self.terrain_body = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=terrain_shape
        )

        p.resetBasePositionAndOrientation(
            self.terrain_body, [0, 0, self.z_offset], [0, 0, 0, 1]
        )

        p.changeDynamics(
            self.terrain_body, -1, 
            lateralFriction=1.0, 
            restitution=0.0,
            contactProcessingThreshold=0.005,
            collisionMargin=0.001
        )

        # Texture
        if os.path.exists(texture_path):
            print("Loading Mars texture...")
            texture_id = p.loadTexture(texture_path)
            p.changeVisualShape(self.terrain_body, -1, textureUniqueId=texture_id)
        else:
            print(f"[warn] texture {texture_path} not found")

        # placeholders
        self.robot_id = None
        self.left_wheels = []
        self.right_wheels = []

        # Action limits
        self.max_throttle = 10.0
        self.max_steering = 0.6
        self.turn_gain = 1.0

        self.goal_xy = np.array([3.0, 4.0])
        self.episode_log = []
        self.step_count = 0

    # ------------------------------------------------------------------
    # helpers for height lookup and local patch
    # ------------------------------------------------------------------
    def world_to_grid(self, x, y):

        i = int(np.round(x / self.terrain_scale_xy + self.terrain_size / 2))
        j = int(np.round(y / self.terrain_scale_xy + self.terrain_size / 2))
        i = np.clip(i, 0, self.terrain_size - 1)
        j = np.clip(j, 0, self.terrain_size - 1)
        return j, i  

    def get_height_at(self, x, y):
        row, col = self.world_to_grid(x, y)
        return float(self.heights[row, col])

    def get_local_height_patch(self, x, y):

        ws = self.height_window_size
        row_c, col_c = self.world_to_grid(x, y)
        half = ws // 2

        row_min = max(0, row_c - half)
        row_max = min(self.terrain_size, row_c + half)
        col_min = max(0, col_c - half)
        col_max = min(self.terrain_size, col_c + half)

        patch = np.zeros((ws, ws), dtype=np.float32)
        sub = self.heights[row_min:row_max, col_min:col_max]
        patch[
            (row_min - (row_c - half)) : (row_max - (row_c - half)),
            (col_min - (col_c - half)) : (col_max - (col_c - half)),
        ] = sub
        return patch

    # ------------------------------------------------------------------
    # core env API
    # ------------------------------------------------------------------
    def reset(self, random_start=True):

        if self.robot_id is not None:
            p.removeBody(self.robot_id)

        if random_start:

            for _ in range(20):
                start_x = np.random.uniform(-5.0, 5.0)
                start_y = np.random.uniform(-5.0, 5.0)
                h_c = self.get_height_at(start_x, start_y)

                h_n = self.get_height_at(start_x + 0.5, start_y)
                if abs(h_n - h_c) < 0.4:  
                    break
        else:
            start_x, start_y = 0.0, 0.0

        hz = self.get_height_at(start_x, start_y)
        start_z = hz + 0.4  

        start_pos = [start_x, start_y, start_z]
        start_orn = p.getQuaternionFromEuler([0, 0, 0])

        print("Loading robot (Husky)...")
        self.robot_id = p.loadURDF(
            "husky/husky.urdf",
            start_pos,
            start_orn,
            globalScaling=1.0,
        )

        # Identify wheel joints
        self.left_wheels = []
        self.right_wheels = []

        num_joints = p.getNumJoints(self.robot_id)
        for j in range(num_joints):
            info = p.getJointInfo(self.robot_id, j)
            name = info[1].decode("utf-8")
            if "wheel" in name:
                if "left" in name:
                    self.left_wheels.append(j)
                elif "right" in name:
                    self.right_wheels.append(j)


                p.setJointMotorControl2(
                    self.robot_id, j, p.VELOCITY_CONTROL, targetVelocity=0, force=0
                )
                p.changeDynamics(self.robot_id, j, lateralFriction=1.0)


        print("Settling robot...")

        for _ in range(100):
            p.stepSimulation()
            if len(p.getContactPoints(self.robot_id)) > 0:
                break
        

        for _ in range(200):
            p.stepSimulation()
            lin, ang = p.getBaseVelocity(self.robot_id)
            if np.linalg.norm(lin) < 0.1 and np.linalg.norm(ang) < 0.1:
                break
        
        # Reset camera to settled position
        final_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        p.resetDebugVisualizerCamera(
            cameraDistance=5.0,
            cameraYaw=45.0,
            cameraPitch=-35.0,
            cameraTargetPosition=final_pos,
        )

        self.step_count = 0
        self.episode_log = []

        obs = self.get_observation()
        return obs

    def step(self, action):


        throttle = float(np.clip(action[0], -self.max_throttle, self.max_throttle))
        steering = float(np.clip(action[1], -self.max_steering, self.max_steering))

        self._apply_action(throttle, steering)

        # p.stepSimulation()
        for _ in range(24): 
            p.stepSimulation()

        self.step_count += 1

        obs_next = self.get_observation()
        state_next = self.get_state()


        pos = state_next["position"]
        xy = np.array(pos[:2])
        dist_to_goal = np.linalg.norm(xy - self.goal_xy)
        reward = -dist_to_goal

        # termination conditions
        done = False
        info = {"dist_to_goal": dist_to_goal}


        base_z = pos[2]
        ground_z = self.get_height_at(pos[0], pos[1])
        if base_z - ground_z > 2.0: 
            done = True
            info["termination"] = "unstable_height"

        # Check Flipping 
        rot_mat = np.array(p.getMatrixFromQuaternion(state_next["orientation_quat"])).reshape(3,3)
        up_vec = rot_mat[:, 2] # local Z axis
        if up_vec[2] < 0.0: # If Z is pointing down
            done = True
            info["termination"] = "flipped"

        if dist_to_goal < 1.0:
            done = True
            info["termination"] = "goal_reached"
        if self.step_count >= self.max_steps:
            done = True
            info["termination"] = "max_steps"

        # simple logging
        if self.log_data:
            self.episode_log.append(
                {
                    "obs": obs_next,
                    "action": np.array([throttle, steering], dtype=np.float32),
                    "reward": reward,
                    "done": done,
                }
            )

        return obs_next, reward, done, info

    # ------------------------------------------------------------------
    # state and observation
    # ------------------------------------------------------------------
    def get_state(self):

        assert self.robot_id is not None, "Robot not loaded yet"
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)

        state = {
            "position": np.array(pos, dtype=np.float32),
            "orientation_quat": np.array(orn, dtype=np.float32),
            "linear_velocity": np.array(lin_vel, dtype=np.float32),
            "angular_velocity": np.array(ang_vel, dtype=np.float32),
        }
        return state

    def get_observation(self):

        st = self.get_state()
        pos = st["position"]
        quat = st["orientation_quat"]
        lin_vel_world = st["linear_velocity"]
        ang_vel_world = st["angular_velocity"]

        # rotation matrix from quaternion
        rot = np.array(p.getMatrixFromQuaternion(quat), dtype=np.float32).reshape(3, 3)

        g_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        g_body = rot.T @ g_world  

        v_body = rot.T @ lin_vel_world
        w_body = rot.T @ ang_vel_world

        height_patch = self.get_local_height_patch(pos[0], pos[1])

        obs = {
            "gravity_body": g_body,
            "v_body": v_body,
            "w_body": w_body,
            "height_patch": height_patch,
            "position_xy": pos[:2],
        }
        return obs

    # ------------------------------------------------------------------
    # actuation and simple tracking controller
    # ------------------------------------------------------------------

    def _apply_action(self, throttle, steering):

        left_vel = throttle - steering * 2.0
        right_vel = throttle + steering * 2.0
        

        motor_force = 100.0

        for j in self.left_wheels:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=j,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=left_vel,
                force=motor_force,
            )

        for j in self.right_wheels:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=j,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=right_vel,
                force=motor_force,
            )

    def tracking_controller(self, ref_pose, state=None):

        if state is None:
            state = self.get_state()

        pos = state["position"]
        quat = state["orientation_quat"]
        yaw = p.getEulerFromQuaternion(quat)[2]

        x, y = pos[0], pos[1]
        x_ref, y_ref, yaw_ref = ref_pose

        # errors
        dx = x_ref - x
        dy = y_ref - y
        dist = np.hypot(dx, dy)
        desired_heading = np.arctan2(dy, dx)
        heading_error = desired_heading - yaw

        # wrap to [-pi, pi]
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        # simple PD-like law
        k_v = 2.0
        k_yaw = 2.5

        throttle = np.clip(k_v * dist, -self.max_throttle, self.max_throttle)
        steering = np.clip(k_yaw * heading_error, -self.max_steering, self.max_steering)

        return np.array([throttle, steering], dtype=np.float32)

    # ------------------------------------------------------------------
    # dataset helpers
    # ------------------------------------------------------------------
    def get_episode_dataset(self):

        if not self.episode_log:
            return None

        actions = np.stack([step["action"] for step in self.episode_log], axis=0)
        rewards = np.array([step["reward"] for step in self.episode_log], dtype=np.float32)
        dones = np.array([step["done"] for step in self.episode_log], dtype=np.bool_)


        gravity = np.stack(
            [step["obs"]["gravity_body"] for step in self.episode_log], axis=0
        )
        v_body = np.stack(
            [step["obs"]["v_body"] for step in self.episode_log], axis=0
        )
        w_body = np.stack(
            [step["obs"]["w_body"] for step in self.episode_log], axis=0
        )
        height_patches = np.stack(
            [step["obs"]["height_patch"] for step in self.episode_log], axis=0
        )

        dataset = {
            "gravity_body": gravity,
            "v_body": v_body,
            "w_body": w_body,
            "height_patches": height_patches,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
        }
        return dataset

    def disconnect(self):
        p.disconnect(self.client_id)

if __name__ == "__main__":
    env = RoughTerrainEnv(gui=True, log_data=True)

    print("--- Random policy rollout ---")
    obs = env.reset(random_start=True)
    done = False
    total_reward = 0.0

    while not done:
        # random actions
        action = np.array(
            [
                np.random.uniform(-env.max_throttle, env.max_throttle),
                np.random.uniform(-env.max_steering, env.max_steering),
            ],
            dtype=np.float32,
        )
        obs, reward, done, info = env.step(action)
        total_reward += reward
        time.sleep(1.0 / 240.0)

    print("Episode finished, total reward:", total_reward)
    env.disconnect()