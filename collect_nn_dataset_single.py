from rough_terrain_env import RoughTerrainEnv
import pybullet as p  
import numpy as np
import time
import os

# ---------------------------
# CONFIGURATION
# ---------------------------
NUM_EPISODES        = 300    
STEPS_PER_EPISODE   = 500  
HEIGHT_WINDOW_SIZE  = 64    
ACTION_HOLD_STEPS   = 100   
USE_GUI             = False  
OUTPUT_FILE         = "nn_dataset_paper_compliant.npz"

# ---------------------------
#  UTIL FUNCTIONS
# ---------------------------

def get_nn_input_state(obs):
    """
    INPUT (X): The R^9 vector the robot 'sees'.
    """
    g = np.asarray(obs["gravity_body"], dtype=np.float32).ravel()
    v = np.asarray(obs["v_body"], dtype=np.float32).ravel()
    w = np.asarray(obs["w_body"], dtype=np.float32).ravel()
    return np.concatenate([g, v, w]).astype(np.float32)

def get_ground_truth_pose(state):
    """
    TARGET (Y): The actual pose (x, y, z, yaw) from the simulator internals.
    """

    pos = state["position"] # (3,)
    

    quat = state["orientation_quat"]
    euler = p.getEulerFromQuaternion(quat)
    yaw = euler[2] # [roll, pitch, yaw]
    
    # Return [x, y, z, yaw]
    return np.array([pos[0], pos[1], pos[2], yaw], dtype=np.float32)

def sample_random_action(env):

    if np.random.rand() < 0.8:
        # Forward: 0 to Max
        throttle = np.random.uniform(0, env.max_throttle)
    else:
        # Reverse: -Max to 0 
        throttle = np.random.uniform(-env.max_throttle, 0)
        
    steering = np.random.uniform(-env.max_steering, env.max_steering)
    return np.array([throttle, steering], dtype=np.float32)

# ---------------------------
# MAIN DATA COLLECTION
# ---------------------------

def collect_full_dataset():
    print(f"\n=== Collecting {NUM_EPISODES} episodes ===")

    env = RoughTerrainEnv(
        gui=USE_GUI,
        log_data=False,
        max_steps=STEPS_PER_EPISODE*10,
        height_window_size=HEIGHT_WINDOW_SIZE,
    )

    recorded_maps    = []  
    recorded_states  = []  
    recorded_actions = []  
    recorded_targets = []  

    total_steps = 0

    for ep in range(NUM_EPISODES):
        obs = env.reset(random_start=True)
        
        # Get initial Ground Truth
        current_gt_state = env.get_state() 
        curr_pose = get_ground_truth_pose(current_gt_state)
        
        done = False
        step = 0
        
        current_action = sample_random_action(env)
        steps_on_current_action = 0

        # Pre-process initial inputs
        curr_map = obs["height_patch"].astype(np.float32).copy()
        curr_nn_state = get_nn_input_state(obs)


        # --- NEW COLLECTION LOOP (0.5s Horizon) ---
        while not done and step < STEPS_PER_EPISODE:
            
            #  Sample Random Action 
            current_action = sample_random_action(env)

            # Capture Starting State 
            curr_map = obs["height_patch"].astype(np.float32).copy()
            curr_nn_state = get_nn_input_state(obs)
            curr_pose = get_ground_truth_pose(env.get_state())


            for _ in range(5): 
                obs_next, reward, done, info = env.step(current_action)
                if done: 
                    break 
                if USE_GUI:
                    time.sleep(1.0 / 240.0)

            # Capture Final State (The Target)
            next_gt_state = env.get_state()
            next_pose = get_ground_truth_pose(next_gt_state)
            next_nn_state = get_nn_input_state(obs_next)

            #  Calculate Target (Delta Pose over 0.5s)
            delta_pose = next_pose - curr_pose
            
            #  Yaw Wrapping
            if delta_pose[3] > np.pi:  delta_pose[3] -= 2*np.pi
            if delta_pose[3] < -np.pi: delta_pose[3] += 2*np.pi

            # Target vector
            target_vec = np.concatenate([delta_pose, next_nn_state])

            #  Store Data
            recorded_maps.append(curr_map)
            recorded_states.append(curr_nn_state)
            recorded_actions.append(current_action)
            recorded_targets.append(target_vec)

            #  Update pointers for next loop
            obs = obs_next
            step += 1
            total_steps += 1



        if (ep+1) % 1 == 0:
            print(f"Episode {ep+1} done. Steps: {step}/{STEPS_PER_EPISODE} (Total: {total_steps})")

    # ------------------------------------
    # Save
    # ------------------------------------
    print("\n=== Stacking Arrays ===")
    maps_np = np.stack(recorded_maps, axis=0)       
    states_np = np.stack(recorded_states, axis=0)   
    actions_np = np.stack(recorded_actions, axis=0) 
    targets_np = np.stack(recorded_targets, axis=0) 
    
    print(" Maps:", maps_np.shape)     # (N, 64, 64)
    print(" States:", states_np.shape) # (N, 9)
    print(" Actions:", actions_np.shape) # (N, 2)
    print(" Targets:", targets_np.shape) # (N, 13)

    np.savez_compressed(
        OUTPUT_FILE,
        maps=maps_np,
        states=states_np,
        actions=actions_np,
        targets=targets_np
    )

    env.disconnect()
    print(f"\nDataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    collect_full_dataset()