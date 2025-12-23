import numpy as np
import torch
import time
import pybullet as p

from rough_terrain_env import RoughTerrainEnv
from dynamics_wrapper import DynamicsWrapper
from algorithms import CMUNavigationOptimizer 

def obs_to_network_inputs(obs):
    map_64 = obs["height_patch"].astype(np.float32)
    state_9 = np.concatenate([
        obs["gravity_body"], 
        obs["v_body"], 
        obs["w_body"]
    ]).astype(np.float32)
    return map_64, state_9

def get_robot_pose(env):
    st = env.get_state()
    curr_yaw = p.getEulerFromQuaternion(st["orientation_quat"])[2]
    return np.array([st["position"][0], st["position"][1], st["position"][2], curr_yaw])

def simple_tracker(curr_pose, ref_pose):

    x, y, yaw = curr_pose[0], curr_pose[1], curr_pose[3]
    x_ref, y_ref = ref_pose[0], ref_pose[1]
    
    dx = x_ref - x
    dy = y_ref - y
    dist = np.sqrt(dx**2 + dy**2)
    
    desired_heading = np.arctan2(dy, dx)
    heading_error = desired_heading - yaw
    # Wrap to [-pi, pi]
    heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
    
    k_v = 2.0
    k_yaw = 2.5
    
    throttle = np.clip(k_v * dist, -10.0, 10.0)
    steering = np.clip(k_yaw * heading_error, -0.6, 0.6)
    
    return np.array([throttle, steering])




def main():
    # 1. Setup
    env = RoughTerrainEnv(gui=True, log_data=False, height_window_size=64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    dynamics = DynamicsWrapper(
        model_path="dynamics_multistep.pt",
        stats_path="normalization_stats.npz", 
        device=device
    )
    
    # 2. Initialize Optimizer with LONG HORIZON

    PLAN_HORIZON = 100
    optimizer = CMUNavigationOptimizer(dynamics, planning_horizon=PLAN_HORIZON, device=device)
    optimizer.set_goal(env.goal_xy)
    
    obs = env.reset(random_start=True)
    
    # A. INPUTS
    map_64, state_9 = obs_to_network_inputs(obs)
    curr_pose = get_robot_pose(env)

    print(f"\n=== PLANNING ONCE (Horizon {PLAN_HORIZON}) ===")
    print("Optimization starting... (This might take 10-20 seconds)...")
    
    start_t = time.time()

    # B. OPTIMIZE ONCE 
    
    # 1.  Warm Start 
    initial_guess = torch.zeros((PLAN_HORIZON, 2), device=device)
    initial_guess[:, 0] = 2.0  # Force Throttle = 5.0
    
    # 2. Pass it to the optimizer
    best_actions, nominal_path = optimizer.optimize(
        curr_pose, state_9, map_64, initial_actions=initial_guess
    )

    ####

    #####
    
    
    print(f"Optimization Finished in {time.time() - start_t:.2f}s")



    

    ##########Debugging things#############


    path_np = nominal_path[:, 0, :]
    

    plan_start_xy = path_np[0, :2]   
    plan_end_xy   = path_np[-1, :2]  
    

    dist_start_to_goal = np.linalg.norm(plan_start_xy - env.goal_xy)
    dist_end_to_goal   = np.linalg.norm(plan_end_xy - env.goal_xy)
    
    print("\n" + "="*40)
    print("      PLANNER DIAGNOSTICS      ")
    print("="*40)
    print(f"Goal Coordinate    : {env.goal_xy}")
    print(f"Plan Start (t=0)   : {plan_start_xy} (Dist: {dist_start_to_goal:.2f}m)")
    print(f"Plan End   (t=H)   : {plan_end_xy} (Dist: {dist_end_to_goal:.2f}m)")
    
    improvement = dist_start_to_goal - dist_end_to_goal
    if improvement > 0:
        print(f"SUCCESS: Plan moves {improvement:.2f}m closer to goal.")
    else:
        print(f"FAILURE: Plan moves {abs(improvement):.2f}m AWAY from goal.")
    print("="*40 + "\n")


    ###########
    

    path_np = nominal_path[:, 0, :]
    for t in range(len(path_np) - 1):
        p.addUserDebugLine(
            path_np[t][:3] + np.array([0,0,0.5]), 
            path_np[t+1][:3] + np.array([0,0,0.5]), 
            lineColorRGB=[0, 1, 0], lineWidth=3.0, lifeTime=0 
        )
    
    print("=== EXECUTING TRAJECTORY ===")
    

    
    for t in range(PLAN_HORIZON):

        target_pose = path_np[t] # [x, y, z, yaw]
        

        nom_action = best_actions[t] # [throttle, steering]
        

        real_pose = get_robot_pose(env)
        

        correction = simple_tracker(real_pose, target_pose)
        final_action = nom_action + correction


  

        
        #  Execute
        obs, reward, done, info = env.step(final_action)
        
        # Log progress
        dist_to_goal = np.linalg.norm(real_pose[:2] - env.goal_xy)
        print(f"Step {t+1}/{PLAN_HORIZON} | Dist: {dist_to_goal:.2f}m | Action: {final_action}")
        
        time.sleep(0.05) 


        ######
        
        if done:
            print("Goal Reached or Episode Ended!")
            break


    time.sleep(2.0)
    env.disconnect()

if __name__ == "__main__":
    main()