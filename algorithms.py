import torch
import numpy as np

class CMUNavigationOptimizer:
    def __init__(self, dynamics_wrapper, planning_horizon=20, device="cuda"):
        self.dyn = dynamics_wrapper
        self.H = planning_horizon
        self.device = device
        
        # Optimization Hyperparameters
        self.lr = 0.1
        self.U_divergence_limit= 300
        self.rho = 1.1                 # Multiplier update factor
        self.num_particles = 10        
        
        # This will be reset every optimize step, but initialized here for safety
        self.lambda_penalty = 1.0      
        
        # Default goal (will be overwritten by set_goal)
        self.goal = torch.tensor([20.0, 0.0], device=device)

    def set_goal(self, goal_xy):

        self.goal = torch.tensor(goal_xy, device=self.device, dtype=torch.float32)

    def differentiable_tracker(self, current_pose, ref_pose):


        x, y, yaw = current_pose[:, 0], current_pose[:, 1], current_pose[:, 3]
        x_ref, y_ref = ref_pose[:, 0], ref_pose[:, 1]
        
        # Calculate Errors
        dx = x_ref - x
        dy = y_ref - y
        dist = torch.sqrt(dx**2 + dy**2 + 1e-6)
        
        desired_heading = torch.atan2(dy, dx)
        heading_error = desired_heading - yaw
        
        # Wrap heading error to [-pi, pi]
        heading_error = (heading_error + torch.pi) % (2 * torch.pi) - torch.pi
        
        # Gains 
        k_v = 2.0
        k_yaw = 2.5
        
        # Control Law with Clamping
        throttle = torch.clamp(k_v * dist, -10.0, 10.0)
        steering = torch.clamp(k_yaw * heading_error, -0.6, 0.6)
        
        return torch.stack([throttle, steering], dim=1)



    def transition_state(self, pose, state_9, prediction_13):
        TIME_SCALE = 1.0  

        # World-frame deltas (from dataset)
        dx_world = prediction_13[:, 0] * TIME_SCALE
        dy_world = prediction_13[:, 1] * TIME_SCALE
        dz_world = prediction_13[:, 2] * TIME_SCALE
        dyaw     = prediction_13[:, 3] * TIME_SCALE


        new_x = pose[:, 0] + dx_world
        new_y = pose[:, 1] + dy_world
        new_z = pose[:, 2] + dz_world

        new_yaw = pose[:, 3] + dyaw
        new_yaw = (new_yaw + torch.pi) % (2 * torch.pi) - torch.pi

        new_pose = torch.stack([new_x, new_y, new_z, new_yaw], dim=1)


        new_state_9 = prediction_13[:, 4:13]

        return new_pose, new_state_9



    

    def rollout_trajectory(self, start_pose, start_state_9, map_patch, actions, closed_loop=False, nominal_traj=None):

        H = self.H

        B = self.num_particles if closed_loop else 1


        curr_pose  = start_pose.repeat(B, 1)        # (B, 4)
        curr_state = start_state_9.repeat(B, 1)     # (B, 9)

        curr_map   = map_patch.expand(B, 1, 64, 64) 

        traj_poses = []
        
        for t in range(H):

            if closed_loop:

                ref_pose = nominal_traj[t].repeat(B, 1) 
                correction = self.differentiable_tracker(curr_pose, ref_pose)
                

                base_act = actions[t].unsqueeze(0).repeat(B, 1) 
                act = base_act + correction
            else:

                act = actions[t].unsqueeze(0) # (1, 2)
            
            # B. Predict Dynamics
            # Deterministic for Nominal (B=1), Stochastic for Closed Loop (B=N)
            is_deterministic = not closed_loop
            pred_13 = self.dyn.predict_torch(curr_map, curr_state, act, deterministic=is_deterministic)
            
            # C. Integrate State
            curr_pose, curr_state = self.transition_state(curr_pose, curr_state, pred_13)
            traj_poses.append(curr_pose)
            
        return torch.stack(traj_poses) # (H, B, 4)


    def optimize(self, start_pose, start_state_9, map_patch, initial_actions=None):

        self.lambda_penalty = 1.0

        # Initialize actions 
        if initial_actions is None:
            actions = torch.zeros(self.H, 2, device=self.device, requires_grad=True)
        else:
            actions = initial_actions.clone().detach().to(self.device).requires_grad_(True)

        optimizer = torch.optim.Adam([actions], lr=self.lr)
        

        p_start = torch.tensor(start_pose, device=self.device, dtype=torch.float32).unsqueeze(0)
        s_start = torch.tensor(start_state_9, device=self.device, dtype=torch.float32).unsqueeze(0)
        m_start = torch.tensor(map_patch, device=self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


        # Augmented Lagrangian Loop
        for outer_i in range(3): 
            for inner_i in range(45): 
                optimizer.zero_grad()
                
                # 1. Forward Pass 
                nominal_poses = self.rollout_trajectory(p_start, s_start, m_start, actions, closed_loop=False)
                closed_loop_poses = self.rollout_trajectory(p_start, s_start, m_start, actions, closed_loop=True, nominal_traj=nominal_poses)
                
                # 2. Costs
                final_pose = nominal_poses[-1, 0, :2] 

                ##

                diff_goal = final_pose - self.goal
                
                cost_goal = torch.sum(diff_goal ** 2) 
                
                ##
                #cost_goal = torch.norm(final_pose - self.goal)
                
                diff = closed_loop_poses[:, :, :2] - nominal_poses[:, :, :2] 
                divergence = torch.mean(torch.sum(diff**2, dim=2)) 
                
                constraint = torch.relu(divergence - self.U_divergence_limit)
                loss = cost_goal + self.lambda_penalty * constraint
                #loss = cost_goal 
                
                # 3. Backward Pass
                loss.backward()
                optimizer.step()
                
                # --- DEBUG PRINT BLOCK ---
                with torch.no_grad():
                    pred_end_x = nominal_poses[-1, 0, 0].item()
                    pred_end_y = nominal_poses[-1, 0, 1].item()
                    
                    curr_action = actions[0].detach().cpu().numpy()
                    
                    grad = actions.grad[0].detach().cpu().numpy()
                    
                    print(f"Step [{outer_i}-{inner_i}]")
                    print(f"  Model Prediction (Final XY): ({pred_end_x:.2f}, {pred_end_y:.2f})")
                    print(f"  Costs: Goal={cost_goal.item():.4f} | Div={divergence.item():.4f} | Total={loss.item():.4f}")
                    print(f"  Action[0]: {curr_action} | Gradient[0]: {grad}")
                    print("-" * 30)
                # -------------------------

                # Clamp actions 
                with torch.no_grad():
                    actions[:, 0].clamp_(-10, 10) 
                    actions[:, 1].clamp_(-0.6, 0.6)

            # Update Penalty
            if constraint.item() > 0.01:
                self.lambda_penalty *= self.rho
                print(f"   >>> Penalty Increased: {self.lambda_penalty:.2f}")

        


        return actions.detach().cpu().numpy(), nominal_poses.detach().cpu().numpy()