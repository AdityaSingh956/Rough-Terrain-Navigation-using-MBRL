import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#ROUGH TERRAIN
# ---------- DynamicsModel ----------

class DynamicsModel(nn.Module):
    def __init__(self, state_dim=9, action_dim=2, target_dim=13):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.target_dim = target_dim

        # CNN for heightmap: 1x64x64 -> 8x32x32 -> 4x16x16
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1)

        conv_feat_dim = 4 * 16 * 16        # 1024
        fc_in_dim = conv_feat_dim + state_dim + action_dim   # 1024 + 9 + 2 = 1035

        self.fc1 = nn.Linear(fc_in_dim, 2048)
        self.dropout1 = nn.Dropout(p=0.1)

        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=256,
            batch_first=True
        )

        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(p=0.1)

        # mean + logvar: 13 + 13 = 26
        self.fc_out = nn.Linear(256, 2 * target_dim)

        self.relu = nn.ReLU()

    def forward(self, maps_seq, states_seq, actions_seq, hidden=None):
        B, T, C, H, W = maps_seq.shape
        assert C == 1 and H == 64 and W == 64

        # CNN over map per timestep
        x = maps_seq.view(B * T, 1, H, W)   # (B*T,1,64,64)
        x = self.relu(self.conv1(x))        # (B*T,8,32,32)
        x = self.relu(self.conv2(x))        # (B*T,4,16,16)
        x = x.view(B * T, -1)               # (B*T,1024)

        # concat state + action
        s = states_seq.view(B * T, -1)      # (B*T,9)
        a = actions_seq.view(B * T, -1)     # (B*T,2)
        x = torch.cat([x, s, a], dim=-1)    # (B*T,1035)

        x = self.relu(self.fc1(x))          # (B*T,2048)
        x = self.dropout1(x)

        x = x.view(B, T, -1)                # (B,T,2048)

        x, new_hidden = self.lstm(x, hidden)   # (B,T,256)

        x = self.relu(self.fc2(x))          # (B,T,256)
        x = self.dropout2(x)

        out = self.fc_out(x)                # (B,T,26)
        mu, log_var = torch.chunk(out, 2, dim=-1)  # each (B,T,13)

        return mu, log_var, new_hidden


# ---------- Wrapper ----------

class DynamicsWrapper:
    def __init__(self,
                 model_path="dynamics_multistep.pt",
                 stats_path="normalization_stats.npz",
                 device="cpu"):
        self.device = torch.device(device)

        # load normalization stats
        stats = np.load(stats_path)
        self.maps_mean   = torch.tensor(stats["maps_mean"],   dtype=torch.float32, device=self.device)
        self.maps_std    = torch.tensor(stats["maps_std"],    dtype=torch.float32, device=self.device)
        self.state_mean  = torch.tensor(stats["state_mean"],  dtype=torch.float32, device=self.device)
        self.state_std   = torch.tensor(stats["state_std"],   dtype=torch.float32, device=self.device)
        self.action_mean = torch.tensor(stats["action_mean"], dtype=torch.float32, device=self.device)
        self.action_std  = torch.tensor(stats["action_std"],  dtype=torch.float32, device=self.device)
        self.target_mean = torch.tensor(stats["target_mean"], dtype=torch.float32, device=self.device)
        self.target_std  = torch.tensor(stats["target_std"],  dtype=torch.float32, device=self.device)

        # load trained model
        self.model = DynamicsModel().to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def predict(self, map_64, state_9, action_2, particles=1):

        map_t   = torch.tensor(map_64,   dtype=torch.float32, device=self.device).view(1, 1, 1, 64, 64)
        state_t = torch.tensor(state_9,  dtype=torch.float32, device=self.device).view(1, 1, 9)
        action_t= torch.tensor(action_2, dtype=torch.float32, device=self.device).view(1, 1, 2)

        # normalize
        map_t   = (map_t   - self.maps_mean)   / (self.maps_std   + 1e-8)
        state_t = (state_t - self.state_mean)  / (self.state_std  + 1e-8)
        action_t= (action_t- self.action_mean) / (self.action_std + 1e-8)
 
        mus = []
        sigmas = []

        for _ in range(particles):
            self.model.train() # Enable dropout
            mu_n, log_var_n, _ = self.model(map_t, state_t, action_t, hidden=None)
            sigma_n = torch.exp(0.5 * log_var_n)

            mu_o    = mu_n[0, 0] * self.target_std + self.target_mean
            sigma_o = sigma_n[0, 0] * self.target_std

            mus.append(mu_o.cpu().numpy())
            sigmas.append(sigma_o.cpu().numpy())

        return np.stack(mus, axis=0), np.stack(sigmas, axis=0)

    def predict_torch(self, map_64, state_9, action_2, deterministic=True):

        # 1. Normalize
        map_in = (map_64 - self.maps_mean) / (self.maps_std + 1e-8)
        state_in = (state_9 - self.state_mean) / (self.state_std + 1e-8)
        action_in = (action_2 - self.action_mean) / (self.action_std + 1e-8)

        # 2. Add Time dimension 
        map_in = map_in.unsqueeze(1)    # (B, 1, 1, 64, 64)
        state_in = state_in.unsqueeze(1) # (B, 1, 9)
        action_in = action_in.unsqueeze(1) # (B, 1, 2)

        # 3. Forward Pass
        mu_n, log_var_n, _ = self.model(map_in, state_in, action_in, hidden=None)
        
        # 4. Reparameterization 
        if not deterministic:
            std_n = torch.exp(0.5 * log_var_n)
            eps = torch.randn_like(std_n)
            prediction_n = mu_n + eps * std_n
        else:
            prediction_n = mu_n 
            
        prediction_n = prediction_n.squeeze(1)

        # 5. Denormalize
        prediction = prediction_n * self.target_std.view(1, -1) + self.target_mean.view(1, -1)
        
        return prediction