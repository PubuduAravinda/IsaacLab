# export_himloco_full.py — Export actor + encoder from checkpoint
import torch
import torch.nn as nn

checkpoint_path = "/home/sripu715/IsaacLab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/go1_himloco/2026-02-03_21-23-53/model_500.pt"
output_actor = "/home/sripu715/IsaacLab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/go1_himloco/2026-02-03_21-23-53/himloco_policy_full.pt"
output_encoder = "/home/sripu715/IsaacLab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/go1_himloco/2026-02-03_21-23-53/himloco_encoder.pt"

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Debug keys
print("Checkpoint keys:", list(checkpoint.keys()))

# Load state_dict (handle different keys)
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
elif 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
elif 'actor' in checkpoint:
    state_dict = checkpoint['actor']
else:
    state_dict = checkpoint  # flat dict
    print("Assuming flat checkpoint as state_dict")

# === Actor (64D input) ===
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 12), nn.Tanh()  # tanh output for bounded actions
        )

    def forward(self, obs):
        return self.net(obs)

actor = Actor()
actor.load_state_dict(state_dict, strict=False)  # ignore extra keys like critic
actor.eval()

scripted_actor = torch.jit.script(actor)
scripted_actor.save(output_actor)
print(f"Exported actor (policy) to: {output_actor}")

# === Encoder (225D history input → 19D latent) ===
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(225, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 19),
        )

    def forward(self, x):
        return self.net(x)

encoder = Encoder()
encoder.load_state_dict(state_dict, strict=False)  # ignore extra keys
encoder.eval()

scripted_encoder = torch.jit.script(encoder)
scripted_encoder.save(output_encoder)
print(f"Exported encoder to: {output_encoder}")

# export_himloco_full.py — Export actor + encoder for full 64D
# export_himloco_full.py — Export actor + encoder from your latest checkpoint
#
# import torch
# import torch.nn as nn
#
# checkpoint_path = "/home/sripu715/IsaacLab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/go1_himloco/2026-02-03_21-23-53/model_500.pt"
# output_actor = "/home/sripu715/IsaacLab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/go1_himloco/2026-02-03_21-23-53/himloco_policy_full.pt"
# output_encoder = "/home/sripu715/IsaacLab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/go1_himloco/2026-02-03_21-23-53/himloco_encoder.pt"
#
# # Load checkpoint
# checkpoint = torch.load(checkpoint_path, map_location="cpu")
#
# # Debug keys
# print("Checkpoint keys:", list(checkpoint.keys()))
#
# # Load model weights (usually 'model_state_dict' in RSL-RL)
# if 'model_state_dict' in checkpoint:
#     model_state_dict = checkpoint['model_state_dict']
#     print("Loaded from 'model_state_dict'")
# else:
#     raise KeyError("No 'model_state_dict' found. Check checkpoint keys above.")
#
# # === Actor (64D input) ===
# class Actor(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(64, 512),
#             nn.ELU(),
#             nn.Linear(512, 256),
#             nn.ELU(),
#             nn.Linear(256, 128),
#             nn.ELU(),
#             nn.Linear(128, 12),
#             nn.Tanh()
#         )
#
#     def forward(self, obs):
#         return self.net(obs)
#
# actor = Actor()
# actor.load_state_dict(model_state_dict, strict=False)
# actor.eval()
# scripted_actor = torch.jit.script(actor)
# scripted_actor.save(output_actor)
# print(f"Exported actor (policy) to: {output_actor}")
#
# # === Encoder (225D input → 19D latent) ===
# class Encoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(225, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 19),
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
# encoder = Encoder()
# encoder.load_state_dict(model_state_dict, strict=False)  # encoder weights are usually in same dict
# encoder.eval()
# scripted_encoder = torch.jit.script(encoder)
# scripted_encoder.save(output_encoder)
# print(f"Exported encoder to: {output_encoder}")



# # save_sim-to-real_models.py — FINAL VERSION (handles flat or nested state_dict)
# import torch
# import torch.nn as nn
#
# checkpoint_path = "/home/sripu715/IsaacLab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/go1_himloco/walk_farward_2026-01-13_14-02-13/model_100.pt"
# output_path = "/home/sripu715/IsaacLab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/go1_himloco/walk_farward_2026-01-13_14-02-13/himloco_policy_45d.pt"
#
# # Load checkpoint
# checkpoint = torch.load(checkpoint_path, map_location="cpu")
#
# # Debug: print top-level keys to see the structure
# print("Checkpoint keys:", list(checkpoint.keys()) if isinstance(checkpoint, dict) else "Not a dict")
#
# # Try common RSL-RL key patterns
# if isinstance(checkpoint, dict):
#     possible_keys = ["actor_state_dict", "model_state_dict", "actor", "state_dict", "model"]
#     actor_state_dict = None
#     for key in possible_keys:
#         if key in checkpoint:
#             actor_state_dict = checkpoint[key]
#             print(f"Found actor state_dict under key: '{key}'")
#             break
#     if actor_state_dict is None:
#         # If no key found, assume the checkpoint IS the flat state_dict
#         actor_state_dict = checkpoint
#         print("Using flat checkpoint as state_dict")
# else:
#     raise ValueError("Checkpoint is not a dict — unexpected format")
#
# # Re-create actor with 45D input (no embedding)
# class Actor45D(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(45, 512),
#             nn.ELU(),
#             nn.Linear(512, 256),
#             nn.ELU(),
#             nn.Linear(256, 128),
#             nn.ELU(),
#             nn.Linear(128, 12),
#             nn.Tanh()  # assuming policy outputs [-1,1]
#         )
#
#     def forward(self, obs):
#         return self.net(obs)
#
# model = Actor45D()
# model.load_state_dict(actor_state_dict, strict=False)  # strict=False ignores missing/extra keys
# model.eval()
#
# # Script it
# scripted_model = torch.jit.script(model)
# scripted_model.save(output_path)
# print(f"\nSUCCESS! Exported 45D policy to:\n{output_path}")
# print("Copy this file to Raspberry Pi and use in himloco_sim_to_real.py")