#!/usr/bin/env python3
"""
Export HIMLoco policy and encoder for deployment on real Go1
Exports from dual checkpoint system:
  - model_XXXX.pt (contains actor/critic)
  - model_XXXX_himloco.pt (contains encoder_source, encoder_target, prototypes)
"""

import torch
import torch.nn as nn
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================
checkpoint_base = "/home/sripu715/IsaacLab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/go1_himloco/2026-02-21_19-33-22/model_24500"
output_dir = "/home/sripu715/IsaacLab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/go1_himloco/2026-02-21_19-33-22/go1_deployment"

policy_checkpoint = f"{checkpoint_base}.pt"
himloco_checkpoint = f"{checkpoint_base}_himloco.pt"

output_actor = f"{output_dir}/actor.pt"
output_encoder = f"{output_dir}/encoder.pt"


# ============================================================================
# ACTOR NETWORK (Policy)
# ============================================================================
class Actor(nn.Module):
    """
    HIMLoco Actor: 64D observation → 12D action
    Input: [base_obs(45) + encoder_latent(19)] = 64D
    Output: 12D joint position targets (NOT tanh! Direct targets)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 12)  # NO ACTIVATION! Direct joint targets
        )

    def forward(self, obs):
        """
        Args:
            obs: (batch, 64) = [base_obs(45) + encoder_out(19)]
        Returns:
            actions: (batch, 12) joint position targets
        """
        return self.net(obs)


# ============================================================================
# ENCODER NETWORK
# ============================================================================
class Encoder(nn.Module):
    """
    HIMLoco Encoder: 225D history → 19D latent
    Input: [obs_history(45 × 5)] = 225D (last 5 timesteps)
    Output: 19D latent encoding
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(225, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 19)  # 19D latent
        )

    def forward(self, history):
        """
        Args:
            history: (batch, 225) = flattened last 5 obs
        Returns:
            latent: (batch, 19) encoded representation
        """
        return self.net(history)


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================
def export_actor(checkpoint_path, output_path):
    """Export actor (policy) from main checkpoint"""
    print(f"\n{'=' * 80}")
    print("EXPORTING ACTOR (Policy Network)")
    print(f"{'=' * 80}")

    # Load checkpoint
    print(f"Loading from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Debug structure
    print("\nCheckpoint keys:", list(checkpoint.keys())[:10])

    # Extract actor state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Filter actor weights (remove critic, other components)
    actor_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('actor.'):
            # Remove 'actor.' prefix and add 'net.' prefix
            # 'actor.0.weight' → 'net.0.weight'
            new_key = key.replace('actor.', 'net.')
            actor_state_dict[new_key] = value

    print(f"\nActor layers found: {len(actor_state_dict)} parameters")
    print("Sample keys:", list(actor_state_dict.keys())[:5])

    # Create and load actor
    actor = Actor()
    actor.load_state_dict(actor_state_dict, strict=True)
    actor.eval()

    # Test forward pass
    test_input = torch.randn(1, 64)
    with torch.no_grad():
        output = actor(test_input)
    print(f"\nTest forward pass:")
    print(f"  Input shape:  {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Export as TorchScript
    scripted_actor = torch.jit.script(actor)
    scripted_actor.save(output_path)
    print(f"\n✓ Exported actor to: {output_path}")

    return actor


def export_encoder(checkpoint_path, output_path):
    """Export encoder from HIMLoco checkpoint"""
    print(f"\n{'=' * 80}")
    print("EXPORTING ENCODER (History Encoding Network)")
    print(f"{'=' * 80}")

    # Load checkpoint
    print(f"Loading from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Debug structure
    print("\nCheckpoint keys:", list(checkpoint.keys()))

    # Extract encoder_source (the active encoder at end of training)
    if 'encoder_source' in checkpoint:
        raw_state_dict = checkpoint['encoder_source']
    else:
        raise ValueError("No 'encoder_source' found in HIMLoco checkpoint!")

    # Add 'net.' prefix if needed
    encoder_state_dict = {}
    for key, value in raw_state_dict.items():
        if not key.startswith('net.'):
            new_key = f'net.{key}'
        else:
            new_key = key
        encoder_state_dict[new_key] = value

    print(f"\nEncoder layers found: {len(encoder_state_dict)} parameters")
    print("Sample keys:", list(encoder_state_dict.keys())[:5])

    # Create and load encoder
    encoder = Encoder()
    encoder.load_state_dict(encoder_state_dict, strict=True)
    encoder.eval()

    # Test forward pass
    test_input = torch.randn(1, 225)  # 45 obs × 5 history
    with torch.no_grad():
        output = encoder(test_input)
    print(f"\nTest forward pass:")
    print(f"  Input shape:  {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Export as TorchScript
    scripted_encoder = torch.jit.script(encoder)
    scripted_encoder.save(output_path)
    print(f"\n✓ Exported encoder to: {output_path}")

    return encoder


# ============================================================================
# MAIN EXPORT
# ============================================================================
if __name__ == "__main__":
    import os

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("HIMLoco Go1 Deployment Export")
    print("=" * 80)
    print(f"\nPolicy checkpoint: {policy_checkpoint}")
    print(f"HIMLoco checkpoint: {himloco_checkpoint}")
    print(f"Output directory: {output_dir}")

    # Export both networks
    try:
        actor = export_actor(policy_checkpoint, output_actor)
        encoder = export_encoder(himloco_checkpoint, output_encoder)

        print("\n" + "=" * 80)
        print("✓ EXPORT SUCCESSFUL!")
        print("=" * 80)
        print(f"\nDeployment files ready:")
        print(f"  Actor:   {output_actor}")
        print(f"  Encoder: {output_encoder}")
        print(f"\nCopy these files to Go1's Raspberry Pi for deployment.")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n✗ EXPORT FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)