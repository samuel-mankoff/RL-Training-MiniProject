# # src/mjlab/scripts/command_tracking.py

# import argparse
# from dataclasses import asdict
# from pathlib import Path

# import numpy as np
# import torch
# from rsl_rl.runners import OnPolicyRunner

# from mjlab.envs import ManagerBasedRlEnv
# from mjlab.rl import RslRlVecEnvWrapper
# from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
# from mjlab.utils.torch import configure_torch_backends

# TASK_NAME = "Mjlab-Velocity-Flat-Unitree-Go1"


# def build_command_sequence(num_steps_per_phase: int = 125):
#     """Return (T, 3) array of commanded [vx, vy, wz] for the 4 phases."""
#     cmds = []

#     # 1) Forward walking: (vx, vy, wz) = (0 -> 0.6, 0, 0)
#     vx_ramp = np.linspace(0.0, 0.6, num_steps_per_phase)
#     for vx in vx_ramp:
#         cmds.append([vx, 0.0, 0.0])

#     # 2) Lateral walking: (0, 0.4, 0)
#     for _ in range(num_steps_per_phase):
#         cmds.append([0.0, 0.4, 0.0])

#     # 3) Turning: (0, 0, 0.4)
#     for _ in range(num_steps_per_phase):
#         cmds.append([0.0, 0.0, 0.4])

#     # 4) Mixed: (0.5, 0, 0.3)
#     for _ in range(num_steps_per_phase):
#         cmds.append([0.5, 0.0, 0.3])

#     return np.asarray(cmds, dtype=np.float32)  # (T, 3)


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--checkpoint",
#         type=str,
#         required=True,
#         help="Path to trained checkpoint, e.g. logs/rsl_rl/go1_velocity/.../model_300.pt",
#     )
#     parser.add_argument(
#         "--out-npz",
#         type=str,
#         default="command_tracking.npz",
#         help="Where to save commanded vs measured velocities.",
#     )
#     args = parser.parse_args()

#     ckpt_path = Path(args.checkpoint)
#     if not ckpt_path.exists():
#         raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

#     configure_torch_backends()
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     print(f"[INFO] Using device: {device}")

#     # 1) Load env + RL configs from YOUR fork
#     env_cfg = load_env_cfg(TASK_NAME)
#     agent_cfg = load_rl_cfg(TASK_NAME)

#     # Single environment for evaluation
#     env_cfg.scene.num_envs = 1

#     # (Optional but nice): disable corruption & random pushes for clean eval
#     if "policy" in env_cfg.observations:
#         env_cfg.observations["policy"].enable_corruption = False
#     if env_cfg.events is not None:
#         env_cfg.events.pop("push_robot", None)

#     # 2) Build environment (no viewer)
#     base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
#     env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

#     # 3) Build runner + load policy
#     runner = OnPolicyRunner(
#         env, asdict(agent_cfg), log_dir=str(ckpt_path.parent), device=device
#     )
#     runner.load(str(ckpt_path), map_location=device)
#     policy = runner.get_inference_policy(device=device)

#     # 4) Build command sequence
#     cmd_seq = build_command_sequence(num_steps_per_phase=125)  # (T, 3)
#     T = cmd_seq.shape[0]

#     # Storage for measured velocities
#     measured_lin = np.zeros((T, 2), dtype=np.float32)  # [vx, vy]
#     measured_ang = np.zeros((T,), dtype=np.float32)    # [wz]

#     # 5) Rollout
#     obs, _ = env.reset()

#     # --- IMPORTANT PART: how we inject commands ---
#     # We assume the environment uses a "twist" command with components
#     # [vx, vy, wz] and that there is some way to set it on the base_env.
#     # You will likely need to tweak this based on the actual CommandManager API.
#     cmd_mgr = base_env.command_manager  # this attribute exists in training logs
#     cmd_name = "twist"

#     for t in range(T):
#         vx_cmd, vy_cmd, wz_cmd = cmd_seq[t]

#         # TODO: replace this line with the actual way to set the command
#         # The idea is to write the 3 numbers into the twist command state.
#         # Example (you will have to inspect cmd_mgr to find the real method):
#         #
#         #   cmd_mgr.set_command(cmd_name, np.array([vx_cmd, vy_cmd, wz_cmd], dtype=np.float32))
#         #
#         # For now, this is just a placeholder:
#         if hasattr(cmd_mgr, "set_command"):
#             cmd_mgr.set_command(cmd_name, np.array([vx_cmd, vy_cmd, wz_cmd], dtype=np.float32))
#         else:
#             raise RuntimeError(
#                 "CommandManager API unknown. Inspect `dir(cmd_mgr)` in a small test "
#                 "script to see how to write a command for 'twist'."
#             )

#         # Get action from policy and step
#         with torch.no_grad():
#             obs_tensor = torch.from_numpy(obs).to(device)
#             act = policy(obs_tensor).cpu().numpy()

#         obs, rew, terminated, truncated, info = env.step(act)

#         # Extract measured velocities from env observations / info.
#         # During training, you already logged `Metrics/twist/error_vel_*`,
#         # which are based on base lin/ang velocities. Here we directly grab them.
#         # This assumes base_env has sensors "robot/imu_lin_vel" and "robot/imu_ang_vel".
#         sim_state = base_env.state  # again, you might need to adapt this

#         lin_vel = sim_state["robot/imu_lin_vel"][0]  # shape (3,)
#         ang_vel = sim_state["robot/imu_ang_vel"][0]  # shape (3,)

#         measured_lin[t, 0] = lin_vel[0]  # vx
#         measured_lin[t, 1] = lin_vel[1]  # vy
#         measured_ang[t] = ang_vel[2]     # wz

#     env.close()

#     # 6) Save results
#     np.savez(
#         args.out_npz,
#         cmd_seq=cmd_seq,
#         measured_lin=measured_lin,
#         measured_ang=measured_ang,
#     )
#     print(f"[INFO] Saved command-tracking data to {args.out_npz}")


# # if __name__ == "__main__":
# #     main()
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import tyro
# from dataclasses import asdict

# # Import the pre-made configuration function from your file
# from mjlab.tasks.velocity.config.go1.env_cfgs import UNITREE_GO1_FLAT_ENV_CFG
# from mjlab.tasks.velocity.config.go1.rl_cfg import UNITREE_GO1_PPO_RUNNER_CFG
# from mjlab.envs import ManagerBasedRlEnv
# from rsl_rl.runners import OnPolicyRunner

# def get_command_for_step(step_idx):
#     """Returns (vx, vy, wz) based on the assignment timeline."""
#     vx, vy, wz = 0.0, 0.0, 0.0
    
#     # Sequence length is 125 steps per phase
#     phase = step_idx // 125
#     local_step = step_idx % 125
    
#     if phase == 0: 
#         # Phase 1: Forward walking 0 -> 0.6
#         vx = 0.6 * (local_step / 125.0)
#     elif phase == 1:
#         # Phase 2: Lateral walking vy = 0.4
#         vy = 0.4
#     elif phase == 2:
#         # Phase 3: Turning wz = 0.4
#         wz = 0.4
#     elif phase == 3:
#         # Phase 4: Mixed command
#         vx = 0.5
#         wz = 0.3
        
#     return torch.tensor([vx, vy, wz], dtype=torch.float)

# def run_tracking_eval(log_dir: str, num_steps: int = 500):

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Running evaluation on device: {device}")

#     # 1. Load the Configuration
#     # We simply call the function from env_cfgs.py
#     env_cfg = UNITREE_GO1_FLAT_ENV_CFG
    
#     # Force 1 environment for testing
#     env_cfg.scene.num_envs = 1
#     # Disable randomization for consistent testing graph
#     env_cfg.events = {} 
    
#     # 2. Setup Environment
#     env = ManagerBasedRlEnv(cfg=env_cfg, device = device )

#     # 3. Load the RL Configuration (Fix for KeyError: 'algorithm')
#     # We convert the config object into a dictionary
#     train_cfg = asdict(UNITREE_GO1_PPO_RUNNER_CFG)

#     # 4. Load the Trained Policy
#     runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=device)
#     runner.load(resume=True) 
#     policy = runner.get_inference_policy(device=device)

#     # # 3. Load the Trained Policy
#     # # The runner looks into log_dir and finds the latest model_XXX.pt automatically
#     # runner = OnPolicyRunner(env, {"train_dir": log_dir}, log_dir=log_dir, device="cuda")
#     # runner.load(resume=True) 
#     # policy = runner.get_inference_policy(device="cuda")

#     # 4. Storage for plotting
#     logs = {
#         "cmd_vx": [], "cmd_vy": [], "cmd_wz": [],
#         "meas_vx": [], "meas_vy": [], "meas_wz": []
#     }

#     obs, _ = env.reset()
    
#     print(f"Starting tracking evaluation for {num_steps} steps...")

#     for i in range(num_steps):
#         # --- A. Overwrite Command ---
#         target_cmd_twist = get_command_for_step(i).to("cuda")
        
#         # Inject command into the environment manager
#         # We append 0.0 for the heading command if your config expects 4 dimensions
#         full_command = torch.cat([target_cmd_twist, torch.tensor([0.0]).to(device)])
#         env.command_manager.get_command("twist")[:] = full_command
#         # env.command_manager.get_command("twist")[:] = torch.cat([target_cmd_twist, torch.tensor([0.0]).cuda()])

#         # --- B. Inference ---
#         with torch.no_grad():
#             actions = policy(obs)
        
#         # --- C. Step ---
#         obs, _, _, _, _ = env.step(actions)

#         # --- D. Log Data ---
#         # Actual Base Velocity (in base frame)
#         base_vel = env.scene["robot"].data.root_link_lin_vel_b[0] # (env, 3)
#         ang_vel = env.scene["robot"].data.root_link_ang_vel_b[0]  # (env, 3)
        
#         logs["cmd_vx"].append(target_cmd_twist[0].item())
#         logs["cmd_vy"].append(target_cmd_twist[1].item())
#         logs["cmd_wz"].append(target_cmd_twist[2].item())
        
#         logs["meas_vx"].append(base_vel[0].item())
#         logs["meas_vy"].append(base_vel[1].item())
#         logs["meas_wz"].append(ang_vel[2].item())

#     # 5. Plotting
#     fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
#     time = np.arange(num_steps) * env.step_dt

#     # Plot Vx
#     axs[0].plot(time, logs["cmd_vx"], 'r--', label="Command")
#     axs[0].plot(time, logs["meas_vx"], 'b-', label="Measured")
#     axs[0].set_ylabel("Forward Vel (m/s)")
#     axs[0].set_title("Forward Velocity")
#     axs[0].legend()
#     axs[0].grid(True)

#     # Plot Vy
#     axs[1].plot(time, logs["cmd_vy"], 'r--', label="Command")
#     axs[1].plot(time, logs["meas_vy"], 'b-', label="Measured")
#     axs[1].set_ylabel("Lateral Vel (m/s)")
#     axs[1].set_title("Lateral Velocity")
#     axs[1].grid(True)

#     # Plot Wz
#     axs[2].plot(time, logs["cmd_wz"], 'r--', label="Command")
#     axs[2].plot(time, logs["meas_wz"], 'b-', label="Measured")
#     axs[2].set_ylabel("Yaw Vel (rad/s)")
#     axs[2].set_title("Yaw Velocity")
#     axs[2].set_xlabel("Time (s)")
#     axs[2].grid(True)

#     # Save to the specific log folder
#     save_path = os.path.join(log_dir, "tracking_performance.png")
#     plt.tight_layout()
#     plt.savefig(save_path)
#     print(f"Plot saved to {save_path}")

# if __name__ == "__main__":
#     tyro.cli(run_tracking_eval)

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import tyro
import glob
from dataclasses import asdict

# 1. Imports for Configs
from mjlab.tasks.velocity.config.go1.env_cfgs import UNITREE_GO1_FLAT_ENV_CFG
from mjlab.tasks.velocity.config.go1.rl_cfg import UNITREE_GO1_PPO_RUNNER_CFG

# 2. Imports for Environment and Runner
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper  # <--- THIS IS THE MISSING KEY
from rsl_rl.runners import OnPolicyRunner

def get_command_for_step(step_idx):
    """Returns (vx, vy, wz) based on the assignment timeline."""
    vx, vy, wz = 0.0, 0.0, 0.0
    
    # Sequence length is 125 steps per phase
    phase = step_idx // 125
    local_step = step_idx % 125
    
    if phase == 0: 
        # Phase 1: Forward walking 0 -> 0.6
        vx = 0.6 * (local_step / 125.0)
    elif phase == 1:
        # Phase 2: Lateral walking vy = 0.4
        vy = 0.4
    elif phase == 2:
        # Phase 3: Turning wz = 0.4
        wz = 0.4
    elif phase == 3:
        # Phase 4: Mixed command
        vx = 0.5
        wz = 0.3
        
    return torch.tensor([vx, vy, wz], dtype=torch.float)

def run_tracking_eval(log_dir: str, num_steps: int = 500):
    # 0. Detect Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running evaluation on device: {device}")

    # 1. Load the Configuration
    env_cfg = UNITREE_GO1_FLAT_ENV_CFG
    env_cfg.scene.num_envs = 1
    env_cfg.events = {} 
    
    # 2. Setup Environment
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    
    # --- FIX START: Add the Wrapper ---
    # This translates MJLab env into something RSL-RL can read
    env = RslRlVecEnvWrapper(env, clip_actions=True)
    # --- FIX END ---

    # 3. Load the RL Configuration
    train_cfg = asdict(UNITREE_GO1_PPO_RUNNER_CFG)

    # 4. Load the Trained Policy
    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=device)
    # runner.load(resume=True) 
    # policy = runner.get_inference_policy(device=device)

    #  FIX: Manually find the latest checkpoint 
    # Search for all model_*.pt files in the directory
    model_files = glob.glob(os.path.join(log_dir, "model_*.pt"))
    
    if not model_files:
        raise FileNotFoundError(f"No model checkpoints found in {log_dir}")
        
    # Sort to find the highest number (e.g., model_300.pt)
    # This splits the filename to extract the number: "model_300.pt" -> 300
    latest_model_path = max(model_files, key=lambda p: int(p.split("model_")[-1].split(".pt")[0]))
    
    print(f"Loading latest checkpoint: {latest_model_path}")
    
    # Load the specific file (resume=True removed)
    runner.load(latest_model_path) 
    
    policy = runner.get_inference_policy(device=device)

    # 5. Storage for plotting
    logs = {
        "cmd_vx": [], "cmd_vy": [], "cmd_wz": [],
        "meas_vx": [], "meas_vy": [], "meas_wz": []
    }

    obs, _ = env.reset()
    
    print(f"Starting tracking evaluation for {num_steps} steps...")

    for i in range(num_steps):
        # --- A. Overwrite Command ---
        target_cmd_twist = get_command_for_step(i).to(device)
        
        # Inject command into the environment manager
        # Note: We access env.unwrapped to bypass the wrapper we just added
        full_command = torch.cat([target_cmd_twist, torch.tensor([0.0]).to(device)])
        env.unwrapped.command_manager.get_command("twist")[:] = full_command

        # --- B. Inference ---
        with torch.no_grad():
            actions = policy(obs)
        
        # --- C. Step ---
        obs, _, _, _, _ = env.step(actions)

        # --- D. Log Data ---
        # Access physical data from the unwrapped environment
        base_vel = env.unwrapped.scene["robot"].data.root_link_lin_vel_b[0] 
        ang_vel = env.unwrapped.scene["robot"].data.root_link_ang_vel_b[0]
        
        logs["cmd_vx"].append(target_cmd_twist[0].item())
        logs["cmd_vy"].append(target_cmd_twist[1].item())
        logs["cmd_wz"].append(target_cmd_twist[2].item())
        
        logs["meas_vx"].append(base_vel[0].item())
        logs["meas_vy"].append(base_vel[1].item())
        logs["meas_wz"].append(ang_vel[2].item())

    # 6. Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    time = np.arange(num_steps) * env.unwrapped.step_dt

    # Plot Vx
    axs[0].plot(time, logs["cmd_vx"], 'r--', label="Command")
    axs[0].plot(time, logs["meas_vx"], 'b-', label="Measured")
    axs[0].set_ylabel("Forward Vel (m/s)")
    axs[0].set_title("Forward Velocity")
    axs[0].legend()
    axs[0].grid(True)

    # Plot Vy
    axs[1].plot(time, logs["cmd_vy"], 'r--', label="Command")
    axs[1].plot(time, logs["meas_vy"], 'b-', label="Measured")
    axs[1].set_ylabel("Lateral Vel (m/s)")
    axs[1].set_title("Lateral Velocity")
    axs[1].grid(True)

    # Plot Wz
    axs[2].plot(time, logs["cmd_wz"], 'r--', label="Command")
    axs[2].plot(time, logs["meas_wz"], 'b-', label="Measured")
    axs[2].set_ylabel("Yaw Vel (rad/s)")
    axs[2].set_title("Yaw Velocity")
    axs[2].set_xlabel("Time (s)")
    axs[2].grid(True)

    save_path = os.path.join(log_dir, "tracking_performance.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    tyro.cli(run_tracking_eval)