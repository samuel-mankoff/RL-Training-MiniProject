

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
        # full_command = torch.cat([target_cmd_twist, torch.tensor([0.0]).to(device)])
        env.unwrapped.command_manager.get_command("twist")[:] = target_cmd_twist

        # --- B. Inference ---
        with torch.no_grad():
            actions = policy(obs)
        
        # --- C. Step ---
        obs, _, _, _ = env.step(actions)

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
