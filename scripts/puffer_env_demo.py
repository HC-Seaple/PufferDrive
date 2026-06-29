import os
import sys
from pathlib import Path
import shutil
import subprocess
import time
import tempfile
import platform
from enum import IntEnum

# ==========================================
# FIX 1: Set working directory to project root
# This fixes the "Error while loading pufferlib/config/ocean/drive.ini"
# ==========================================
working_dir = Path.cwd()
# Traverse up until we find the root directory containing 'pufferlib'
while not (working_dir / 'pufferlib').exists():
    if working_dir == working_dir.parent:
        raise FileNotFoundError("Could not find the PufferDrive project root containing 'pufferlib'")
    working_dir = working_dir.parent
os.chdir(working_dir)
# Add the project root first so this script can be run directly from scripts/.
sys.path.insert(0, str(working_dir))
os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")

import numpy as np

if platform.system() == "Windows":
    Drive = None
    NATIVE_DRIVE_IMPORT_ERROR = "native Ocean binding is not built for Windows in this repository"
else:
    try:
        from pufferlib.ocean.drive.drive import Drive, RenderView

        NATIVE_DRIVE_IMPORT_ERROR = None
    except ImportError as exc:
        Drive = None
        NATIVE_DRIVE_IMPORT_ERROR = exc

if Drive is None:
    class RenderView(IntEnum):
        FULL_SIM_STATE = 0
        BEV_AGENT_OBS = 1
        AGENT_PERSP = 2


RENDER_WINDOW = 0
RENDER_HEADLESS = 1


class FallbackDrive:
    """Small Windows-friendly demo path when the native Ocean binding is unavailable."""

    def __init__(
        self,
        map_dir,
        num_maps,
        num_agents,
        action_type,
        episode_length,
        render_mode,
        **kwargs,
    ):
        import gymnasium

        map_path = Path(map_dir) / "map_000.bin"
        if not map_path.exists():
            raise FileNotFoundError(f"Fallback map not found: {map_path}")

        self.map_dir = map_dir
        self.num_agents = num_agents
        self.num_envs = num_maps
        self.map_ids = [0]
        self.scenario_ids = ["fallback_map_000"]
        self.episode_length = episode_length
        self.render_mode = render_mode
        self.tick = 0
        self._frames = []

        self.observation_space = gymnasium.spaces.Box(
            low=-1,
            high=1,
            shape=(num_agents, 32),
            dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Box(
            low=-1,
            high=1,
            shape=(num_agents, 2),
            dtype=np.float32,
        )
        self.observations = np.zeros(self.observation_space.shape, dtype=np.float32)

    def reset(self):
        self.tick = 0
        self.observations.fill(0)
        return self.observations, []

    def step(self, actions):
        self.tick += 1
        self.observations[:, 0] = min(1.0, self.tick / max(1, self.episode_length))
        self.observations[:, 1:3] = np.clip(actions, -1, 1)
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        terminals = np.zeros(self.num_agents, dtype=bool)
        truncations = np.full(self.num_agents, self.tick >= self.episode_length, dtype=bool)
        return self.observations, rewards, terminals, truncations, [{} for _ in range(self.num_agents)]

    def render(self, view_mode, draw_traces, env_id):
        width, height = 640, 368
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = 34
        frame[:, :, 1] = 46
        frame[:, :, 2] = 58

        frame[height // 2 - 26 : height // 2 + 26, :, :] = (54, 64, 72)
        frame[height // 2 - 2 : height // 2 + 2, 40 : width - 40, :] = (230, 210, 94)
        frame[height // 2 - 58 : height // 2 - 54, :, :] = (220, 220, 220)
        frame[height // 2 + 54 : height // 2 + 58, :, :] = (220, 220, 220)

        progress = self.tick / max(1, self.episode_length)
        car_x = int(60 + progress * (width - 120))
        car_y = height // 2
        frame[car_y - 14 : car_y + 14, car_x - 28 : car_x + 28, :] = (65, 146, 229)
        frame[car_y - 9 : car_y - 3, car_x - 14 : car_x + 14, :] = (190, 228, 255)
        self._frames.append(frame)

    def close(self):
        if not self._frames:
            return

        import imageio.v2 as imageio

        output = Path(f"{self.scenario_ids[0]}.mp4")
        imageio.mimsave(output, self._frames, fps=20)


def start_xvfb_if_needed():
    """Start a virtual display before raylib/GLFW initializes."""
    if os.environ.get("DISPLAY"):
        return None

    x11_socket_dir = Path("/tmp/.X11-unix")
    if x11_socket_dir.exists() and x11_socket_dir.stat().st_uid != 0:
        raise RuntimeError(
            f"Cannot start Xvfb because {x11_socket_dir} is not owned by root. "
            "Fix that directory or run this script inside a working Xvfb/display session."
        )

    errors = []
    for display_num in range(99, 120):
        display = f":{display_num}"
        lock_path = Path(f"/tmp/.X{display_num}-lock")
        socket_path = Path(f"/tmp/.X11-unix/X{display_num}")
        if lock_path.exists() or socket_path.exists():
            continue

        error_log = tempfile.NamedTemporaryFile(prefix="pufferdrive_xvfb_", suffix=".log", delete=False)
        error_log_path = Path(error_log.name)
        proc = subprocess.Popen(
            [
                "Xvfb",
                display,
                "-screen",
                "0",
                "1280x720x24",
                "+extension",
                "GLX",
                "-ac",
                "-noreset",
            ],
            stdout=subprocess.DEVNULL,
            stderr=error_log,
        )
        error_log.close()
        os.environ["DISPLAY"] = display

        for _ in range(40):
            if proc.poll() is not None:
                break
            if lock_path.exists() or socket_path.exists():
                time.sleep(0.5)
                error_log_path.unlink(missing_ok=True)
                return proc
            time.sleep(0.1)

        proc.terminate()
        proc.wait(timeout=2)
        os.environ.pop("DISPLAY", None)
        errors.append(error_log_path.read_text(errors="replace").strip())
        error_log_path.unlink(missing_ok=True)

    detail = "\n".join(error for error in errors if error)
    raise RuntimeError(f"Could not start Xvfb for headless rendering:\n{detail}")


def stop_xvfb(proc):
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)
    os.environ.pop("DISPLAY", None)


def unique_output_path(path):
    """Avoid overwriting previous demo videos."""
    path = Path(path)
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    for i in range(1, 1000):
        candidate = path.with_name(f"{stem}_{i:03d}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an unused output filename for {path}")


def main():
    MAP_DIR = "pufferlib/resources/drive/binaries"
    
    if not Path(MAP_DIR).exists():
        print(f"Warning: {MAP_DIR} not found. Please ensure you are running from the correct root and maps are processed.")
        return

    NUM_MAPS = 1           
    NUM_AGENTS = 1       
    EPISODE_LEN = 91       
    RENDER_ENV_ID = 0
    RENDER_AGENT_IDX = 0
    RENDER_VIEW = RenderView.AGENT_PERSP
    DRAW_TRACES = False
    OUTPUT_PATH = Path("pufferdrive_headless_demo.mp4")

    print("Initializing PufferDrive Environment...")
    drive_cls = Drive
    if drive_cls is None:
        print("Native PufferDrive binding is unavailable; using Windows fallback demo.")
        print(f"Import error: {NATIVE_DRIVE_IMPORT_ERROR}")
        drive_cls = FallbackDrive

    env = None
    xvfb_proc = start_xvfb_if_needed() if Drive is not None else None
    try:
        env = drive_cls(
            map_dir=MAP_DIR,
            num_maps=NUM_MAPS,
            num_agents=NUM_AGENTS,
            control_mode="control_sdc_only", 
            # control_mode="control_vehicles",
            init_mode="create_all_valid",    
            goal_behavior=2, # 
            action_type="continuous",        
            episode_length=EPISODE_LEN,
            render_mode=RENDER_HEADLESS,
            human_agent_idx=RENDER_AGENT_IDX,
            max_controlled_agents=1,
        )

        obs, _ = env.reset()
        print("env obs space:", env.observation_space)
        print("env action space:", env.action_space)
        num_controlled_agents = obs.shape[0]
        print(f"Environment initialized with {num_controlled_agents} agents and the full obs is {obs.shape}")
        print(f"Vectorized env count: {env.num_envs}")
        print(f"Rendering env_id={RENDER_ENV_ID}, map_id={env.map_ids[RENDER_ENV_ID]}")
        print(f"Rendering local controlled agent index: {RENDER_AGENT_IDX}")
        print(f"Render view: {RENDER_VIEW.name}")
        print(f"Scenario id: {env.scenario_ids[RENDER_ENV_ID]}")

        scenario_video = Path(f"{env.scenario_ids[RENDER_ENV_ID]}.mp4")
        output_path = unique_output_path(OUTPUT_PATH)

        print("Starting headless render rollout...")

        # Render the reset state too, so the video includes the initial frame.
        env.render(view_mode=RENDER_VIEW, draw_traces=DRAW_TRACES, env_id=RENDER_ENV_ID)

        for t in range(EPISODE_LEN):
            rl_actions = np.random.uniform(-0.1, 0.1, size=(num_controlled_agents, 2)).astype(np.float32)
            obs, rewards, terminals, truncations, infos = env.step(rl_actions)

            # In headless mode the native renderer pipes this frame to ffmpeg.
            env.render(view_mode=RENDER_VIEW, draw_traces=DRAW_TRACES, env_id=RENDER_ENV_ID)
            
            if t % 10 == 0:
                print(f"Rendered step {t:02d}/{EPISODE_LEN}...")

            if terminals.all() or truncations.all():
                print(f"Environment terminated at step {t}.")
                break
    finally:
        # Closing the env closes the ffmpeg pipe and finalizes the mp4.
        if env is not None:
            env.close()
        stop_xvfb(xvfb_proc)

    if scenario_video.exists():
        shutil.move(str(scenario_video), str(output_path))
        print(f"Headless render saved to {output_path}")
    else:
        print(
            f"Expected native render output {scenario_video} was not found. "
            "Check that ffmpeg and Xvfb are installed and available on PATH."
        )

if __name__ == "__main__":
    main()
