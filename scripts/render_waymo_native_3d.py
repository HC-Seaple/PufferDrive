import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
import sys

sys.path.insert(0, str(REPO_ROOT))

resources_path = REPO_ROOT / "resources"
if not resources_path.is_dir():
    resources_path.unlink(missing_ok=True)
    resources_path.symlink_to(REPO_ROOT / "pufferlib" / "resources", target_is_directory=True)

import numpy as np

from pufferlib.ocean.drive.drive import Drive, RenderView, load_map


def start_xvfb_if_needed():
    if os.environ.get("DISPLAY"):
        return None

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
        detail = error_log_path.read_text(errors="replace").strip()
        error_log_path.unlink(missing_ok=True)
        if detail:
            print(detail)

    raise RuntimeError("Could not start Xvfb for native Raylib rendering")


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


def prepare_json(source_path, prepared_path):
    with source_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    sdc_track_index = data.get("metadata", {}).get("sdc_track_index")
    for index, obj in enumerate(data.get("objects", [])):
        # Let background actors follow the WOMD replay when the native simulator
        # supports expert stepping. Keep the SDC policy-controlled so AGENT_PERSP
        # can follow it as the selected active agent.
        obj["mark_as_expert"] = index != sdc_track_index

    with prepared_path.open("w", encoding="utf-8") as f:
        json.dump(data, f)

    return str(data.get("scenario_id") or source_path.stem)


def render_map(map_dir, output_dir, scenario_id, episode_length, fps_hint=False):
    env = None
    xvfb_proc = start_xvfb_if_needed()
    try:
        env = Drive(
            map_dir=str(map_dir),
            num_maps=1,
            num_agents=1,
            control_mode="control_sdc_only",
            init_mode="create_all_valid",
            goal_behavior=2,
            action_type="continuous",
            episode_length=episode_length,
            resample_frequency=0,
            render_mode=1,
            human_agent_idx=0,
            max_controlled_agents=1,
        )
        obs, _ = env.reset()
        env.render(view_mode=RenderView.AGENT_PERSP, draw_traces=True, env_id=0)

        for step in range(episode_length):
            actions = np.zeros((obs.shape[0], 2), dtype=np.float32)
            obs, rewards, terminals, truncations, infos = env.step(actions)
            env.render(view_mode=RenderView.AGENT_PERSP, draw_traces=True, env_id=0)
            if step % 10 == 0:
                print(f"  rendered step {step:02d}/{episode_length}")
            if terminals.all() or truncations.all():
                break
    finally:
        if env is not None:
            env.close()
        stop_xvfb(xvfb_proc)

    native_output = Path(f"{scenario_id}.mp4")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{scenario_id}_agent_persp.mp4"
    if native_output.exists():
        shutil.move(str(native_output), target)
        print(f"  saved {target}")
    else:
        raise FileNotFoundError(f"Native renderer did not produce {native_output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_files", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("native_3d_renders"))
    parser.add_argument("--episode-length", type=int, default=91)
    args = parser.parse_args()

    work_dir = Path("resources/drive/binaries/waymo_native")
    source_dir = Path("resources/drive/waymo_native_json")
    work_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    for map_index, json_path in enumerate(args.json_files):
        prepared_json = source_dir / f"map_{map_index:03d}.json"
        scenario_id = prepare_json(json_path, prepared_json)
        binary_path = work_dir / "map_000.bin"
        if binary_path.exists():
            binary_path.unlink()

        print(f"Converting {json_path} -> {binary_path}")
        load_map(str(prepared_json), unique_map_id=0, binary_output=str(binary_path))

        print(f"Rendering {scenario_id} with RenderView.AGENT_PERSP")
        render_map(work_dir, args.output_dir, scenario_id, args.episode_length)


if __name__ == "__main__":
    main()
