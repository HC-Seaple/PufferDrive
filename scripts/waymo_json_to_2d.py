import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from visualize_waymo_json import (
    ROAD_STYLES,
    bounds_for,
    convert_raw_waymo,
    draw_object,
    draw_polyline,
    make_transform,
    point_at,
    valid_at,
)


def downloads_folder():
    return Path.home() / "Downloads" / "Waymo_2D"


def draw_legend(draw, width):
    items = [
        ((25, 103, 210), "SDC / ego"),
        ((191, 63, 47), "track to predict"),
        ((44, 139, 87), "other vehicle"),
        ((141, 90, 194), "pedestrian / cyclist"),
    ]
    box_width = 460
    left = width - box_width - 18
    draw.rounded_rectangle((left, 18, width - 18, 54), radius=5, fill=(250, 250, 248, 230))
    x = left + 12
    for color, label in items:
        draw.rectangle((x, 30, x + 12, 42), fill=color)
        draw.text((x + 17, 29), label, fill=(35, 40, 44), font=ImageFont.load_default())
        x += 105


def render_static(path, output_dir, width, height, trajectory_threshold):
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    data = convert_raw_waymo(raw)
    frame = int(raw.get("current_time_index", 0))
    max_frame = max(len(obj.get("position", [])) for obj in data.get("objects", [])) - 1
    frame = max(0, min(frame, max_frame))
    transform, scale = make_transform(bounds_for(data), width, height, pad=55)

    image = Image.new("RGB", (width, height), (235, 231, 223))
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    for road in data.get("roads", []):
        color, line_width = ROAD_STYLES.get(road.get("type"), ((110, 110, 110), 1))
        alpha = 125 if road.get("type") == "lane" else 195
        geometry = road.get("geometry", [])
        draw_polyline(draw, geometry, transform, color + (alpha,), line_width)
        if road.get("type") in {"crosswalk", "speed_bump", "driveway"} and len(geometry) >= 3:
            polygon = [transform(point["x"], point["y"]) for point in geometry]
            draw.polygon(polygon, outline=color + (150,))

    moving_count = 0
    for index, obj in enumerate(data.get("objects", [])):
        trajectory = [
            point_at(obj, i)
            for i in range(len(obj.get("position", [])))
            if point_at(obj, i) and valid_at(obj, i)
        ]
        if len(trajectory) >= 2:
            dx = trajectory[-1]["x"] - trajectory[0]["x"]
            dy = trajectory[-1]["y"] - trajectory[0]["y"]
            if (dx * dx + dy * dy) ** 0.5 >= trajectory_threshold:
                moving_count += 1
                metadata = data.get("metadata", {})
                predicted = {item.get("track_index") for item in metadata.get("tracks_to_predict", [])}
                if index == metadata.get("sdc_track_index"):
                    color = (25, 103, 210)
                elif index in predicted:
                    color = (191, 63, 47)
                elif obj.get("type") in {"pedestrian", "cyclist"}:
                    color = (141, 90, 194)
                else:
                    color = (44, 139, 87)
                draw_polyline(draw, trajectory, transform, color + (150,), 2)

        draw_object(draw, data, obj, index, frame, transform, scale, trails=False)

    scenario_id = str(data.get("scenario_id") or path.stem)
    draw.rounded_rectangle((18, 18, 520, 72), radius=5, fill=(250, 250, 248, 230))
    draw.text((30, 29), f"Scenario: {scenario_id}", fill=(28, 35, 40), font=ImageFont.load_default())
    draw.text(
        (30, 50),
        f"frame {frame}/{max_frame} | objects {len(data.get('objects', []))} | moving tracks {moving_count}",
        fill=(65, 72, 78),
        font=ImageFont.load_default(),
    )
    draw_legend(draw, width)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{scenario_id}_2d.png"
    Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB").save(output_path)
    return output_path, scenario_id, moving_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert Waymo or PufferDrive scenario JSON files to top-down PNG images."
    )
    parser.add_argument("json_files", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=downloads_folder())
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--trajectory-threshold", type=float, default=1.0)
    parser.add_argument("--open", action="store_true", dest="open_outputs")
    args = parser.parse_args()

    outputs = []
    for path in args.json_files:
        if not path.is_file():
            print(f"Skipped missing file: {path}")
            continue
        output, scenario_id, moving_count = render_static(
            path,
            args.output_dir,
            args.width,
            args.height,
            args.trajectory_threshold,
        )
        outputs.append(output)
        print(f"Rendered {path.name}")
        print(f"  scenario: {scenario_id}")
        print(f"  moving tracks: {moving_count}")
        print(f"  image: {output}")

    if args.open_outputs and os.name == "nt":
        for output in outputs:
            os.startfile(output)


if __name__ == "__main__":
    main()
