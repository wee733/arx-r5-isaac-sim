#!/usr/bin/env python3
"""Validate Pico YAML and emit a ROS-compatible typed copy, never edit input."""
import math
import sys
import yaml

FLOATS = {
    "position_scale", "rotation_scale", "timeout_s", "linear_speed_m_s",
    "angular_speed_rad_s", "max_clutch_translation_m", "max_clutch_rotation_rad",
    "max_joint_speed_rad_s",
}


def normalize(data):
    if not isinstance(data, dict) or not data:
        raise ValueError("配置必须是非空 YAML 映射")
    for node, block in data.items():
        if not isinstance(block, dict) or not isinstance(block.get("ros__parameters"), dict):
            raise ValueError(f"{node}: 缺少 ros__parameters 映射")
        params = block["ros__parameters"]
        for key in FLOATS & params.keys():
            value = params[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{node}.{key}: 必须是大于 0 的有限数字，收到 {value!r}")
            params[key] = float(value)
        for key in {"orientation_enabled", "gripper_enabled"} & params.keys():
            if type(params[key]) is not bool:
                raise ValueError(f"{node}.{key}: 必须填写 true 或 false")
        if params.get("translation_mode", "pose") not in ("pose", "joystick"):
            raise ValueError("translation_mode 必须是 pose 或 joystick")
        if "joystick_deadzone" in params:
            v=params["joystick_deadzone"]
            if type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<1:
                raise ValueError("joystick_deadzone 必须满足 0 <= 值 < 1")
            params["joystick_deadzone"]=float(v)
        if "rotation_axis_map" in params:
            axes=params["rotation_axis_map"]
            if not isinstance(axes,list) or len(axes)!=3 or any(type(v) is not int for v in axes) or sorted(abs(v) for v in axes)!=[1,2,3]:
                raise ValueError("rotation_axis_map 必须由带正负号的 1、2、3 各一个组成")
    return data


if __name__ == "__main__":
    try:
        with open(sys.argv[1], encoding="utf-8") as stream:
            data = normalize(yaml.safe_load(stream))
        print(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), end="")
    except (ValueError, OSError, yaml.YAMLError, IndexError) as error:
        print(f"Pico 配置错误：{error}", file=sys.stderr)
        sys.exit(2)
