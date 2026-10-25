# Canonical Robot USD Package

This directory contains the versioned Isaac Sim runtime asset generated from
`../my_robot.urdf`:

- `my_robot.usd`
- `configuration/my_robot_base.usd`
- `configuration/my_robot_physics.usd`
- `configuration/my_robot_robot.usd`
- `configuration/my_robot_sensor.usd`

Fetch these files with Git LFS together with the mesh directory; see
[`assets/README.md`](../../README.md). This is the existing source conversion,
not a complete task environment. The task's runtime package is prepared by
[`prepare_b2arx.py`](../../../scripts/door/autodata/prepare_b2arx.py).
Regenerate only after intentionally changing the URDF or converter settings.

Commit all five USD files together. Do not commit `.asset_hash` or `config.yaml`:
they are converter metadata and `config.yaml` contains machine-local absolute
paths.
