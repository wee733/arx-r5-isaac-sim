# Model validation before high-fidelity use

The current model is sufficient to establish the software execution loop, but
it is not yet a calibrated digital twin.

## Known findings

- The description-package `r5a_cumotion.urdf` must be used. The legacy
  repository-root `R5a.urdf` has stale broad joint limits and no reliable
  simulation contract.
- The vendor driver's `X5liteaa0.urdf` differs from the MoveIt/cuMotion model at
  the joint 5 and joint 6 origins. The deterministic audit below measured
  4.427633 mm median, 6.930694 mm p95, and 7.192397 mm maximum translation
  disagreement at `link6`. This does not block joint-space execution, but it
  affects Cartesian truth and the vendor-reported end pose.
- `link6` is currently the cuMotion tool frame, while the physical grasp center
  is forward of that frame. A measured TCP frame should be added and used by
  MoveIt, cuMotion, and task logic together.
- The collision elements reference high-triangle visual STL meshes. Isaac Sim
  therefore uses convex collision approximation unless
  `--convex-decomposition` is selected. Production scenes should use reviewed,
  simplified convex collision meshes.
- A sphere-vs-mesh audit found exposed areas larger than the current 2 mm XRDF
  buffer on several links. The worst mesh vertex was 22.642073 mm outside the
  nominal sphere union and 20.642073 mm outside after the buffer, on `link3`.
  Treat the current XRDF as provisional for obstacle-clearance claims.

## Reproducing the audit

The measurements above were regenerated on 2026-07-21 with
[`scripts/audit_model.py`](../scripts/audit_model.py). The script is read-only,
uses seed `20260721`, and defaults to 10,000 joint-space poses plus 10,000
triangle-area-weighted surface samples per XRDF link. FK uses the URDF-standard
`T_origin * T_axis(q)` convention and samples each joint uniformly inside the
`r5a_cumotion.urdf` hard limits. Mesh gaps are
`min(norm(point - center) - radius - buffer)`; positive values are uncovered.

Inputs used for the recorded result:

| Input | Revision / SHA-256 |
|---|---|
| ARX description repository | `c85c2c7c84bb630f30f87904bebebbd83dddd2db` |
| `r5a_cumotion.urdf` | `87aa0150c65c547f35ce604b29ba82efdee53c8199db40d6887890beb6b7a27e` |
| `r5a.xrdf` | `ec59b7ce9a2b015113492299886cefc7972db22901eb2a5d9ef8a8524f93fa16` |
| Vendor R5 repository | `f58330999779dc4c79907c8f73950311ca56f2b0` |
| Vendor `X5liteaa0.urdf` | `77a3cd0ae94e0920dcbd4475409b22c1786dd0e2fdabee5d9915a9ee90948df6` |

The recorded environment was Python 3.11.15, NumPy 1.26.0, and trimesh 4.5.1.
NumPy is required; trimesh is optional for FK-only runs and required for the
sphere/mesh section:

```bash
python -m pip install numpy trimesh
./scripts/audit_model.py \
  --description-share /path/to/isaac_ros_manipulation_arx_r5a_robot_description \
  --vendor-urdf /path/to/X5liteaa0.urdf \
  --require-mesh
```

Use `--output json` to archive the full inputs, hashes, joint-origin deltas,
sampling settings, per-link coverage metrics, and worst-case pose. Vertex
extrema are useful for finding corner gaps but are tessellation-biased; the
script reports the area-weighted surface sample separately.

## Acceptance gates

Before calling the simulator a digital twin:

1. Select the authoritative calibrated URDF and compare random-pose FK among
   the vendor model, MoveIt, cuMotion, and Isaac Sim.
2. Measure and add the actual TCP/grasp-center frame.
3. Verify joint sign, zero, limit, and mimic behavior at all eight joints.
4. Replace visual-mesh collision proxies with simplified convex meshes.
5. Re-audit XRDF collision spheres against those approved collision meshes.
6. Run stepped joint targets, then MoveIt trajectories, and record tracking
   error, overshoot, settling time, and limit behavior before tuning the default
   625/50 position-drive gains.
