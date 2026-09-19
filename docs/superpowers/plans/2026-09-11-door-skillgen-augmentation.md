# ARX Door SkillGen Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the approved `candidate_006` door demonstration into a native Isaac Lab SkillGen seed and generate physically replayed trials at `0`, `-3`, `+3`, `-5`, and `+5` degrees with the existing native cuMotion backend.

**Architecture:** Keep the existing 7-D absolute joint target contract at the simulator boundary and map it to SkillGen end-effector poses with deterministic FK/seeded IK. A project-local `ManagerBasedRLMimicEnv` loads the proven workcell snapshot, exposes door-specific object frames and success signals, and is driven by a project-local generator using an adapter for the existing isolated cuMotion process. SkillGen receives two non-overlapping macro skills—knob interaction and door-edge interaction—while all six semantic contact boundaries remain recorded as diagnostics; this makes cuMotion run only for the two genuine free-space transitions.

**Tech Stack:** Python 3.11, Isaac Sim 5.1, Isaac Lab `v2.3.2-13-gf4aa17f87e2`, Isaac Lab Mimic/SkillGen, native cuMotion Python 3.12 worker, NumPy, SciPy, PyTorch, h5py, pytest.

**Spec:** `docs/door_augmentation_route.md`

## Execution status — 2026-09-16

- Seed conversion, native environment registration, scene reset, and cuMotion
  adapter are implemented and exercised with the real simulator.
- The corrected 0-degree run passed physical acceptance at 32.373 degrees;
  both videos and the HDF5 contain 2058 pre-action frames at 30 Hz.
- The released panel keeps moving during free-space transfer. The project-local
  generator therefore uses an intermediate panel-facing pose, refreshes the
  panel reference once after transfer, and plans a short correction before the
  unchanged side-grasp skill. Global Isaac Lab code is not modified.
- All four offset trials completed, once per position: +3 and +5 degrees passed
  at 31.984 and 31.760 degrees, with 2054 and 2047 frames respectively. Both
  negative offsets passed knob manipulation but failed IK at the refreshed edge
  entrance; they remain failed samples, not successful augmentations.
- The five-position report is
  `generated/door_skillgen/validated_batch_20260916/batch_results.json`.
  All five HDF5 files were read and all ten videos decoded for frame counting:
  each video matches its episode at 30 Hz. There is one approved source seed
  and three automatically accepted generated episodes (6159 frames total).
- LeRobot export and N1.7 loader validation remain outside the completed work.

The original checklists below are the implementation design, not evidence of
physical completion. Actual acceptance is recorded in each trial's result.json
and the aggregate batch report.

## Global Constraints

- No reinforcement learning or reward optimization.
- Keep all business code in `/home/workspace/arx-r5-isaac-sim`; do not patch `/home/lbz/IsaacLab`.
- Preserve the training action contract: six arm joint targets in radians plus active gripper `joint7` in metres.
- Keep the wrist camera rigidly attached to `link6` with the approved downward-30-degree relationship.
- Every generated sample must be executed in physics and rendered again; transformed labels without execution are not accepted.
- Keep failed attempts with a structured failure stage and reason.
- Run the first batch in one environment at `0`, `-3`, `+3`, `-5`, and `+5` degrees, radius `0.59 m`, base height `0.63 m`, facing the handle.
- Preserve the current dirty working tree and do not commit unrelated user changes.

---

### Task 1: Skill and Seed Contract

**Files:**
- Create: `arx_r5_isaac_sim_bringup/arx_r5_isaac_sim_bringup/door_skillgen/__init__.py`
- Create: `arx_r5_isaac_sim_bringup/arx_r5_isaac_sim_bringup/door_skillgen/seed.py`
- Create: `scripts/convert_door_skillgen_seed.py`
- Create: `arx_r5_isaac_sim_bringup/test/test_door_skillgen_seed.py`

**Interfaces:**
- Consumes: approved attempt directory containing `episode.json` and `trajectory.npz`; `ArmKinematics.fk(joints) -> np.ndarray`.
- Produces: `load_seed(path: Path) -> DoorSeed`, `build_skillgen_arrays(seed, kin) -> dict[str, object]`, and `write_skillgen_hdf5(arrays, output, env_name) -> None`.

- [ ] **Step 1: Write failing seed validation and boundary tests**

```python
def test_approved_candidate_maps_to_two_non_overlapping_macro_skills(candidate_dir, kin):
    arrays = build_skillgen_arrays(load_seed(candidate_dir), kin)
    assert arrays["macro_boundaries"] == {
        "knob_interaction": (355, 1009),
        "edge_interaction": (1505, 1955),
    }
    assert arrays["eef_pose"][0].shape == (1955, 4, 4)
    assert arrays["target_eef_pose"][0].shape == (1955, 4, 4)
```

- [ ] **Step 2: Run the test and confirm it fails because the new module is absent**

Run: `python3 -m pytest arx_r5_isaac_sim_bringup/test/test_door_skillgen_seed.py -v`

- [ ] **Step 3: Implement validated loading, FK enrichment, object reference poses, fine-phase IDs, two macro start/term signals, and atomic HDF5 writing**

```python
SKILLGEN_MACRO_SKILLS = (
    MacroSkill("knob_interaction", "approach_knob", "retreat_from_knob", "knob"),
    MacroSkill("edge_interaction", "approach_door_edge", "hold_door_open", "panel"),
)
```

- [ ] **Step 4: Run the focused test and inspect the HDF5 with h5py**

Run: `python3 -m pytest arx_r5_isaac_sim_bringup/test/test_door_skillgen_seed.py -v`

- [ ] **Step 5: Convert `candidate_006` and retain a conversion report beside the HDF5**

Run: `python3 scripts/convert_door_skillgen_seed.py --input generated/door_teaching/upright02/candidate_006 --output generated/door_skillgen/seed/candidate_006.hdf5`

### Task 2: Native cuMotion SkillGen Adapter

**Files:**
- Create: `arx_r5_isaac_sim_bringup/arx_r5_isaac_sim_bringup/door_skillgen/cumotion_planner.py`
- Modify: `scripts/door_cumotion_worker.py`
- Create: `arx_r5_isaac_sim_bringup/test/test_door_skillgen_cumotion.py`

**Interfaces:**
- Consumes: SkillGen target pose `torch.Tensor[4,4]`, current ARX joint state, current scene cuboids, and `scripts/run_door_cumotion.sh`.
- Produces: `DoorCuMotionPlanner.update_world_and_plan_motion(...) -> bool`, `has_next_waypoint() -> bool`, and `get_next_waypoint_ee_pose() -> torch.Tensor` compatible with `MotionPlannerBase`.

- [ ] **Step 1: Write failing tests for request framing, failure retention, waypoint iteration, and planner reset**

```python
def test_planner_converts_world_pose_to_base_and_iterates_fk_waypoints(fake_runner, fake_env):
    planner = DoorCuMotionPlanner(fake_env, fake_env.scene["robot"], runner=fake_runner)
    assert planner.update_world_and_plan_motion(target_pose=TARGET_WORLD)
    assert planner.has_next_waypoint()
    assert planner.get_next_waypoint_ee_pose().shape == (4, 4)
```

- [ ] **Step 2: Run the test and confirm it fails because the adapter is absent**

Run: `python3 -m pytest arx_r5_isaac_sim_bringup/test/test_door_skillgen_cumotion.py -v`

- [ ] **Step 3: Implement a dependency-injected adapter and add a worker mode that accepts an already validated cuboid JSON path**

```python
planner = DoorCuMotionPlanner(
    env=env,
    robot=env.scene["robot"],
    runner=SubprocessCuMotionRunner(run_script, artifact_dir),
    kinematics=ArmKinematics(urdf_path),
)
```

- [ ] **Step 4: Run focused tests and one existing native-worker validation request**

Run: `python3 -m pytest arx_r5_isaac_sim_bringup/test/test_door_skillgen_cumotion.py -v`

### Task 3: Native Isaac Lab Mimic Environment

**Files:**
- Create: `arx_r5_isaac_sim_bringup/arx_r5_isaac_sim_bringup/door_skillgen/mdp.py`
- Create: `arx_r5_isaac_sim_bringup/arx_r5_isaac_sim_bringup/door_skillgen/env_cfg.py`
- Create: `arx_r5_isaac_sim_bringup/arx_r5_isaac_sim_bringup/door_skillgen/env.py`
- Create: `arx_r5_isaac_sim_bringup/arx_r5_isaac_sim_bringup/door_skillgen/registry.py`
- Create: `arx_r5_isaac_sim_bringup/test/test_door_skillgen_contract.py`

**Interfaces:**
- Consumes: a prepared workcell USD containing `/R5a`, `/World/Door`, and `TeachingWristCamera`; seven absolute joint target actions.
- Produces: registered environment `Isaac-ARX-R5-Door-SkillGen-v0`, `get_robot_eef_pose`, `target_eef_pose_to_action`, `action_to_target_eef_pose`, `actions_to_gripper_actions`, `get_object_poses`, and door success/subtask signals.

- [ ] **Step 1: Write failing pure contract tests for action/FK round trips, signal monotonicity, base-placement selection, and success thresholds**

```python
def test_action_pose_round_trip_preserves_the_seed_branch(contract, seed_action):
    pose = contract.action_to_target_pose(seed_action)
    recovered = contract.target_pose_to_action(pose, seed_action)
    np.testing.assert_allclose(recovered[:6], seed_action[:6], atol=5e-4)
```

- [ ] **Step 2: Run the contract tests and confirm the production contract does not exist**

Run: `python3 -m pytest arx_r5_isaac_sim_bringup/test/test_door_skillgen_contract.py -v`

- [ ] **Step 3: Implement pure math/threshold logic, then the Isaac Lab-only environment/configuration modules behind lazy runtime imports**

```python
subtask_configs = {
    "link6": [
        SubTaskConfig(object_ref="knob", subtask_term_signal="knob_interaction"),
        SubTaskConfig(object_ref="panel", subtask_term_signal="edge_interaction"),
    ]
}
```

- [ ] **Step 4: Run focused tests and import the registry with Isaac Lab Python**

Run: `/home/lbz/IsaacLab/isaaclab.sh -p -c 'from arx_r5_isaac_sim_bringup.door_skillgen.registry import ENV_ID; print(ENV_ID)'`

### Task 4: Scene Preparation, Generation, and Five-Position Batch

**Files:**
- Create: `scripts/prepare_door_skillgen_scene.py`
- Create: `scripts/generate_door_skillgen.py`
- Create: `scripts/run_door_skillgen_batch.py`
- Create: `arx_r5_isaac_sim_bringup/test/test_door_skillgen_batch.py`
- Modify: `arx_r5_isaac_sim_bringup/setup.py`

**Interfaces:**
- Consumes: approved flattened snapshot, converted HDF5, explicit angle list, and native planner adapter.
- Produces: per-angle scene, generated HDF5, wrist/overview video, `result.json`, and aggregate `batch_results.json` with every attempt preserved.

- [ ] **Step 1: Write failing batch-manifest tests**

```python
def test_default_first_batch_is_fixed_and_reproducible():
    trials = build_trial_specs([0.0, -3.0, 3.0, -5.0, 5.0], seed=0)
    assert [trial.angle_deg for trial in trials] == [0.0, -3.0, 3.0, -5.0, 5.0]
    assert len({trial.output_name for trial in trials}) == 5
```

- [ ] **Step 2: Run the test and confirm the batch API is absent**

Run: `python3 -m pytest arx_r5_isaac_sim_bringup/test/test_door_skillgen_batch.py -v`

- [ ] **Step 3: Implement atomic scene/output staging, a project-local copy of the official generation entry point with custom planner injection, and sequential batch orchestration**

```bash
/home/lbz/IsaacLab/isaaclab.sh -p scripts/run_door_skillgen_batch.py \
  --seed-hdf5 generated/door_skillgen/seed/candidate_006.hdf5 \
  --angles 0 -3 3 -5 5 --headless
```

- [ ] **Step 4: Run offline batch tests, then execute the five-position simulator batch**

Run: `python3 -m pytest arx_r5_isaac_sim_bringup/test/test_door_skillgen_batch.py -v`

- [ ] **Step 5: Validate every attempt has physics status, failure stage/reason, frame count, camera artifacts, and final door angle**

Run: `python3 scripts/run_door_skillgen_batch.py --validate-only --output generated/door_skillgen/first_batch`

### Task 5: Training Export and Documentation

**Files:**
- Modify: `scripts/export_door_lerobot.py`
- Modify: `docs/door_augmentation_route.md`
- Create: `docs/door_skillgen_usage.md`
- Create: `arx_r5_isaac_sim_bringup/test/test_door_skillgen_lerobot.py`

**Interfaces:**
- Consumes: successful generated episodes plus their HDF5/diagnostic manifests.
- Produces: LeRobot v2.1-compatible dataset with `meta/modality.json`, source-seed lineage, angle metadata, and no failed trials in training episodes.

- [ ] **Step 1: Write a failing export test using one successful and one failed generated attempt**

```python
def test_export_keeps_success_only_and_records_skillgen_lineage(tmp_path, attempts):
    export_generated_attempts(attempts, tmp_path / "lerobot")
    info = json.loads((tmp_path / "lerobot/meta/info.json").read_text())
    assert info["total_episodes"] == 1
    assert info["source_seed"] == "upright02/candidate_006"
```

- [ ] **Step 2: Run the test and confirm generated-attempt discovery is absent**

Run: `python3 -m pytest arx_r5_isaac_sim_bringup/test/test_door_skillgen_lerobot.py -v`

- [ ] **Step 3: Extend export discovery and write operator documentation with exact commands and artifact layout**

- [ ] **Step 4: Run all directly affected tests and lint the new Python files**

Run: `python3 -m pytest arx_r5_isaac_sim_bringup/test/test_door_skillgen_*.py -v`

Run: `python3 -m flake8 arx_r5_isaac_sim_bringup/arx_r5_isaac_sim_bringup/door_skillgen scripts/*door_skillgen*.py`

- [ ] **Step 5: Inspect the final diff against this plan and the approved route before reporting results**

Run: `git diff --check && git status --short`
