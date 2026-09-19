# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Guard the episode boundary ordering in the driver.

Two orderings matter and neither raises when wrong:

* Homing must happen BEFORE the scene reset. Going home with the new layout
  already in place drives the arm through wherever the block was just
  teleported to.
* The STARTED event must be published AFTER setup completes. A recorder opens
  its episode on that event, so homing published inside the window would be
  recorded as part of the demonstration and learned as one.
"""

from pathlib import Path


DRIVER_SOURCE = (
    Path(__file__).resolve().parents[1]
    / 'arx_r5_isaac_sim_bringup'
    / 'vla'
    / 'episode_driver.py'
).read_text(encoding='utf-8')


def _run_episode_source() -> str:
    start = DRIVER_SOURCE.index('    def _run_episode(self')
    end = DRIVER_SOURCE.index('    # ------', start)
    return DRIVER_SOURCE[start:end]


def _live_episode_source() -> str:
    """Return only the executing path, past the dry-run early return."""
    source = _run_episode_source()
    return source[source.index('# Setup, deliberately outside'):]


def test_homing_precedes_the_scene_reset():
    """Resetting first would drop the block where the arm is about to travel."""
    source = _live_episode_source()
    home = source.index('self._go_home()')
    reset = source.index('self._reset_scene(seed)')
    assert home < reset


def test_started_event_is_published_after_setup():
    """A recorder opens its episode on STARTED; setup must be outside it."""
    source = _live_episode_source()
    reset = source.index('self._reset_scene(seed)')
    started = source.index('EpisodeStatus.STARTED')
    assert reset < started


def test_acquisition_reachability_screening_is_outside_recording():
    """Real APPROACH probes must not become demonstration frames."""
    source = _live_episode_source()
    preflight = source.index('self._select_acquisition_rotation(waypoints)')
    started = source.index('EpisodeStatus.STARTED')
    assert preflight < started


def test_scene_is_reset_after_real_acquisition_screening():
    """Probe motion must be followed by a fresh deterministic block reset."""
    source = _live_episode_source()
    preflight = source.index('self._select_acquisition_rotation(waypoints)')
    reset = source.index('self._reset_scene(seed)', preflight)
    started = source.index('EpisodeStatus.STARTED')
    assert preflight < reset < started


def test_episode_homes_exactly_once():
    """The old flow homed at both ends, duplicating the motion every episode."""
    assert _run_episode_source().count('self._go_home()') == 1


def test_episode_ends_after_placement_verification():
    """The terminal event must label validation, not the prior retreat move."""
    source = _live_episode_source()
    verification = source.index(
        'self._wait_for_placement(place_center, place_yaw)'
    )
    terminal = source.rindex('EpisodeStatus.SUCCEEDED')
    assert verification < terminal
    assert 'EpisodePhase.VERIFY_PLACE' in source[terminal - 200:]
    assert 'self._go_home()' not in source[terminal:]


def test_success_requires_the_released_block_to_settle_on_target():
    """Release plus retreat is insufficient without a ground-truth check."""
    source = _live_episode_source()
    verification = source.index(
        'self._wait_for_placement(place_center, place_yaw)'
    )
    terminal = source.rindex('EpisodeStatus.SUCCEEDED')
    assert verification < terminal


def test_waypoints_run_after_the_started_event():
    """Every recorded frame must fall inside the episode window."""
    source = _live_episode_source()
    started = source.index('EpisodeStatus.STARTED')
    waypoints = source.index('for phase, translation, rotation in waypoints')
    assert started < waypoints


def test_observe_labels_initial_frames_before_planning():
    """Frames captured while the first goal is planned are observations."""
    source = _live_episode_source()
    started = source.index('EpisodeStatus.STARTED')
    observe = source.index('EpisodePhase.OBSERVE', started)
    waypoints = source.index('for phase, translation, rotation in waypoints')
    assert started < observe < waypoints


def test_planning_failure_is_attributed_to_the_attempted_phase():
    """A plan can fail before RUNNING is published, so mark its phase first."""
    source = _live_episode_source()
    loop = source.index('for phase, translation, rotation')
    mark = source.index('self._current_episode_phase = phase', loop)
    plan = source.index('self._motion.plan_to_pose_goalset(', loop)
    assert mark < plan


def test_failure_event_uses_the_tracked_phase():
    """Do not mislabel every MotionError as a retreat failure."""
    run_start = DRIVER_SOURCE.index('    def _run(self, run_id: str')
    run_end = DRIVER_SOURCE.index('    def start_run', run_start)
    run_source = DRIVER_SOURCE[run_start:run_end]
    failed = run_source.index('EpisodeStatus.FAILED')
    assert 'self._current_episode_phase' in run_source[failed - 150:failed]


def test_dry_run_short_circuits_before_any_motion():
    """A dry run must plan only; it must not home or move the arm."""
    source = _run_episode_source()
    dry_run = source.index('if self._dry_run:')
    home = source.index('self._go_home()')
    assert dry_run < home
    assert 'return' in source[dry_run:home]


def test_dry_run_does_not_open_a_recordable_episode():
    """Reachability checks must never be committed as demonstrations."""
    source = _run_episode_source()
    dry_run = source[source.index('if self._dry_run:'):]
    dry_run = dry_run[:dry_run.index('# Setup, deliberately outside')]
    assert 'self._publish_event(' not in dry_run


def test_collection_worker_is_joinable_not_daemonized():
    """Process shutdown must wait for the worker to close its episode."""
    assert 'daemon=False' in DRIVER_SOURCE
    assert 'def stop_run(' in DRIVER_SOURCE
    assert 'EpisodeStatus.ABORTED' in DRIVER_SOURCE


def test_repeated_collect_calls_advance_the_seed_range():
    """A second run must not silently regenerate the first run's layouts."""
    start = DRIVER_SOURCE.index('    def start_run(self)')
    end = DRIVER_SOURCE.index('    def stop_run(', start)
    source = DRIVER_SOURCE[start:end]
    assert 'start_seed = self._next_seed' in source
    assert 'self._next_seed += self._episode_count' in source


def test_unexpected_episode_exception_still_publishes_a_terminal_event():
    """Non-MotionError failures must not leave the recorder half-open."""
    run_start = DRIVER_SOURCE.index('    def _run(self, run_id: str')
    run_end = DRIVER_SOURCE.index('    def start_run', run_start)
    source = DRIVER_SOURCE[run_start:run_end]
    unexpected = source.index('except Exception as error')
    failed = source.index('EpisodeStatus.FAILED', unexpected)
    assert unexpected < failed
