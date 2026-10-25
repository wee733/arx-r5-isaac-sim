#!/usr/bin/env python3
import unittest
import numpy as np
from arx_door.real.contract import observation, from_policy_reply


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.state = {"schema": "arx_vla_observation_v1", "calibrated": True,
                      "session_id": "test", "observation_id": 2,
                      "state": {"eef": [0] * 6, "gripper": [0.01], "joints": [0] * 6}}
        self.rgb = {key: np.zeros((480, 640, 3), dtype=np.uint8) for key in ("wrist", "ego")}
        self.reply = {"eef": np.zeros((1, 16, 6)), "gripper": np.full((1, 16, 1), 0.01)}

    def test_observation(self):
        obs = observation(self.state, self.rgb, "open the door")
        self.assertEqual(obs["state"]["joints"].shape, (1, 1, 6))
        self.assertEqual(obs["video"]["ego"].shape, (1, 1, 480, 640, 3))
        self.state["calibrated"] = False
        with self.assertRaises(ValueError):
            observation(self.state, self.rgb, "open")

    def test_absolute_no_double_add(self):
        self.reply["eef"][:] = 0.2
        self.state["state"]["eef"] = [0.3] * 6
        chunk = from_policy_reply(self.reply, self.state, 1)
        self.assertEqual(chunk["eef"][0], [0.2] * 6)
        self.assertEqual(len(chunk["eef"]), 8)
        self.assertEqual(chunk["observation_id"], 2)
        self.assertAlmostEqual(chunk["dt_s"], 1 / 30)

    def test_malformed_or_out_of_range(self):
        for bad in ({"single_arm": self.reply["eef"], "gripper": self.reply["gripper"]},
                    {"eef": np.zeros((16, 6)), "gripper": self.reply["gripper"]},
                    {"eef": self.reply["eef"], "gripper": np.full((1, 16, 1), 0.05)},
                    {"eef": np.full((1, 16, 6), np.nan), "gripper": self.reply["gripper"]}):
            with self.assertRaises(ValueError):
                from_policy_reply(bad, self.state, 1)


if __name__ == "__main__":
    unittest.main()
