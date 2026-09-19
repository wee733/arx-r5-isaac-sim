# ARX door — single-wrist GR00T N1.7 dataset

1 successful generated episodes, 2058 frames at 30 Hz.

LeRobot v2.1; wrist RGB only; absolute joint1..joint6 radians and joint7 metres. Use training/door_wrist_n17.py: relative arm targets, absolute gripper, horizon 16.

Official loader and statistics verified against NVIDIA Isaac-GR00T 9c7e746b2cd37a810070a98ef41d290a07e806c2. See meta/n17_validation.json for runtime, input hashes and validation coverage.

No model training was performed. This small, narrow-position pilot is not a generalization benchmark or a claim of trained policy success. Source seed lineage and physical acceptance are in meta/door_export_report.json.
