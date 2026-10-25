# Door asset provenance

Copied as regular files from the user's existing local `assets/door` on 2026-09-11.
The scene does not require the old workspace or symbolic links.

`door.usda` composes `door_visuals.usdc` and its `textures/` directory. Existing
hinge, handle, latch, collision shapes and dynamics are preserved. Dynamics in
the source asset are marked as unmeasured initial values; the placement scene
does not claim validated opening forces or successful robot manipulation.

Closed-pose handle spindle: `(0.313, -0.3082, 0.928)` m.
Front grasp target: `(0.313, -0.3752, 0.928)` m. Front normal: world `-Y`.
The placement radius uses the front grasp target's XY projection.

This file records the local provenance; it does not assign a new license to
the source meshes or textures.
