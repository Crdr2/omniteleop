# Third-party notices

`src/omniteleop/assets/vega_with_robotiq.urdf` is a kinematics-only derivative of:

- Dexmate's `dexmate-urdf` 0.8.0 Vega model, licensed under Apache-2.0.
- `yixuan_utilities`' Vega/Robotiq model, copyright 2024 Yixuan Wang, licensed under MIT.

The derivative was modified on 2026-08-28 to remove visual and collision mesh elements
while retaining the joint/link kinematics and Robotiq frames needed by the IK pipeline.
The corresponding license texts are in `src/omniteleop/assets/`.
