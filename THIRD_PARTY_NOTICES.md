# Third-party notices

`src/omniteleop/assets/vega_with_robotiq.urdf` is a kinematics-only derivative of:

- Dexmate's `dexmate-urdf` 0.8.0 Vega model, licensed under Apache-2.0.
- `yixuan_utilities`' Vega/Robotiq model, copyright 2024 Yixuan Wang, licensed under MIT.

The derivative was refreshed on 2026-10-25 from the calibrated wrist-camera variant.
It removes visual and collision mesh elements while retaining joint/link kinematics,
inertial data, Robotiq and wrist-camera frames, and per-link mesh-origin metadata needed
to align the external collision spheres. No rendering meshes are redistributed.
The corresponding license texts are in `src/omniteleop/assets/`.

`third_party/record3d/latest-frame-only.patch` modifies the Record3D library by
Marek Simonik (LGPL-2.1). It targets upstream revision
`79e6f44697150a4f036034af8fec903d11ac6f00`; its license is included at
`third_party/record3d/LICENSE`. No compiled SDK binaries are included.
