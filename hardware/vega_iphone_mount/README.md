# iPhone base-tracking mount

Current fused mount for an iPhone 13 Pro Max with a rubber case, attached above
the Vega's left E-stop. The phone is rigid to `base`, stands vertically, and looks
along robot `+x`. This is the geometry covered by the IK's phone-mount envelope.

Print one `07_estop_tower_cradle.stl` and one `06_top_stop.stl`. STEP files are
included for editing. Print `08_fit_test_coupon.stl` first to check the case fit.
The nominal phone is 78.1 x 7.65 x 160.8 mm; case parameters live in
`02_iphone_cradle.py`. The fused part imports those same parameters.

The fused part is approximately 80 x 100 x 244 mm. Check printer height and phone
fit before printing. Secure it with the two existing E-stop M4 anchors, spaced
47 mm apart. New screw length is the measured original screw length plus 6 mm;
the model does not establish the real thread depth. Check seating, E-stop access,
fan clearance, and tool access on the actual robot before operation.

The top stop uses integral press pins and a 6 mm soft foam pad in its recess.
Press it onto the rail tops after seating the phone; the compressed pad prevents
vertical play. Check that the case buttons do not prevent full seating.

`07` and its STL are already in robot `base` coordinates, in millimetres: scale by
0.001 for metres, with no additional transform. `06` is in cradle coordinates;
place it with a +90 degree Z rotation and base-frame translation
`(149, 240, 524.8)` mm. `mount_pose.json` records the placement and axes.

To regenerate in an environment with `build123d` installed:

```bash
python hardware/vega_iphone_mount/08_fit_test_coupon.py
python hardware/vega_iphone_mount/07_estop_tower_cradle.py
python hardware/vega_iphone_mount/06_top_stop.py
```

These commands write STEP/STL files beside the scripts. Calibrate the actual
phone-to-base transform with the [tracking tools](../../scripts/base_tracking/README.md);
the CAD camera position is an approximate prior, not a usable extrinsic calibration.
