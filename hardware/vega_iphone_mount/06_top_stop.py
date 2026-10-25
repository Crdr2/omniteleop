"""Press-fit top stop with integral pins and a soft-foam preload recess.

Cradle-local mm; the underside sits at cradle Z=196.8. Use a 6 mm soft pad in
the 2 mm recess to retain the phone without vertical play. Check pin fit before
assembly; the nominal 2.6 mm pins fit the cradle's 2.7 mm rail holes.
"""

import os

from build123d import Align, Box, Cone, Cylinder, Pos, export_step, export_stl

OUT = os.path.dirname(os.path.abspath(__file__))  # this folder

BAR_X0, BAR_X1 = -50.0, 50.0  # matches the 100 mm cased-phone cradle
BAR_Y0, BAR_Y1 = -4.0, 15.0  # rails are now 19 mm deep
BAR_T = 8.0

PAD_X0, PAD_X1 = -37.0, 37.0  # over the cased phone (case spans X -41.05..+41.05)
PAD_Y0, PAD_Y1 = 0.0, 10.5  # over the cased phone's thickness (Y 0..9.65)
PAD_DEPTH = 2.0

PIN_X, PIN_Y = 44.0, 4.0  # rail-top holes: D2.7, Z 178.8..196.8 in the cradle
PIN_D = 2.6  # nominal 0.1 clearance; FDM shrinkage makes it a press fit
PIN_L = 13.0  # hole is 18 deep -> 5 mm of air below the tip
TIP_L, TIP_D = 2.5, 1.6  # cone lead-in so the pin finds the hole


def _box(x0, x1, y0, y1, z0, z1):
    return Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * Box(x1 - x0, y1 - y0, z1 - z0)


part = _box(BAR_X0, BAR_X1, BAR_Y0, BAR_Y1, 0.0, BAR_T)
part -= _box(PAD_X0, PAD_X1, PAD_Y0, PAD_Y1, -0.5, PAD_DEPTH)

for sx in (-1, 1):
    shank = Pos(sx * PIN_X, PIN_Y, -(PIN_L - TIP_L)) * Cylinder(
        PIN_D / 2, PIN_L - TIP_L, align=(Align.CENTER, Align.CENTER, Align.MIN)
    )
    tip = Pos(sx * PIN_X, PIN_Y, -PIN_L) * Cone(
        TIP_D / 2, PIN_D / 2, TIP_L, align=(Align.CENTER, Align.CENTER, Align.MIN)
    )
    part += shank + tip

b = part.bounding_box()
print(f"bounds  x {b.min.X:7.2f} .. {b.max.X:7.2f}")
print(f"        y {b.min.Y:7.2f} .. {b.max.Y:7.2f}")
print(f"        z {b.min.Z:7.2f} .. {b.max.Z:7.2f}")
print(f"volume  {part.volume / 1000:.1f} cm3   solids {len(part.solids())}")
print(f"pins    D{PIN_D} x {PIN_L} at (+-{PIN_X}, {PIN_Y}), cone tip to D{TIP_D}")
print("print   top face down, pins up — no supports")

export_step(part, f"{OUT}/06_top_stop.step")
export_stl(part, f"{OUT}/06_top_stop.stl")
print("exported 06_top_stop.step / .stl  (cradle local frame)")
