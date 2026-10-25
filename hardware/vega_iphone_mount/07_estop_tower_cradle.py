"""Fused E-stop tower and phone cradle in robot-base millimetres.

Import the cradle builder so the fit coupon and fused part share dimensions.
Omit its plate holes, then cut the two robot-anchor bores through the union so
they remain accessible through the cradle plate.
"""

import math
import os
import runpy

from build123d import Align, Box, Cylinder, Location, Pos, Rot, export_step, export_stl

OUT = os.path.dirname(os.path.abspath(__file__))  # this folder
C = runpy.run_path(f"{OUT}/02_iphone_cradle.py", run_name="_cradle")  # cradle constants

# ================================================================== TOWER ===
BTN_X, BTN_Y, BTN_Z = 149.0, 240.00, 290.95  # ledge plane point on the button axis
TILT = 20.0  # deg, ledge normal from vertical toward +y
BOLT_DX = 23.5  # bolt offset from the button axis, along x
LEDGE = Pos(BTN_X, BTN_Y, BTN_Z) * Rot(X=-TILT)

SKIRT_OD, SKIRT_ID = 54.0, 40.0  # r 20..27, inside the free annulus r 19..28
SKIRT_H = 20.5  # bezel tabs top out at h=18.0
DISC_OD = 68.0
DISC_TOP = 45.0

PLATE_X0, PLATE_X1 = 109.0, 189.0  # = cradle local Y +-40, mapped to robot x
PLATE_Y0, PLATE_Y1 = 190.0, 290.0  # = cradle local X +-50, mapped to robot y
PLATE_BOT, PLATE_TOP = 316.0, 328.0

BORE = 4.5  # M4 clearance
CBORE = 9.5  # M4 socket-head pocket, h 6..11
TOOL_D = 14.0  # wide tool shaft, takes a 1/4in bit holder
TOOL_Z = 11.0
CBORE_Z = 6.0  # head seat: screw length = (original E-stop screw) + 6 mm

# ================================================================= CRADLE ===
# cradle local frame:  X -> robot +y,  Y -> robot -x,  Z -> robot +z
CR_O = (149.0, 240.0, 328.0)
CRADLE = Pos(*CR_O) * Rot(Z=90)


def box(x0, x1, y0, y1, z0, z1):
    return Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * Box(x1 - x0, y1 - y0, z1 - z0)


# tower solid
skirt = LEDGE * (
    Cylinder(SKIRT_OD / 2, SKIRT_H, align=(Align.CENTER, Align.CENTER, Align.MIN))
    - Cylinder(SKIRT_ID / 2, SKIRT_H, align=(Align.CENTER, Align.CENTER, Align.MIN))
)
disc = (
    LEDGE
    * Pos(0, 0, SKIRT_H)
    * Cylinder(DISC_OD / 2, DISC_TOP - SKIRT_H, align=(Align.CENTER, Align.CENTER, Align.MIN))
)
tower = skirt + disc + box(PLATE_X0, PLATE_X1, PLATE_Y0, PLATE_Y1, PLATE_BOT, PLATE_TOP)
tower -= Pos(BTN_X, BTN_Y, PLATE_TOP + 100) * Box(500, 500, 200)  # flat top at z=328
tower -= LEDGE * Pos(0, 0, -100) * Box(500, 500, 200)  # nothing below the ledge

# cradle solid: the SAME builder 02 uses, minus the base-plate bolts
# Calling 02's function rather than restating its geometry means a feature added
# there (the button relief, for instance) cannot go missing here.
part = tower + CRADLE * C["build_cradle"](with_bolt_holes=False)

# ---- bolt bores cut AFTER the union, so they break out above the base plate -
for sx in (-1, 1):
    part -= (
        LEDGE
        * Pos(sx * BOLT_DX, 0, -5)
        * Cylinder(BORE / 2, 130, align=(Align.CENTER, Align.CENTER, Align.MIN))
    )
    part -= (
        LEDGE
        * Pos(sx * BOLT_DX, 0, CBORE_Z)
        * Cylinder(CBORE / 2, 130, align=(Align.CENTER, Align.CENTER, Align.MIN))
    )
    part -= (
        LEDGE
        * Pos(sx * BOLT_DX, 0, TOOL_Z)
        * Cylinder(TOOL_D / 2, 130, align=(Align.CENTER, Align.CENTER, Align.MIN))
    )

# report
b = part.bounding_box()
print()
print(f"cased envelope  {C['ENV_W']:.2f} W x {C['ENV_D']:.2f} D x {C['ENV_H']:.2f} H")
print(
    f"channel {C['CHANNEL']:.2f} (slop {C['SLOP_X']:.2f})   "
    f"pocket {C['POCKET']:.2f} (slop {C['SLOP_Y']:.2f})   top gap {C['TOP_GAP']:.2f}"
)
print(f"bounds  x {b.min.X:7.2f} .. {b.max.X:7.2f}   ({b.max.X - b.min.X:6.1f} mm)")
print(f"        y {b.min.Y:7.2f} .. {b.max.Y:7.2f}   ({b.max.Y - b.min.Y:6.1f} mm)")
print(f"        z {b.min.Z:7.2f} .. {b.max.Z:7.2f}   ({b.max.Z - b.min.Z:6.1f} mm)")
print(f"volume  {part.volume / 1000:.1f} cm3   solids {len(part.solids())}")
n = (0.0, math.sin(math.radians(TILT)), math.cos(math.radians(TILT)))
print(f"ledge normal N = ({n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f})")
for nm, sx in (("rear", -1), ("front", +1)):  # robot +x is forward
    p = LEDGE * Location((sx * BOLT_DX, 0, 0))
    print(
        f"  {nm} bolt axis through ({p.position.X:7.2f}, {p.position.Y:7.2f}, {p.position.Z:7.2f})"
    )

export_step(part, f"{OUT}/07_estop_tower_cradle.step")
export_stl(part, f"{OUT}/07_estop_tower_cradle.stl")
print("\nexported 07_estop_tower_cradle.step / .stl  (robot base frame)")
