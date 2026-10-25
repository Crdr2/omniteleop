"""Parametric iPhone 13 Pro Max cradle for a rubber case.

Adjust CASE_T and CASE_LIP to measured case dimensions, then print the fit coupon.
Coordinates are cradle-local mm: X across the phone, Y toward its screen, Z up.
Mounted axes are robot +y, -x, +z respectively.
"""

import os

from build123d import Align, Box, Cylinder, Pos, export_step, export_stl

OUT = os.path.dirname(os.path.abspath(__file__))  # this folder

# phone
PHONE_W, PHONE_D, PHONE_H = 78.10, 7.65, 160.80
CASE_T = 2.00  # case wall: top, bottom, left, right, rear

ENV_W = PHONE_W + 2 * CASE_T  # 82.10
CASE_LIP = 1.00  # raised lip standing proud of the screen
ENV_D = PHONE_D + CASE_T + CASE_LIP  # 10.65 — rear wall + phone + front lip
ENV_H = PHONE_H + 2 * CASE_T  # 164.80

# Keep the case channel snug; relieve protruding button covers locally
# rather than increasing lateral play along the full rail.
SLOP_X = 0.80
CHANNEL = ENV_W + SLOP_X  # 82.50
RAIL_IN = CHANNEL / 2  # 41.25
RAIL_OUT = 50.0
PLATE_HW = 50.0
PLATE_HD = 40.0
PLATE_T = 6.0

# Pocket depth follows printed case-fit measurements, not the bare-phone
# estimate. Preserve a light slip fit without rocking or press-fit force.
SLOP_Y = 0.75
POCKET = ENV_D + SLOP_Y  # 10.90
LIP_W = 6.0
RAIL_Y0, RAIL_Y1 = -4.0, 15.0
LIP_Y0, LIP_Y1 = POCKET, RAIL_Y1

# Button relief.  Apple publishes no button coordinates, so this is built to not
# need them: the channel is snug only over the LOWER part of the case, and is
# relieved from BUTTON_Z upwards.  On every iPhone the side buttons sit well
# inside the upper half, so a snug band over the bottom ~45 % of the case is
# always below them.  75 mm of snug rail is plenty to fully constrain a rigid
# phone in X and in yaw — a body clamped over 75 mm of its length cannot
# translate or rotate, so leaving the top free costs nothing.
BUTTON_Z = 105.0  # relief starts here, ~45 % of the case height
BUTTON_RELIEF = 1.5  # per side, clears a proud button cover

LEDGE_TOP = 30.0  # 26 mm of room beneath for a charging plug
LEDGE_Y1 = 14.0
# Top gap.  It cannot be zero, and it cannot be closed by geometry alone.
# The stop seats on the RAIL TOPS; the phone's top is a different surface, and
# the distance between them is a tolerance stack — phone height, both moulded
# case walls, the printed ledge height and the printed rail height, ~+-1.4 mm
# worst case now that the case has measured thicker than its nominal 2 mm.  Set
# the gap to zero and roughly half the time the case top lands ABOVE the rail
# top: the stop perches on the phone, its pins sit half out, and nothing is
# located.  Whatever the nominal, a filler closes the residual.
# It cannot be a rigid filler either.  Rubber is nearly incompressible and, once
# confined, needs tens to hundreds of newtons to give up a millimetre — pressing
# the stop onto the case's own rubber top would either not seat or bow the part.
# Foam gives the same millimetre for a few newtons, which is why the pad is foam.
# 2.00 is as low as the stack allows: real gap 0.6..3.4 mm, so the stop always
# lands on the rails, and a 6 mm pad in the stop's 2 mm recess is always
# compressed (0.6..3.4 mm).  Down from 3.00.
TOP_GAP = 2.00
RAIL_TOP = LEDGE_TOP + ENV_H + TOP_GAP  # 200.00
BACK_TOP = 131.0  # stays clear of the camera island

BOLT_D, BOLT_XY = 4.5, 26.0  # to the tower, two-part build only
PIN_D, PIN_X, PIN_Y = 2.7, 44.0, 4.0
PIN_Z0 = RAIL_TOP - 18.0

WELL_HW, WELL_Y0, WELL_Y1, WELL_Z0, WELL_Z1 = 17.0, -6.0, 17.0, 4.0, LEDGE_TOP + 1
GROOVE_HW, GROOVE_Z0, GROOVE_Z1 = 9.0, 3.0, PLATE_T + 0.1


def box(x0, x1, y0, y1, z0, z1):
    return Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * Box(x1 - x0, y1 - y0, z1 - z0)


def build_cradle(with_bolt_holes=True):
    """The cradle solid.  07_estop_tower_cradle calls this with
    with_bolt_holes=False, so the two builds cannot drift apart."""
    p = box(-PLATE_HW, PLATE_HW, -PLATE_HD, PLATE_HD, 0, PLATE_T)  # base plate
    p += box(-RAIL_OUT, -RAIL_IN, RAIL_Y0, RAIL_Y1, PLATE_T, RAIL_TOP)  # left rail
    p += box(RAIL_IN, RAIL_OUT, RAIL_Y0, RAIL_Y1, PLATE_T, RAIL_TOP)  # right rail
    p += box(-RAIL_IN, -RAIL_IN + LIP_W, LIP_Y0, LIP_Y1, PLATE_T, RAIL_TOP)  # left lip
    p += box(RAIL_IN - LIP_W, RAIL_IN, LIP_Y0, LIP_Y1, PLATE_T, RAIL_TOP)  # right lip
    p += box(-PLATE_HW, PLATE_HW, RAIL_Y0, 0, PLATE_T, BACK_TOP)  # back panel
    p += box(-PLATE_HW, PLATE_HW, 0, LEDGE_Y1, PLATE_T, LEDGE_TOP)  # bottom ledge

    p -= box(-WELL_HW, WELL_HW, WELL_Y0, WELL_Y1, WELL_Z0, WELL_Z1)  # connector well
    p -= box(-GROOVE_HW, GROOVE_HW, LEDGE_Y1, PLATE_HD + 1, GROOVE_Z0, GROOVE_Z1)  # cable groove

    # button relief: slots in the rail inner faces over the phone band only
    # (Y 0..POCKET), so the lips keep their root at Y POCKET..RAIL_Y1 and the
    # back panel keeps its own attachment to the rail
    for sx in (-1, 1):
        lo, hi = sorted((sx * RAIL_IN, sx * (RAIL_IN + BUTTON_RELIEF)))
        p -= box(lo, hi, 0, POCKET, BUTTON_Z, RAIL_TOP + 1)

    if with_bolt_holes:  # to the tower, split build
        for sx in (-1, 1):
            for sy in (-1, 1):
                p -= Pos(sx * BOLT_XY, sy * BOLT_XY, -1) * Cylinder(
                    BOLT_D / 2, PLATE_T + 2, align=(Align.CENTER, Align.CENTER, Align.MIN)
                )
    for sx in (-1, 1):  # rail-top holes, top stop pins
        p -= Pos(sx * PIN_X, PIN_Y, PIN_Z0) * Cylinder(
            PIN_D / 2, 19.0, align=(Align.CENTER, Align.CENTER, Align.MIN)
        )
    return p


part = build_cradle(with_bolt_holes=True)

# report
b = part.bounding_box()
print(f"phone with case : {ENV_W:.2f} W x {ENV_D:.2f} D x {ENV_H:.2f} H")
print(f"channel         : {CHANNEL:.2f}  -> {CHANNEL - ENV_W:.2f} mm nominal slop")
print(
    f"button relief   : +{BUTTON_RELIEF:.2f}/side above Z={BUTTON_Z:.0f}  -> {CHANNEL + 2 * BUTTON_RELIEF:.2f} there"
)
print(f"pocket (Y)      : {POCKET:.2f}  -> {POCKET - ENV_D:.2f} mm over the cased envelope")
print(f"ledge top       : {LEDGE_TOP:.1f}   case top at {LEDGE_TOP + ENV_H:.1f}")
print(f"rail top        : {RAIL_TOP:.1f}   -> {TOP_GAP:.2f} mm top gap for the stop's pad")
print(
    f"back panel top  : {BACK_TOP:.1f}   camera island starts ~"
    f"{LEDGE_TOP + CASE_T + PHONE_H - 44:.1f} -> {LEDGE_TOP + CASE_T + PHONE_H - 44 - BACK_TOP:.1f} mm clear"
)
print(
    f"bounds  x {b.min.X:7.2f} .. {b.max.X:7.2f}   y {b.min.Y:7.2f} .. {b.max.Y:7.2f}"
    f"   z {b.min.Z:7.2f} .. {b.max.Z:7.2f}"
)
print(f"volume  {part.volume / 1000:.1f} cm3   solids {len(part.solids())}")

export_step(part, f"{OUT}/02_iphone_cradle.step")
export_stl(part, f"{OUT}/02_iphone_cradle.stl")
print("exported 02_iphone_cradle.step / .stl")
