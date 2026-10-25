"""Fit-test coupon — the bottom 50 mm of 02_iphone_cradle, nothing else.

Print this first.  It checks the three dimensions that decide whether the cased
phone seats properly — channel width, lip pocket depth, and the ledge — for
about a fifth of the material and an hour of machine time.  If the case slides
in with a light rub and no side-to-side shake, print the real part.
"""

import os
import runpy

from build123d import Box, Pos, export_step, export_stl

OUT = os.path.dirname(os.path.abspath(__file__))  # this folder
ns = runpy.run_path(f"{OUT}/02_iphone_cradle.py", run_name="_cradle")
part = ns["part"]

CUT = 50.0
part -= Pos(0, 0, CUT + 200) * Box(400, 400, 400)  # keep only the bottom 50 mm
# the base plate only has to tie the two rails together here, so trim it to the
# band that actually carries the back panel, ledge and lips
part -= Pos(0, -10 - 200, 0) * Box(400, 400, 400)
part -= Pos(0, 20 + 200, 0) * Box(400, 400, 400)

b = part.bounding_box()
print(
    f"\ncoupon bounds  x {b.min.X:7.2f} .. {b.max.X:7.2f}   y {b.min.Y:7.2f} .. {b.max.Y:7.2f}"
    f"   z {b.min.Z:7.2f} .. {b.max.Z:7.2f}"
)
print(f"coupon volume  {part.volume / 1000:.1f} cm3")
export_step(part, f"{OUT}/08_fit_test_coupon.step")
export_stl(part, f"{OUT}/08_fit_test_coupon.stl")
print("exported 08_fit_test_coupon.step / .stl")
