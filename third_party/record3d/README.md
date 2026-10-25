# Record3D USB latency patch

The base publisher requires `latest_frame_only` and `skipped_frames`. The stock
Record3D Python package does not provide these properties. `latest-frame-only.patch`
contains the deployed changes to the C++ receiver, header, and Python bindings.
It drops queued frames before decoding, bounds the drain loop, and stops on a stalled
partial read. The default remains unchanged until the publisher enables the option.

Build on the Linux host with the USB-connected iPhone, using a dedicated Python
3.10 or 3.12 environment. Set `OMNITELEOP_ROOT` to this public checkout:

```bash
export OMNITELEOP_ROOT=/absolute/path/to/omniteleop
sudo apt-get install build-essential libusbmuxd-dev
python -m pip install 'numpy<2' 'setuptools<70' wheel 'cmake<4' 'dexcomm>=0.6,<0.7'
git clone https://github.com/marek-simonik/record3d.git record3d-source
git -C record3d-source checkout 79e6f44697150a4f036034af8fec903d11ac6f00
git -C record3d-source apply --check "$OMNITELEOP_ROOT/third_party/record3d/latest-frame-only.patch"
git -C record3d-source apply "$OMNITELEOP_ROOT/third_party/record3d/latest-frame-only.patch"
python -m pip install --no-build-isolation ./record3d-source
python -c 'from record3d import Record3DStream; s=Record3DStream(); s.latest_frame_only=True; print(s.latest_frame_only, s.skipped_frames)'
```

The final command must print `True 0`; it does not connect to the phone. Do not
replace an existing working build just to run it again. The patch was checked
against the pinned source and the deployed robot build exposes both properties.
The new upstream build still needs validation on the target host.

Upstream: [marek-simonik/record3d](https://github.com/marek-simonik/record3d),
revision `79e6f44697150a4f036034af8fec903d11ac6f00`, LGPL-2.1. The license is
included in `LICENSE`; the modified source is distributed here as a patch.
