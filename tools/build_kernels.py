#!/usr/bin/env python3
"""Embed luce-render's GLSL kernels into src/render/kernels.lucb.

Every kernel in KERNELS is compiled through luce-gpu's tools/embed_shaders.py
(SPIR-V for Vulkan, Metal through spirv-cross); tracing kernels are embedded
twice, spectral (SPECTRAL=1) and RGB (SPECTRAL=0). Run it after changing
anything under shaders/; builds use the checked-in module.

    python3 tools/build_kernels.py
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
EMBED = ROOT.parent / 'luce-gpu' / 'tools' / 'embed_shaders.py'
SHADERS = ROOT / 'shaders'
OUTPUT = ROOT / 'src' / 'render' / 'kernels.lucb'

# (source, stem, defines): kernels as the host creates them.
KERNELS = [('probe.comp', 'probe', ''), ('tables.comp', 'tables', '')]
# The integrator, spectral and RGB.
for name in ['camera', 'schedule', 'intersect_closest', 'shade_surface', 'shade_miss', 'intersect_shadow', 'finish', 'film_convert']:
    KERNELS.append((f'{name}.comp', f'{name}', 'SPECTRAL=1'))
    KERNELS.append((f'{name}.comp', f'{name}_rgb', 'SPECTRAL=0'))

def main() -> int:
    specs = []
    for source, stem, defines in KERNELS:
        spec = f'{SHADERS / source}:{stem}'
        if defines:
            spec += ':' + defines
        specs.append(spec)
    command = [sys.executable, str(EMBED), str(OUTPUT), '--public', '--shared', '-I', str(SHADERS)] + specs
    return subprocess.call(command)

if __name__ == '__main__':
    sys.exit(main())
