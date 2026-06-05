#!/usr/bin/env bash
set -euo pipefail

mkdir -p third_party
git clone https://github.com/ZBang/USEF-TSE.git third_party/USEF-TSE || true
git clone https://github.com/JorisCos/LibriMix.git third_party/LibriMix || true
git clone --depth 1 https://github.com/espnet/espnet.git third_party/espnet || true
git clone --depth 1 https://github.com/asteroid-team/asteroid.git third_party/asteroid || true

