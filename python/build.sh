#!/bin/bash
set -e
cd "$(dirname "$0")"
rm -rf build/ dist/ *.egg-info/
uv pip install --reinstall --no-build-isolation .
