#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ai-toolkit
exec .venv/bin/python -u run.py config/qwen_image_21_kibm.yaml
