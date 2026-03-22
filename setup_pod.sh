#!/bin/bash
set -e

cd /workspace

if [ -d "parameter-golf" ]; then
  cd parameter-golf
  git pull
else
  git clone https://github.com/Theocat321/parameter-golf.git
  cd parameter-golf
fi

pip install huggingface-hub
python data/cached_challenge_fineweb.py

echo "Setup complete. Run: bash /workspace/parameter-golf/run.sh"
