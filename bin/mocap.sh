#!/bin/sh

CURRENT_DIR="$(dirname "$0")"
CURRENT_DIR=$(realpath "$CURRENT_DIR/..")

# Parse options
USE_NOHUP=0
while getopts "n" option; do
  case $option in
    n) USE_NOHUP=1 ;;
    *) echo "Usage: $0 [-n]" >&2; exit 1 ;;
  esac
done

# I/O processing
model=modeplait
uuid=1
input_dir=mocap

COMMAND="poetry run python src/main.py --multirun \
  model=${model} \
  io.input_dir=${input_dir} \
  io.uuid=${uuid} \
  io.root_out_dir=out/ \
  model.h=30 \
  model.err_th=0.4 \
  model.lcurr=50 \
  model.lstep=5"

# judge whether to use nohup
if [ $USE_NOHUP -eq 1 ]; then
  nohup sh -c "$COMMAND" > nohup/${model}_${input_dir}_${uuid}.out 2>&1 &
else
  sh -c "$COMMAND"
fi
