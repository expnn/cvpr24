#!/usr/bin/env bash

(return 0 2>/dev/null) && sourced=1 || sourced=0
if [ "$sourced" = "0" ]; then
    echo -e "This script is ment to be sourced. Try\n\n    source $0 $*\n"
    exit 1
fi

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PROJECT_PATH="$SCRIPT_PATH"
echo "PROJECT_PATH = $PROJECT_PATH"

if [ $# -eq 0 ]; then
    echo "no conda environment given";
else
    echo "Activating '$1'..."
    conda activate "$1"
fi

export PYTHONPATH="$PROJECT_PATH":"$PYTHONPATH"
echo "PYTHONPATH = $PYTHONPATH"
