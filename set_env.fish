#!/usr/bin/env fish

if ! string match -q -- "*from sourcing file*" (status)
    echo -e "This script is ment to be sourced. Try\n\n    source "(status filename)" $argv"
    exit 1
end

function show_scalar_variable
    echo $argv[1] = $$argv[1]
    echo
end

function show_list_variable
    echo "$argv[1]:"
    for p in $$argv[1]
        echo "  * $p"
    end
    echo
end

set -l SCRIPT_PATH (realpath (dirname (status -f)))
set -x -g PROJECT_PATH "$SCRIPT_PATH"
show_scalar_variable PROJECT_PATH

if set -q argv[1]
    echo "Activating '$argv[1]'..."
    conda activate "$argv[1]"
end
set -x -g --path --prepend PYTHONPATH "$PROJECT_PATH"
show_list_variable PYTHONPATH
