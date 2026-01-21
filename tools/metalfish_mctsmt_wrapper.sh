#!/bin/bash
# MetalFish MCTS wrapper - intercepts 'go' and runs 'mctsmt' (GPU MCTS) instead

ENGINE="/Users/nripeshn/Documents/PythonPrograms/chess_with_gpu/metalfish/build/metalfish"

# Read UCI commands and transform 'go' to 'mctsmt threads=4'
while IFS= read -r line; do
    if [[ "$line" == go* ]]; then
        # Replace 'go' with 'mctsmt threads=4' for multi-threaded GPU MCTS
        echo "mctsmt threads=4 ${line#go}"
    else
        echo "$line"
    fi
done | "$ENGINE"
