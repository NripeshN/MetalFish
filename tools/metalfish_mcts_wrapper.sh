#!/bin/bash
# MetalFish MCTS wrapper - intercepts 'go' and runs 'mcts' instead

ENGINE="/Users/nripeshn/Documents/PythonPrograms/chess_with_gpu/metalfish/build/metalfish"

# Read UCI commands and transform 'go' to 'mcts'
while IFS= read -r line; do
    if [[ "$line" == go* ]]; then
        # Replace 'go' with 'mcts'
        echo "mcts ${line#go}"
    else
        echo "$line"
    fi
done | "$ENGINE"
