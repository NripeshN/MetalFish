#!/bin/bash
# MetalFish Hybrid wrapper - intercepts 'go' and runs 'parallel_hybrid' (parallel MCTS+AB) instead

ENGINE="/Users/nripeshn/Documents/PythonPrograms/chess_with_gpu/metalfish/build/metalfish"

# Read UCI commands and transform 'go' to 'parallel_hybrid'
while IFS= read -r line; do
    if [[ "$line" == go* ]]; then
        # Replace 'go' with 'parallel_hybrid' for parallel hybrid search
        echo "parallel_hybrid ${line#go}"
    else
        echo "$line"
    fi
done | "$ENGINE"
