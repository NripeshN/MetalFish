#!/bin/bash
DIR="$(cd "$(dirname "$0")/.." && pwd)"
"$DIR/build/metalfish" "$@" 2>/tmp/hybrid_stderr.log
EC=$?
echo "EXIT_CODE=$EC" >> /tmp/hybrid_stderr.log
