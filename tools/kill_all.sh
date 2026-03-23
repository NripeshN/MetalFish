#!/bin/bash
# Kill all engine processes on all EC2 instances + local
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
PEM="$PROJ/m1 ultra.pem"
HOSTS=(18.212.251.224 54.91.84.176 54.163.13.25 107.21.161.159)

echo "Killing local..."
tmux kill-session -t metalfish-tournament 2>/dev/null || true
pkill -f "cutechess-cli" 2>/dev/null || true

echo "Killing remote..."
for HOST in "${HOSTS[@]}"; do
    ssh -i "$PEM" -o StrictHostKeyChecking=no -o ConnectTimeout=5 ec2-user@$HOST \
        "killall -9 metalfish stockfish berserk patricia lc0 cutechess-cli 2>/dev/null; echo $HOST:clean" &
done
wait
echo "All clean."
