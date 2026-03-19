#!/bin/bash
# Kill all engine processes on all EC2 instances + local
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
PEM="$PROJ/m1 ultra.pem"
HOSTS=(44.220.150.2 98.81.229.157 98.84.106.208 32.192.83.249)

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
