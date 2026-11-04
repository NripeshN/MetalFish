#!/bin/bash
# Set-up steps of the paper's runner jobs (.github/workflows/paper-runner-cells.yml, protocol
# amendment 15). Each subcommand fetches one input into the place the workstation tools expect
# and checks it against the SHA-256 in the workflow's environment.
#
#   runner_setup.sh fastchess        reference/fastchess/fastchess (the workstation's release)
#   runner_setup.sh book             reference/books/unzipped/UHO_Lichess_4852_v1.epd
#   runner_setup.sh syzygy           syzygy/ (3-4-5 men, 290 files, checked against the mirror)
#   runner_setup.sh runner OUT.json  what this runner is (image, hardware, Metal device)
#   runner_setup.sh sampler OUT.log  one sample a minute of the VM's busiest processes, memory,
#                                    swap and GPU utilisation, until killed (health only)
set -euo pipefail
cd "$(dirname "$0")/../.."

case "${1:-}" in
  fastchess)
    tmp=$(mktemp -d)
    gh release download "$FASTCHESS_RELEASE" -R Disservin/fastchess -p fastchess-mac-arm64.tar -D "$tmp"
    tar -xf "$tmp/fastchess-mac-arm64.tar" -C "$tmp"
    mkdir -p reference/fastchess
    find "$tmp" -name fastchess -type f -perm -u+x -exec cp {} reference/fastchess/fastchess \;
    echo "$FASTCHESS_SHA256  reference/fastchess/fastchess" | shasum -a 256 -c
    ;;
  book)
    tmp=$(mktemp -d)
    curl -fsSL --retry 3 -o "$tmp/book.zip" "$BOOK_URL"
    mkdir -p reference/books/unzipped
    unzip -o -q "$tmp/book.zip" -d reference/books/unzipped
    echo "$BOOK_SHA256  reference/books/unzipped/UHO_Lichess_4852_v1.epd" | shasum -a 256 -c
    ;;
  syzygy)
    # the lichess mirror's 3-4-5 tables; every file is checked against the mirror's SHA-256 list,
    # which the workstation's 290 files also match (2026-11-04), so both machines probe the same
    base=https://tablebase.lichess.ovh/tables/standard
    tmp=$(mktemp -d)
    mkdir -p syzygy
    curl -fsSL --retry 3 --max-time 120 -o "$tmp/sha256" "$base/sha256"
    : > "$tmp/urls"
    for d in 3-4-5-wdl 3-4-5-dtz; do
      curl -fsSL --retry 3 --max-time 120 "$base/$d/" | grep -oE 'href="[^"/]+\.rtb[wz]"' | cut -d'"' -f2 | sort -u > "$tmp/$d"
      while read -r f; do [ -s "syzygy/$f" ] || echo "$base/$d/$f"; done < "$tmp/$d" >> "$tmp/urls"
    done
    if [ -s "$tmp/urls" ]; then
      (cd syzygy && xargs -P 6 -n 1 curl -fsSL --retry 5 --retry-all-errors --connect-timeout 20 --max-time 900 -O < "$tmp/urls")
    fi
    cat "$tmp/3-4-5-wdl" "$tmp/3-4-5-dtz" | while read -r f; do grep -E "  $f\$" "$tmp/sha256"; done > "$tmp/want"
    [ "$(wc -l < "$tmp/want")" -eq 290 ] || { echo "syzygy: $(wc -l < "$tmp/want") of 290 checksums"; exit 1; }
    (cd syzygy && shasum -a 256 -c --quiet "$tmp/want")
    echo "syzygy: 290 files, $(du -sh syzygy | cut -f1), SHA-256 as on the mirror's list"
    ;;
  runner)
    python3 - "$2" <<'EOF'
import json, os, platform, subprocess, sys

def cap(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return repr(exc)

info = {
    "image": f"{os.environ.get('ImageOS', '')} {os.environ.get('ImageVersion', '')}".strip(),
    "runner_name": os.environ.get("RUNNER_NAME"),
    "run": f"{os.environ.get('GITHUB_RUN_ID')}/{os.environ.get('GITHUB_RUN_ATTEMPT')}",
    "job": os.environ.get("GITHUB_JOB"),
    "cpu": cap("sysctl", "-n", "machdep.cpu.brand_string"),
    "model": cap("sysctl", "-n", "hw.model"),
    "ncpu": cap("sysctl", "-n", "hw.ncpu"),
    "mem_bytes": cap("sysctl", "-n", "hw.memsize"),
    "os": platform.platform(),
    "sw_vers": cap("sw_vers", "-buildVersion"),
    "xcode": cap("xcodebuild", "-version").replace("\n", " "),
    "gpu": [l.strip() for l in cap("system_profiler", "SPDisplaysDataType").splitlines()
            if l.strip().startswith(("Chipset Model", "Total Number of Cores", "Metal"))],
}
json.dump(info, open(sys.argv[1], "w"), indent=2)
print(json.dumps(info, indent=2))
EOF
    ;;
  sampler)
    out="$2"
    while true; do
      {
        echo "== $(date -u +%FT%TZ)"
        top -l 2 -s 1 -n 6 -o cpu -stats pid,command,cpu,mem | awk '/^PID/{n++} n==2'
        vm_stat | grep -E "Pages free|Pageouts|Swapouts" || true
        sysctl -n vm.swapusage
        ioreg -r -d 1 -w 0 -c IOAccelerator | grep -oE '"Device Utilization %"=[0-9]+' | head -1 || true
      } >> "$out" 2>&1
      sleep 58
    done
    ;;
  *)
    sed -n '2,13p' "$0"
    exit 2
    ;;
esac
