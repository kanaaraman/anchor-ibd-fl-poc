#!/usr/bin/env bash
# One-command live demo of decentralised (swarm) training across three hospitals.
#
#   bash demo/run_demo.sh            # on a laptop, WSL or one DGX Spark
#   ROUNDS=10 bash demo/run_demo.sh
#
# It provisions real certificates, starts a coordinator and three hospital
# clients as separate processes (each with its own identity and data folder),
# submits the swarm job as the RISE admin, and narrates what happens.
# Needs: Python 3.10+ with torch and nvflare==2.9.0 (pip install -r requirements.txt).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
ROUNDS=${ROUNDS:-5}
W=${DEMO_DIR:-/tmp/anchor_demo}
K=$W/ws/anchor_ibd/prod_00
SITES="rost oru ous"
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

cleanup() {
  for s in $SITES localhost; do [ -d "$K/$s" ] && (echo y | bash "$K/$s/startup/stop_fl.sh" >/dev/null 2>&1 || true); done
  pkill -f "$W/ws" 2>/dev/null || true
}
trap cleanup EXIT

say "1. Each hospital gets its own synthetic OMOP CDM extract (no real patients)"
rm -rf "$W"; mkdir -p "$W/data"
for s in $SITES; do python generate_data.py --out "$W/data" --site "$s" | sed 's/^/   /'; done

say "2. Provisioning: a certificate and private key per participant (mutual TLS)"
sed 's/anchor-fl-server/localhost/' deploy/project.yml > "$W/project.yml"
nvflare provision -p "$W/project.yml" -w "$W/ws" --force >/dev/null 2>&1
ls "$K" | sed 's/^/   kit: /'

say "3. Starting the coordinator (admits sites, starts jobs, holds no data or weights)"
bash "$K/localhost/startup/start.sh" >/dev/null 2>&1
sleep 8

say "4. Starting three hospital clients, each with only its own kit"
for s in $SITES; do bash "$K/$s/startup/start.sh" >/dev/null 2>&1; done
for i in $(seq 1 30); do
  n=$( (grep -l "Successfully registered client" "$K"/{rost,oru,ous}/log.txt 2>/dev/null || true) | wc -l)
  [ "$n" -ge 3 ] && break; sleep 2
done
grep -h "Successfully registered client" "$K"/{rost,oru,ous}/log.txt | sed 's/.*INFO - /   /'

say "5. The RISE admin submits the swarm job ($ROUNDS rounds)"
python nvflare_job/job.py --workflow swarm --env prod --startup_kit "$K/admin@rise.se" \
  --data_root "$W/data" --rounds "$ROUNDS" 2>/dev/null | grep -E "job id" | sed 's/^/   /'

say "6. Training: hospitals train locally, one hospital averages each round"
last=$((ROUNDS - 1)); seen=""
for i in $(seq 1 120); do
  for r in $(seq 0 $last); do
    case " $seen " in *" $r "*) continue ;; esac
    agg=$(grep -rhoE --include=log.txt "round $r to .*aggregation happens on [a-z]+" "$K"/{rost,oru,ous} 2>/dev/null | head -1 | grep -oE "[a-z]+$" || true)
    done_all=0
    for s in $SITES; do grep -rqE --include=log.txt "] round $r:" "$K/$s" 2>/dev/null && done_all=$((done_all + 1)); done
    if [ -n "$agg" ] && [ "$done_all" -ge 3 ]; then
      echo "   round $r: averaged at $agg"
      grep -rhE --include=log.txt "] round $r:" "$K"/{rost,oru,ous} | sed 's/.*\(\[[a-z]*\] round\)/\1/' | sort -u | sed 's/^/      /'
      seen="$seen $r"
    fi
  done
  [ "$(echo $seen | wc -w)" -ge "$ROUNDS" ] && break
  sleep 3
done

say "7. What the coordinator did"
echo "   aggregation steps in the coordinator log: $(grep -ciE 'aggregating [0-9]+ update' "$K/localhost/log.txt" || true)"
echo "   patient records sent: 0 (each client only read its own folder under $W/data)"
say "Done. Open dashboard/index.html for the full comparison with local-only and pooled training."
