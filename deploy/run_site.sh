#!/usr/bin/env bash
# Step 2 of the rollout: one DGX Spark per hospital, plus a server host.
#
# On the server host:   deploy/run_site.sh server <kit dir>
# On each hospital:     deploy/run_site.sh client <kit dir> <local OMOP data dir> <server IP>
#
# The client container gets the GPU, its own kit and read-only access to its
# own data. Only outbound gRPC/TLS to the server (port 8002) is needed; no
# inbound ports are opened at the hospital.
set -euo pipefail
role=$1 kit=$(realpath "$2")
site=$(basename "$kit")
case $role in
  server)
    docker run -d --name anchor-fl-server --restart unless-stopped \
      -p 8002:8002 -p 8003:8003 -v "$kit:/workspace/kit" anchor-ibd-fl ;;
  client)
    data=$(realpath "$3") server_ip=$4
    docker run -d --name "anchor-fl-$site" --restart unless-stopped --gpus all \
      --add-host "anchor-fl-server:$server_ip" \
      -v "$kit:/workspace/kit" -v "$data:/data/$site:ro" anchor-ibd-fl ;;
  *) echo "usage: $0 server|client <kit> [data] [server ip]"; exit 1 ;;
esac
docker logs -f "$(docker ps -lq)"
