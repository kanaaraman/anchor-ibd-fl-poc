# One image for the server and every hospital client.
#
# On a DGX Spark (arm64, GB10), use NVIDIA's PyTorch container, which ships a
# CUDA build of torch for Blackwell. Check the NGC catalog for the newest tag
# that lists DGX Spark support and pin it here.
#   docker build -f deploy/Dockerfile -t anchor-ibd-fl .
#
# For a CPU-only smoke test on a laptop:
#   docker build -f deploy/Dockerfile --build-arg BASE_IMAGE=python:3.12-slim --build-arg INSTALL_TORCH=1 -t anchor-ibd-fl .
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.09-py3
FROM ${BASE_IMAGE}

ARG INSTALL_TORCH=0
RUN if [ "$INSTALL_TORCH" = "1" ]; then \
      pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    fi \
 && pip install --no-cache-dir nvflare==2.9.0 numpy

WORKDIR /opt/anchor
COPY ibdfl ./ibdfl
COPY generate_data.py ./
ENV PYTHONPATH=/opt/anchor

# The startup kit is mounted at /workspace/kit; sub_start.sh runs in the foreground.
WORKDIR /workspace
CMD ["bash", "/workspace/kit/startup/sub_start.sh"]
