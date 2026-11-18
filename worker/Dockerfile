# syntax=docker/dockerfile:1.7
# fastvideo-studio worker image (docs/design.md §1):
#   stage 1: build `h3` (h3-rs) and `fastwan` (fastwan-rs) with --features cuda
#            (cudarc dynamic loading: no CUDA toolkit at build time);
#   stage 2: the fastvideo-rs serve image (fv-serve, fv-boot, CUDA 13.4 libs,
#            pinned to :cuda-latest's amd64 digest of 2026-11-18, since
#            :cuda-stable does not exist until fastvideo-rs promotes a release;
#            ffmpeg with libx264 + aac at /usr/local/bin) plus a venv with the
#            runpod SDK, boto3 and the CUDA 12.8 runtime wheels whose libraries
#            are linked into /opt/cuda12/lib (LD_LIBRARY_PATH for h3 / fastwan
#            only; fv-serve keeps the image's 13.4).
#
#   docker build -f worker/Dockerfile \
#     --build-arg H3_REF=<sha> --build-arg FASTWAN_REF=<sha> \
#     --secret id=repo_token,env=REPO_READ_TOKEN -t fastvideo-studio-worker .

ARG FV_SERVE_IMAGE=ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:aa2a46574879f51544bad5da6d34cf643835d25e1cc6b09417907581ac3fc699
ARG RUST_IMAGE=rust:1.97-bookworm

# ---- stage 1: the two engine binaries ---------------------------------------
FROM ${RUST_IMAGE} AS engines
ARG H3_REPO=https://github.com/zaitrarrio/h3-rs.git
ARG FASTWAN_REPO=https://github.com/zaitrarrio/fastwan-rs.git
# Pinned commits of branch claude/wizardly-clarke-1a2jta (h3-rs: serve takes keyframe / reference inputs per request).
ARG H3_REF=e029781510585f8fc5b88cae71bcd8a6cddd26fc
ARG FASTWAN_REF=74c27bb1de3b0862175889caab6da10704fedbe9
RUN apt-get update && apt-get install -y --no-install-recommends git pkg-config libssl-dev clang \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
# A read token (private repositories) comes in as a build secret, never a layer.
RUN --mount=type=secret,id=repo_token,required=false \
    set -eu; \
    auth() { if [ -s /run/secrets/repo_token ]; then echo "https://x-access-token:$(cat /run/secrets/repo_token)@github.com/"; else echo "https://github.com/"; fi; }; \
    fetch() { git init -q "$2" && git -C "$2" remote add origin "$1" && git -C "$2" fetch -q --depth 1 origin "$3" && git -C "$2" checkout -q FETCH_HEAD; }; \
    H3="$(echo "$H3_REPO" | sed "s#https://github.com/#$(auth)#")"; \
    FW="$(echo "$FASTWAN_REPO" | sed "s#https://github.com/#$(auth)#")"; \
    fetch "$H3" h3-rs "$H3_REF"; fetch "$FW" fastwan-rs "$FASTWAN_REF"; \
    git -C h3-rs remote set-url origin "$H3_REPO"; git -C fastwan-rs remote set-url origin "$FASTWAN_REPO"; \
    git -C h3-rs rev-parse HEAD > /src/h3.sha; git -C fastwan-rs rev-parse HEAD > /src/fastwan.sha
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/h3-rs/target \
    cd h3-rs && cargo build --release -p h3-cli --no-default-features --features cuda \
 && mkdir -p /out && cp target/release/h3 /out/h3
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/fastwan-rs/target \
    cd fastwan-rs && cargo build --release -p fastwan-cli --features cuda \
 && cp target/release/fastwan /out/fastwan
RUN /out/h3 --help >/dev/null && /out/fastwan --help >/dev/null && ls -l /out

# ---- stage 2: the worker ------------------------------------------------------
FROM ${FV_SERVE_IMAGE}
ARG DEBIAN_FRONTEND=noninteractive
# python3 (3.10 on Ubuntu 22.04) is in the serve image; add pip / venv. ffmpeg
# (static, libx264 + aac + nvenc) is the image's at /usr/local/bin.
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3-pip python3-venv \
 && rm -rf /var/lib/apt/lists/*
COPY worker/requirements.txt /opt/studio/requirements.txt
RUN python3 -m venv /opt/studio/venv \
 && /opt/studio/venv/bin/pip install --no-cache-dir -U pip \
 && /opt/studio/venv/bin/pip install --no-cache-dir -r /opt/studio/requirements.txt \
      "nvidia-cublas-cu12==12.8.*" "nvidia-cuda-nvrtc-cu12==12.8.*" "nvidia-nvjitlink-cu12==12.8.*" \
 && mkdir -p /opt/cuda12/lib \
 && for d in cublas cuda_nvrtc nvjitlink; do \
      for f in /opt/studio/venv/lib/python3*/site-packages/nvidia/$d/lib/*.so*; do ln -sf "$f" /opt/cuda12/lib/; done; done \
 && cd /opt/cuda12/lib \
 # The soname links the wheels leave out: libnvrtc ships libnvrtc.so.12 itself (a
 # link to that name onto itself was a loop), libcublas ships libcublas.so.12.x.y;
 # every name points at the real file.
 && for n in libcublas libcublasLt libnvrtc libnvJitLink; do \
      real="$(ls -1 $n.so.12* 2>/dev/null | sort -V | tail -1)"; [ -n "$real" ] || { echo "missing $n"; exit 1; }; \
      target="$(readlink -f "$real")"; \
      [ -e "$n.so.12" ] || ln -s "$target" "$n.so.12"; ln -sfn "$target" "$n.so"; \
    done \
 && ls -l /opt/cuda12/lib \
 && test -e /opt/cuda12/lib/libcublasLt.so.12 && test -e /opt/cuda12/lib/libnvrtc.so.12 \
 # every 12.8 library resolves against the others (libcuda comes from the host driver at run time)
 && for f in /opt/cuda12/lib/libcublasLt.so.12 /opt/cuda12/lib/libcublas.so.12 /opt/cuda12/lib/libnvrtc.so.12; do \
      if LD_LIBRARY_PATH=/opt/cuda12/lib ldd "$f" | grep 'not found' | grep -v libcuda.so; then echo "unresolved: $f"; exit 1; fi; done
COPY --from=engines /out/h3 /out/fastwan /opt/studio/bin/
COPY --from=engines /src/h3.sha /src/fastwan.sha /opt/studio/
COPY catalog/catalog.json /opt/studio/catalog/catalog.json
COPY worker/configs/fastvideo.toml /opt/studio/configs/fastvideo.toml
COPY worker/handler.py /opt/studio/handler.py
COPY worker/fvstudio /opt/studio/fvstudio
RUN /opt/studio/venv/bin/python -c "import runpod, boto3; print('runpod', runpod.__version__, 'boto3', boto3.__version__)" \
 && /opt/studio/venv/bin/python -I -c "import sys; sys.path.insert(0, '/opt/studio'); import fvstudio.catalog as c; print(len(c.Catalog.load('/opt/studio/catalog/catalog.json').models), 'catalog models')" \
 && ffmpeg -hide_banner -encoders | grep -q ' libx264 ' && ffmpeg -hide_banner -encoders | grep -q ' aac ' \
 && test -x /opt/fastvideo-rs/bin/fv-serve \
 && mkdir -p /workspace/w /workspace/stage /workspace/jobs /workspace/fv-state
ARG STUDIO_IMAGE=ghcr.io/zaitrarrio/fastvideo-studio-worker:dev
ENV STUDIO_IMAGE=${STUDIO_IMAGE} \
    STUDIO_CATALOG=/opt/studio/catalog/catalog.json \
    STUDIO_BIN_DIR=/opt/studio/bin \
    STUDIO_FV_SERVE=/opt/fastvideo-rs/bin/fv-serve \
    STUDIO_FV_CONFIG=/opt/studio/configs/fastvideo.toml \
    STUDIO_CUDA12_LIB=/opt/cuda12/lib \
    PYTHONUNBUFFERED=1
WORKDIR /opt/studio
ENTRYPOINT ["/opt/studio/venv/bin/python", "/opt/studio/handler.py"]
