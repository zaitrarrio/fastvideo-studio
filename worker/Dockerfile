# syntax=docker/dockerfile:1.7
# fastvideo-studio worker image, the smallest that runs the three engines:
#
#   engines  ubuntu:22.04 + rustup (glibc 2.35, the final stage's) builds `h3`
#            (h3-rs) and `fastwan` (fastwan-rs) with `--features cuda-13`:
#            cudarc binds CUDA 13 and loads the libraries at run time, so the
#            three binaries share one runtime (no second 12.8 set, no wheels).
#   fv       the fastvideo-rs serve image, used only as a file source: fv-serve,
#            its pinned CUDA 13.4 cuBLAS / cuBLASLt / NVRTC / cudart, cuDNN 9
#            (fv-serve's SDPA) and its ffmpeg / ffprobe (dynamically linked
#            against Ubuntu 22.04's libx264 / libvpx / libdav1d / libssl).
#   studio   the same ubuntu:22.04 + rustup builder: `studio-worker`, the Rust
#            worker (worker-rs/: RunPod protocol, R2 SigV4, the catalog, the
#            sidecars; rustls, no OpenSSL): a binary that needs only libc in
#            the final image.
#   final    ubuntu:22.04 (no Python: the worker is the `studio-worker` binary)
#            + the four codec libraries + the files above, and llama-server
#            (llama.cpp, CUDA 13, +82 MB) for prompt expansion with
#            Qwen3-VL-4B-Instruct fetched from R2 on first use. Not copied:
#            fv-boot, perl, apt lists, the Python reference tree (worker/).
#
#   docker build -f worker/Dockerfile --build-arg H3_REF=<sha> --build-arg FASTWAN_REF=<sha> -t fastvideo-studio-worker .

ARG FV_SERVE_IMAGE=ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:aa2a46574879f51544bad5da6d34cf643835d25e1cc6b09417907581ac3fc699
ARG BUILDER_IMAGE=ubuntu:22.04
ARG RUST_VERSION=1.97.0

# ---- stage 0: the Rust builder ------------------------------------------------
# The same Ubuntu 22.04 as the final stage (glibc 2.35), with rustup: the
# rust:*-bullseye images' apt archives have gone stale (bullseye is past its
# end of life) and bookworm's glibc 2.36 is newer than the final stage's.
FROM ${BUILDER_IMAGE} AS rust-builder
ARG RUST_VERSION
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git build-essential pkg-config \
 && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}"
ENV PATH=/root/.cargo/bin:${PATH}

# ---- stage 1: the two engine binaries, CUDA 13 bindings ----------------------
FROM rust-builder AS engines
ARG H3_REPO=https://github.com/zaitrarrio/h3-rs.git
ARG FASTWAN_REPO=https://github.com/zaitrarrio/fastwan-rs.git
# Pinned commits of branch claude/wizardly-clarke-1a2jta: the `cuda-13` feature
# (h3-rs PR #6, fastwan-rs PR #7) on top of serve's per-request inputs.
ARG H3_REF=697fb9a9289f84ae6c6120d87ff284edcb9effdf
ARG FASTWAN_REF=b50c8420b0a6fdcb017a52398cfc3894b9956019
WORKDIR /src
RUN --mount=type=secret,id=repo_token,required=false \
    set -eu; \
    auth() { if [ -s /run/secrets/repo_token ]; then echo "https://x-access-token:$(cat /run/secrets/repo_token)@github.com/"; else echo "https://github.com/"; fi; }; \
    fetch() { git init -q "$2" && git -C "$2" remote add origin "$1" && git -C "$2" fetch -q --depth 1 origin "$3" && git -C "$2" checkout -q FETCH_HEAD; }; \
    H3="$(echo "$H3_REPO" | sed "s#https://github.com/#$(auth)#")"; \
    FW="$(echo "$FASTWAN_REPO" | sed "s#https://github.com/#$(auth)#")"; \
    fetch "$H3" h3-rs "$H3_REF"; fetch "$FW" fastwan-rs "$FASTWAN_REF"; \
    git -C h3-rs remote set-url origin "$H3_REPO"; git -C fastwan-rs remote set-url origin "$FASTWAN_REPO"; \
    git -C h3-rs rev-parse HEAD > /src/h3.sha; git -C fastwan-rs rev-parse HEAD > /src/fastwan.sha
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/src/h3-rs/target \
    cd h3-rs && cargo build --release -p h3-cli --no-default-features --features cuda-13 \
 && mkdir -p /out && cp target/release/h3 /out/h3
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/src/fastwan-rs/target \
    cd fastwan-rs && cargo build --release -p fastwan-cli --features cuda-13 \
 && cp target/release/fastwan /out/fastwan
RUN strip /out/h3 /out/fastwan && /out/h3 --help >/dev/null && /out/fastwan --help >/dev/null && ls -l /out

# ---- stage 1b: the worker binary (worker-rs/, the Rust port of handler.py) -----
FROM rust-builder AS studio
WORKDIR /src/worker-rs
COPY worker-rs/Cargo.toml worker-rs/Cargo.lock ./
COPY worker-rs/src ./src
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/src/worker-rs/target \
    cargo build --release --locked \
 && mkdir -p /out && cp target/release/studio-worker /out/studio-worker \
 && strip /out/studio-worker && /out/studio-worker --help >/dev/null && ls -l /out

# ---- stage 2: the file source --------------------------------------------------
FROM ${FV_SERVE_IMAGE} AS fv
# cuDNN's libraries are symlink chains (libcudnn_graph.so.9 → .so.9.26.0); a
# glob COPY resolves every name into a full copy (the first build doubled
# them to an 836 MB layer), so gather them with cp -a into one directory the
# final stage copies as a directory, which keeps the links.
RUN mkdir -p /cudnn && cp -a /usr/lib/x86_64-linux-gnu/libcudnn*.so* /cudnn/

# ---- stage 3: the worker -------------------------------------------------------
FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates libx264-163 libvpx7 libdav1d5 libssl3 zlib1g \
 && rm -rf /var/lib/apt/lists/* /usr/share/doc /usr/share/man
# CUDA 13.4 (cuBLAS, cuBLASLt, NVRTC + builtins, cudart) and cuDNN 9 from the
# serve image; libcuda comes from the host driver at run time (R580+).
COPY --from=fv /usr/local/cuda-13.4/lib64/ /usr/local/cuda-13.4/lib64/
COPY --from=fv /cudnn/ /usr/lib/x86_64-linux-gnu/
COPY --from=fv /opt/fastvideo-rs/bin/fv-serve /opt/fastvideo-rs/bin/llama-server /opt/fastvideo-rs/bin/llama-server.version /opt/fastvideo-rs/bin/
COPY --from=fv /usr/local/bin/ffmpeg /usr/local/bin/ffprobe /usr/local/bin/
RUN echo /usr/local/cuda-13.4/lib64 > /etc/ld.so.conf.d/cuda.conf && ldconfig \
 && for f in /opt/fastvideo-rs/bin/fv-serve /opt/fastvideo-rs/bin/llama-server /usr/local/bin/ffmpeg /usr/local/bin/ffprobe \
             /usr/local/cuda-13.4/lib64/libcublasLt.so.13 /usr/local/cuda-13.4/lib64/libcublas.so.13 \
             /usr/local/cuda-13.4/lib64/libnvrtc.so.13 /usr/lib/x86_64-linux-gnu/libcudnn.so.9; do \
      if ldd "$f" | grep 'not found' | grep -v libcuda.so; then echo "unresolved: $f"; exit 1; fi; done \
 && ffmpeg -hide_banner -encoders | grep -q ' libx264 ' && ffmpeg -hide_banner -encoders | grep -q ' aac '
COPY --from=engines /out/h3 /out/fastwan /opt/studio/bin/
COPY --from=engines /src/h3.sha /src/fastwan.sha /opt/studio/
RUN for f in /opt/studio/bin/h3 /opt/studio/bin/fastwan; do \
      if ldd "$f" | grep 'not found'; then echo "unresolved: $f"; exit 1; fi; "$f" --help >/dev/null; done
COPY catalog/catalog.json /opt/studio/catalog/catalog.json
COPY worker/configs/fastvideo.toml /opt/studio/configs/fastvideo.toml
COPY --from=studio /out/studio-worker /opt/studio/bin/studio-worker
# The worker loads and checks the catalog at start: a local job with an unknown
# op exits 1 with "unknown op" once the catalog parsed (2 when it did not).
RUN if ldd /opt/studio/bin/studio-worker | grep 'not found'; then echo "unresolved: studio-worker"; exit 1; fi \
 && echo '{"input": {"op": "catalog-check"}}' > /tmp/catalog-check.json \
 && (STUDIO_CATALOG=/opt/studio/catalog/catalog.json /opt/studio/bin/studio-worker --test-input /tmp/catalog-check.json || true) | grep -q 'unknown op' \
 && rm /tmp/catalog-check.json \
 && test -x /opt/fastvideo-rs/bin/fv-serve \
 && mkdir -p /workspace/w /workspace/stage /workspace/jobs /workspace/fv-state
ARG STUDIO_IMAGE=ghcr.io/zaitrarrio/fastvideo-studio-worker:dev
ENV STUDIO_IMAGE=${STUDIO_IMAGE} \
    STUDIO_CATALOG=/opt/studio/catalog/catalog.json \
    STUDIO_BIN_DIR=/opt/studio/bin \
    STUDIO_FV_SERVE=/opt/fastvideo-rs/bin/fv-serve \
    STUDIO_FV_CONFIG=/opt/studio/configs/fastvideo.toml \
    STUDIO_LLAMA_SERVER=/opt/fastvideo-rs/bin/llama-server \
    STUDIO_CUDA12_LIB= \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
WORKDIR /opt/studio
ENTRYPOINT ["/opt/studio/bin/studio-worker"]
