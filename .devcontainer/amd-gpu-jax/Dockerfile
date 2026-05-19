FROM rocm/dev-ubuntu-24.04:7.2.1-complete

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git wget zstd libdw1 \
        && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 marks system Python as externally-managed (PEP 668).
RUN rm -f /usr/lib/python*/EXTERNALLY-MANAGED

# Install JAX with ROCm support.  Pin to 0.9.1 because jax-rocm7-plugin
# is currently only released for that version; `jax[rocm]` would pull a
# newer jax that then fails to load the plugin with missing-symbol errors.
RUN pip install --no-cache-dir \
        "jax==0.9.1" "jaxlib==0.9.1" \
        "jax-rocm7-plugin==0.9.1.*" "jax-rocm7-pjrt==0.9.1.*"

# Install our ROCm-aware tokamax fork before `pip install -e .` so the
# alphafold3 dependency resolver picks this up instead of pulling stock
# tokamax==0.0.11 from PyPI. The fork is upstream tokamax v0.0.11 (commit
# b2e34ba) with a single commit on top adding ROCm-awareness to the
# `compute_capability` checks (see `alphafold3-rocm` branch on
# github.com/wilhelmi-lab/tokamax). SHA pinned for reproducibility.
RUN pip install --no-cache-dir \
        "tokamax @ git+https://github.com/wilhelmi-lab/tokamax.git@0c8e896"

ENV PATH="/hmmer/bin:$PATH"

# Download, check hash, and extract the HMMER source code.
RUN mkdir /hmmer_build /hmmer ; \
    wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz --directory-prefix /hmmer_build ; \
    (cd /hmmer_build && echo "ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3 hmmer-3.4.tar.gz" | sha256sum --check) && \
    (cd /hmmer_build && tar zxf hmmer-3.4.tar.gz && rm hmmer-3.4.tar.gz)

# Apply the --seq_limit patch to HMMER.
COPY docker/jackhmmer_seq_limit.patch /hmmer_build/
RUN (cd /hmmer_build && patch -p0 < jackhmmer_seq_limit.patch)

# Build HMMER.
RUN (cd /hmmer_build/hmmer-3.4 && ./configure --prefix /hmmer) && \
    (cd /hmmer_build/hmmer-3.4 && make -j) && \
    (cd /hmmer_build/hmmer-3.4 && make install) && \
    (cd /hmmer_build/hmmer-3.4/easel && make install) && \
    rm -R /hmmer_build

WORKDIR /workspace

# Install multiview-stitcher in editable mode + dev tools
COPY pyproject.toml README.md LICENSE uv.lock fetch_databases.sh CMakeLists.txt run_alphafold.py run_alphafold_test.py run_alphafold_data_test.py OUTPUT_TERMS_OF_USE.md WEIGHTS_TERMS_OF_USE.md WEIGHTS_PROHIBITED_USE_POLICY.md ./
COPY src/ src/
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
RUN pip install --no-cache-dir -e . \
    && pip install --no-cache-dir jupyterlab

# Build chemical components database pickles required at import time.
RUN python3 -c "from alphafold3.build_data import build_data; build_data()"

ENV XLA_PYTHON_CLIENT_PREALLOCATE=true
ENV XLA_CLIENT_MEM_FRACTION=0.95

# ROCm-specific runtime flags (previously passed on the command line):
# - xla_disable_hlo_passes: disable fusion passes that miscompile on ROCm.
# - xla_gpu_enable_triton_gemm: Triton GEMM is NVIDIA-only and crashes on ROCm.
# - HSA_ENABLE_SDMA=0: avoids HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION on
#   AMD GPUs where SDMA and compute see different memory aperture mappings.
ENV XLA_FLAGS="--xla_disable_hlo_passes=fusion,multi-output-fusion --xla_gpu_enable_triton_gemm=false"
ENV HSA_ENABLE_SDMA=0
