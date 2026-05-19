# Running AlphaFold 3 with Apptainer (AMD GPU / ROCm)

This is the Apptainer + ROCm analogue of the Docker-based instructions in
[`README.md`](README.md) and
[`docs/installation.md`](docs/installation.md). The upstream instructions
target NVIDIA GPUs and Docker; this document covers the AMD GPU + Apptainer
path used for deployment on HPC systems or hosts where Docker is unavailable
at prediction time.

## Prerequisites

- Linux host with an AMD GPU (verified on MI210 with ROCm 7.2)
- `amdkfd` kernel driver loaded (`/dev/kfd` and `/dev/dri` present)
- [Apptainer](https://apptainer.org/) 1.3 or newer (`apptainer --version`)
- Docker on the **build host** — the container is built as a Docker image
  first and then converted to a SIF. Docker is not required at prediction
  time.

## Building the Container

The ROCm-targeted Dockerfile lives at
[`.devcontainer/amd-gpu-jax/Dockerfile`](.devcontainer/amd-gpu-jax/Dockerfile)
and bakes in all ROCm-specific environment variables. Build the Docker
image, then convert it to a SIF:

```sh
# From the repo root:
docker build \
    -f .devcontainer/amd-gpu-jax/Dockerfile \
    -t alphafold3-rocm:latest .

# The Docker image is ~26 GB, which is more than `/tmp` usually has free.
# Point apptainer's temp + cache dirs at somewhere roomier before the build.
mkdir -p ~/apptainer_tmp
APPTAINER_TMPDIR=$HOME/apptainer_tmp \
APPTAINER_CACHEDIR=$HOME/apptainer_tmp \
apptainer build alphafold3-rocm.sif docker-daemon://alphafold3-rocm:latest
```

The resulting `alphafold3-rocm.sif` is self-contained.

## Obtaining Inputs, Model Parameters, and Databases

These steps are runtime-independent — follow the upstream instructions:

- Example input JSON: see
  [`README.md`](README.md#installation-and-running-your-first-prediction).
- Model parameters: see
  [`docs/model_parameters.md`](docs/model_parameters.md).
- Genetic databases:
  `./fetch_databases.sh [<DB_DIR>]`, per
  [`docs/installation.md`](docs/installation.md#obtaining-genetic-databases).

The examples below assume the same host-side layout as the upstream docs:

- `$HOME/af_input/fold_input.json`
- `$HOME/af_output/`
- `<MODEL_PARAMETERS_DIR>` — wherever the weights were unpacked
- `<DB_DIR>` — wherever `fetch_databases.sh` placed the databases

## Running a Prediction

The Apptainer equivalent of the upstream
`docker run ... --gpus all alphafold3 python run_alphafold.py ...` is:

```sh
apptainer exec \
    --rocm \
    --pwd /workspace \
    --bind $HOME/af_input:/root/af_input \
    --bind $HOME/af_output:/root/af_output \
    --bind <MODEL_PARAMETERS_DIR>:/root/models \
    --bind <DB_DIR>:/root/public_databases \
    alphafold3-rocm.sif \
    python3 run_alphafold.py \
    --json_path=/root/af_input/fold_input.json \
    --model_dir=/root/models \
    --output_dir=/root/af_output
```

Key differences from the Docker invocation:

| Docker                         | Apptainer                                                |
| ------------------------------ | -------------------------------------------------------- |
| `--gpus all`                   | `--rocm` (binds `/dev/kfd`, `/dev/dri`, `/opt/rocm/...`) |
| `--volume src:dst`             | `--bind src:dst`                                         |
| implicit `WORKDIR /app/alphafold` | `--pwd /workspace`                                       |
| `python run_alphafold.py`      | `python3 run_alphafold.py`                               |

### Selecting a GPU

Docker's `--gpus '"device=1"'` maps to `ROCR_VISIBLE_DEVICES`:

```sh
apptainer exec --rocm --env ROCR_VISIBLE_DEVICES=1 ... alphafold3-rocm.sif ...
```

### Databases on SSD With a Slower Fallback

Equivalent of the upstream split-database Docker example:

```sh
apptainer exec \
    --rocm \
    --pwd /workspace \
    --bind $HOME/af_input:/root/af_input \
    --bind $HOME/af_output:/root/af_output \
    --bind <MODEL_PARAMETERS_DIR>:/root/models \
    --bind <SSD_DB_DIR>:/root/public_databases \
    --bind <DB_DIR>:/root/public_databases_fallback \
    alphafold3-rocm.sif \
    python3 run_alphafold.py \
    --json_path=/root/af_input/fold_input.json \
    --model_dir=/root/models \
    --db_dir=/root/public_databases \
    --db_dir=/root/public_databases_fallback \
    --output_dir=/root/af_output
```

### Sanity-checking GPU Access

```sh
apptainer exec --rocm alphafold3-rocm.sif rocm-smi
apptainer exec --rocm --pwd /workspace alphafold3-rocm.sif \
    python3 verify_gpu.py
```

## Flags Reference

`run_alphafold.py` supports the same flags regardless of runtime — see
[`README.md`](README.md) for the common ones (`--run_data_pipeline`,
`--run_inference`, etc.). For the full list:

```sh
apptainer exec --pwd /workspace alphafold3-rocm.sif \
    python3 run_alphafold.py --help
```

## Environment Variables Baked Into the Image

Set at build time for ROCm compatibility; no need to pass on each invocation:

- `XLA_FLAGS="--xla_disable_hlo_passes=fusion,multi-output-fusion --xla_gpu_enable_triton_gemm=false"`
- `HSA_ENABLE_SDMA=0`
- `XLA_PYTHON_CLIENT_PREALLOCATE=true`
- `XLA_CLIENT_MEM_FRACTION=0.95`

Override any of them per invocation with `--env NAME=value`.
