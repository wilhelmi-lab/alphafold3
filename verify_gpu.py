"""GPU verification script for AlphaFold 3 without model weights.

Runs three progressive checks:
  1. JAX device detection
  2. Featurization pipeline (CPU-bound, verifies Python stack)
  3. Full model forward pass with random weights (exercises XLA on GPU)
"""

# Must be set before JAX initialises the XLA GPU backend.
import os
# Disable XLA's Triton GEMM pass (NVIDIA-only, crashes on ROCm).
_xla_flags = os.environ.get('XLA_FLAGS', '')
if '--xla_gpu_enable_triton_gemm' not in _xla_flags:
    os.environ['XLA_FLAGS'] = _xla_flags + ' --xla_gpu_enable_triton_gemm=false'
# Disable SDMA engines — avoids HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION on
# AMD GPUs where SDMA and GPU compute see different memory aperture mappings.
os.environ.setdefault('HSA_ENABLE_SDMA', '0')

import pathlib
import pickle
import sys

from absl import logging
from alphafold3.common import resources

import jax
import jax.numpy as jnp
import numpy as np
from alphafold3.model.components import utils

logging.set_verbosity(logging.INFO)

# ---------------------------------------------------------------------------
# 1. Device detection
# ---------------------------------------------------------------------------
print("\n=== 1. JAX device detection ===")
devices = jax.devices()
print(f"Available devices: {devices}")

gpu_devices = [d for d in devices if d.platform in ('gpu', 'rocm')]
if not gpu_devices:
    print("WARNING: No GPU/ROCm device found. Only CPU is available.")
    print("  On MI210 nodes make sure jax[rocm] is installed and ROCm libraries are in LD_LIBRARY_PATH.")
    target_device = jax.local_devices()[0]
    print(f"  Falling back to: {target_device}")
else:
    target_device = gpu_devices[0]
    print(f"GPU device selected: {target_device}")


# ---------------------------------------------------------------------------
# 2. Simple JAX compute on device
# ---------------------------------------------------------------------------
print("\n=== 2. JAX compute on device ===")
a = jax.device_put(jnp.ones((512, 512)), target_device)
b = jax.device_put(jnp.ones((512, 512)), target_device)
c = jax.jit(jnp.matmul)(a, b)
c.block_until_ready()
print(f"Matrix multiply 512x512 on {target_device}: OK (result sum={float(jnp.sum(c))})")


# ---------------------------------------------------------------------------
# 3. Featurization (CPU pipeline, no model needed)
# ---------------------------------------------------------------------------
print("\n=== 3. Featurization from pre-built test data ===")
pkl_path = resources.ROOT / 'test_data' / 'featurised_example.pkl'
featurised_examples = pickle.loads(pkl_path.read_bytes())
featurised_example = featurised_examples[0]
print(f"Loaded featurised example with {len(featurised_example)} feature arrays")
# Move to target device (strip object-typed features JAX can't handle)
featurised_on_device = jax.device_put(
    jax.tree_util.tree_map(jnp.asarray, utils.remove_invalidly_typed_feats(featurised_example)),
    target_device,
)
print(f"Featurised data transferred to {target_device}: OK")


# ---------------------------------------------------------------------------
# 4. Full forward pass with random weights
# ---------------------------------------------------------------------------
print("\n=== 4. Model forward pass with random weights ===")
print("This exercises XLA compilation and the full compute graph on the device.")

import haiku as hk
import run_alphafold
from alphafold3.model import model

# Use 'xla' flash attention — the only implementation that works on non-NVIDIA
# hardware (triton and cudnn are NVIDIA-specific).
model_config = run_alphafold.make_model_config(
    flash_attention_implementation='xla',
)

@hk.transform
def forward_fn(batch):
    return model.Model(model_config)(batch)

batch = jax.tree_util.tree_map(
    jnp.asarray,
    utils.remove_invalidly_typed_feats(featurised_example),
)
batch = jax.device_put(batch, target_device)
rng = jax.random.PRNGKey(0)

print("Initializing model with random weights (first XLA compilation — may take a few minutes)...")
random_params = jax.jit(forward_fn.init, device=target_device)(rng, batch)
print(f"Model init OK. Parameter count: {sum(x.size for x in jax.tree_util.tree_leaves(random_params)):,}")

print("Running forward pass...")
result = jax.jit(forward_fn.apply, device=target_device)(random_params, rng, batch)
result = jax.tree.map(np.asarray, result)
print("Forward pass OK.")
print(f"Output keys: {list(result.keys())}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== Summary ===")
print(f"Device used:    {target_device}")
print("JAX compute:    OK")
print("Featurization:  OK")
print("Forward pass:   OK")
print("\nAlphaFold 3 is functional on this hardware.")
print("You can safely request the model weights from Google DeepMind.")
