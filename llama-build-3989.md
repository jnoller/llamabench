# New llama.cpp build - 0.0.3989 (upstream tag: b3989)

This is a major update to our llama.cpp package and build, and includes the addition of the 
`llama.cpp-tools` and `gguf` packages to the feedstock.

Feedstock: https://github.com/AnacondaRecipes/llama.cpp-feedstock
Upstream: https://github.com/ggerganov/llama.cpp

# High level changes from 0.0.3747

- Fixed the windows build scripts to work with the new GGML style arguments, add in missing flags to windows cmake build.
- Moved from `CMAKE_CUDA_ARCHITECTURES=all` to `CMAKE_CUDA_ARCHITECTURES=all-major` to cut down on size and build time
  - The `all` target builds micro versions in addition to major releases and does not increase package compatibility and is unnecessary.
- Moved to using github tags instead of build tarballs, this allows the feedstock to more easily align with upstream releases and also allows 
  the git-dependent make logic in upstream to work correctly for version number injection.
- Added v2 (AVX) builds for linux and windows to improve our compatibility and performance story.
  - This is important for windows, as AVX2 can imply f16c support, and f16c is not universally supported on all chipsets that support AVX2, so for machines that do not support f16c, we need to potentially fall back to the AVX (v2) build.
- Moved the feedstock to a meta-package build with multiple outputs, this means the feedstock now emits `llama.cpp`, `llama.cpp-tools` and the `gguf` packages.
  - All 3 of the packages come from the same upstream source/tag, and this is more inline with the intended design of llama.cpp as a single monorepo that has minimal external dependencies.

# Packages included in llama.cpp feedstock (llama.cpp-meta)

- `llama.cpp` is the main package for the core binaries that ship with llama.cpp including:

| Binary                          | Description                                   |
|--------------------------------|------------------------------------------------|
| `llama-batched[.exe]`          | Batched inference interface                    |
| `llama-batched-bench[.exe]`    | Batched inference benchmarking tool            |
| `llama-bench[.exe]`            | Benchmarking interface                         |
| `llama-cli[.exe]`              | Command line interface                         |
| `llama-convert-llama2c[.exe]`  | LLAMA2 to GGML conversion tool                 |
| `llama-cvector-generator[.exe]`| C vector generation tool                       |
| `llama-embedding[.exe]`        | Embedding generation interface                 |
| `llama-eval-callback[.exe]`    | Evaluation callback testing tool               |
| `llama-export-lora[.exe]`      | LoRA export tool                               |
| `llama-gbnf-validator[.exe]`   | GBNF grammar validation tool                   |
| `llama-gen-docs[.exe]`         | Documentation generation tool                  |
| `llama-gguf[.exe]`             | GGUF conversion interface                      |
| `llama-gguf-hash[.exe]`        | GGUF hash generation tool                      |
| `llama-gguf-split[.exe]`       | GGUF file splitting tool                       |
| `llama-gritlm[.exe]`           | GritLM interface                               |
| `llama-imatrix[.exe]`          | Matrix operation tool                          |
| `llama-infill[.exe]`           | Text infilling interface                       |
| `llama-llava-cli[.exe]`        | LLAVA command line interface                   |
| `llama-lookahead[.exe]`        | Lookahead inference tool                       |
| `llama-lookup[.exe]`           | Lookup table interface                         |
| `llama-lookup-create[.exe]`    | Lookup table creation tool                     |
| `llama-lookup-merge[.exe]`     | Lookup table merging tool                      |
| `llama-lookup-stats[.exe]`     | Lookup table statistics tool                   |
| `llama-minicpmv-cli[.exe]`     | MiniCPM-V command line interface               |
| `llama-parallel[.exe]`         | Parallel inference interface                   |
| `llama-passkey[.exe]`          | Passkey generation tool                        |
| `llama-perplexity[.exe]`       | Perplexity calculation tool                    |
| `llama-q8dot[.exe]`            | Q8 dot product calculation tool                |
| `llama-quantize[.exe]`         | Model quantization interface                   |
| `llama-quantize-stats[.exe]`   | Quantization statistics tool                   |
| `llama-retrieval[.exe]`        | Retrieval interface                            |
| `llama-save-load-state[.exe]`  | State saving and loading tool                  |
| `llama-server[.exe]`           | Server interface                               |
| `llama-simple[.exe]`           | Simple inference interface                     |
| `llama-simple-chat[.exe]`      | Simple chat interface                          |
| `llama-speculative[.exe]`      | Speculative inference tool                     |
| `llama-tokenize[.exe]`         | Tokenization tool                              |
| `llama-vdot[.exe]`             | Vector dot product calculation tool            |


- `llama.cpp-tools` is a package that includes useful scripts and model conversion tools that ship with llama.cpp including:

| Script                                | Description                                   |
|---------------------------------------|-----------------------------------------------|
| `llama-convert-hf-to-gguf`            | Convert from Hugging Face safetensors to GGUF |
| `llama-convert-llama-ggml-to-gguf`    | Convert from GGML to GGUF                     |
| `llama-convert-lora-to-gguf`          | Convert from LoRA to GGUF                     |
| `llama-lava-surgery`                  | LLaVA surgery tool                            |
| `llama-lava-surgery_v2`               | LLaVA surgery tool v2                         |
| `llama-convert-image-encoder-to-gguf` | Convert from image encoder to GGUF            |


- `gguf` is a python package for writing binary files in the [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
(GGML Universal File) format and includes the following tools:

| Script                | Description                                   | 
|-----------------------|-----------------------------------------------|
| `gguf-convert-endian` | Convert the endianness of a GGUF file         |
| `gguf-dump`           | Dump the contents of a GGUF file              |
| `gguf-set-metadata`   | Set the metadata of a GGUF file               |
| `gguf-new-metadata`   | Create a new metadata object for a GGUF file  |

Note: The `llama.cpp-tools` and `gguf` packages are Python so there are not CUDA-enabled or hardware optimized builds.

| Package           | Minimum Python Version | Spec                         |
|-------------------|------------------------|------------------------------|
| `llama.cpp-tools` | 3.9                    | llama.cpp-tools==0.0.3989*   |
| `gguf`            | 3.9                    | gguf==0.10.0*                |


# llama.cpp builds available

For llama.cpp, the following build types are available - this includes both hardware optimized CPU, GPU and MPS builds. 

Note: GPU-enabled builds depend on CUDA 12.4 and will automatically install the CUDA toolkit from the `defaults` channel.

## MacOS:

| Build Type        | Architecture       | Spec                                      | Notes                   |
|-------------------|--------------------|-------------------------------------------|-------------------------|
| cpu_v1_accelerate | x86-64             | llama.cpp==0.0.3989=cpu_v1_accelerate*    | Pre-M1/M2 (arm64) Macs  |
| mps               | Apple MPS (arm64)  | llama.cpp==0.0.3989=mps*                  | M1/M2+ Macs             |

* For Mac OS, the metal-enabled `mps` build is recommended for all users running M1, M2 or later Macs.
* For pre-M1/M2 Macs, the `cpu_v1_accelerate` build is recommended.

Example install command for the metal-enabled `mps` build:
```bash
conda install -c ai-staging llama.cpp=0.0.3989=mps*
```

## Linux-64:

| Build Type      | Architecture      | Spec                                     | Notes                    |
|-----------------|-------------------|------------------------------------------|--------------------------|
| cpu_v1_mkl      | Intel MKL, SSE2   | llama.cpp==0.0.3989=cpu_v1_mkl*          |                          |
| cpu_v1_openblas | OpenBLAS, SSE2    | llama.cpp==0.0.3989=cpu_v1_openblas*     |                          |
| cpu_v2_mkl      | Intel MKL, AVX    | llama.cpp==0.0.3989=cpu_v2_mkl*          |                          |
| cpu_v2_openblas | OpenBLAS, AVX     | llama.cpp==0.0.3989=cpu_v2_openblas*     |                          |
| cpu_v3_mkl      | Intel MKL, AVX2   | llama.cpp==0.0.3989=cpu_v3_mkl*          |                          |
| cpu_v3_openblas | OpenBLAS, AVX2    | llama.cpp==0.0.3989=cpu_v3_openblas*     |                          |
| cuda124_v1      | CUDA 12.4, SSE2   | llama.cpp==0.0.3989=cuda124_v1*          |                          |
| cuda124_v2      | CUDA 12.4, AVX    | llama.cpp==0.0.3989=cuda124_v2*          |                          |
| cuda124_v3      | CUDA 12.4, AVX2   | llama.cpp==0.0.3989=cuda124_v3*          |                          |

* The `cpu_v*` builds are optimized for different CPU architectures (SSE2, AVX, AVX2) and use the Intel MKL or OpenBLAS libraries.
* The `cuda*` builds are optimized for the NVIDIA GPUs and use CUDA 12.4.

### Determining which cpu-optimized build to use on linux
To determine which cpu-optimized build to use on linux, you can use the following command to show the CPU architecture flags:

```bash
lscpu | grep -Eo 'sse2|avx2|avx512[^ ]*|avx\s'
```

You should see output like the following:

```
# lscpu | grep -Eo 'sse2|avx2|avx512[^ ]*|avx\s'
sse2
avx
avx2
avx512f
avx512dq
avx512cd
avx512bw
avx512vl
avx512_vnni
```

* If you see `sse2` in the output, but not `avx`, you should use the `cpu_v1_mkl` or `cpu_v1_openblas` variant.
* If you see `avx` in the output, you should use the `cpu_v2_mkl` or `cpu_v2_openblas` variant.
* If you see `avx2` in the output, you should use the `cpu_v3_mkl` or `cpu_v3_openblas` variant.

Note: The `_openblas` builds may lead to some performance improvements in prompt processing using batch sizes higher than 32 (the default is 512). Support with CPU-only BLAS implementations does not affect normal generation performance, do not use these builds unless you are optimizing / benchmarking prompt processing performance.

### Determining which gpu-optimized build to use on linux

If you are using an NVIDIA GPU supported by CUDA 12.4, you should use the `cuda` variants, you can confirm / view your GPU information using the `nvidia-smi` command if you already have the CUDA drivers/utilities installed:

```bash
[root@localhost ~]# nvidia-smi
Thu Nov 14 17:00:39 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:1E.0 Off |                    0 |
| N/A   26C    P0             27W /   70W |       0MiB /  15360MiB |      5%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

If you do not have the CUDA drivers / utilities installed, you can check your GPU by looking at the `lspci` output:

```bash
# lspci | grep -i nvidia
0000:00:1e.0: NVIDIA Corporation Device 1eb0 (rev a1)
```

For this Linux machine, the GPU is a Tesla T4 and the CUDA version is 12.4, it supports SSE2, AVX, AVX2, and AVX512, 
therefore the we would install the `cuda124_v3` variant to enable both GPU acceleration and AVX2 instructions.

Example install command for the `cuda124_v3` build:
```bash
conda install -c ai-staging llama.cpp=0.0.3989=cuda124_v3*
```

## Windows-64:

| Build Type        | Architecture      | Spec                                     | Notes                   |
|-------------------|-------------------|------------------------------------------|-------------------------|
| cpu_v1_mkl        | Intel MKL, SSE2   | llama.cpp==0.0.3989=cpu_v1_mkl*          |                         |
| cpu_v2_mkl        | Intel MKL, AVX    | llama.cpp==0.0.3989=cpu_v2_mkl*          |                         |
| cpu_v3_mkl        | Intel MKL, AVX2   | llama.cpp==0.0.3989=cpu_v3_mkl*          |                         |
| cuda124_v1        | CUDA 12.4, SSE2   | llama.cpp==0.0.3989=cuda124_v1*          |                         |
| cuda124_v2        | CUDA 12.4, AVX    | llama.cpp==0.0.3989=cuda124_v2*          |                         |
| cuda124_v3        | CUDA 12.4, AVX2   | llama.cpp==0.0.3989=cuda124_v3*          |                         |

### Determining which cpu-optimized build to use on Windows

To determine which cpu-optimized build to use on Windows, you can use the following command to show the CPU architecture flags on the command line:

```
conda install py-cpuinfo
python -c "import cpuinfo; flags=cpuinfo.get_cpu_info()['flags']; print('\n'.join(f for f in flags if f in ['sse2', 'avx', 'avx2'] or f.startswith('avx512')))"
```

You should see output like the following:

```
python -c "import cpuinfo; flags=cpuinfo.get_cpu_info()['flags']; print('\n'.join(f for f in flags if f in ['sse2', 'avx', 'avx2'] or f.startswith('avx512')))"
avx
avx2
avx512bw
avx512cd
avx512dq
avx512f
avx512vl
avx512vnni
sse2
```

* If you see `sse2` in the output, but not `avx`, you should use the `cpu_v1` variant.
* If you see `avx` but not `avx2` in the output, you should use the `cpu_v2` variant.
* If you see `avx2` in the output, you should use the `cpu_v3` variant.

### Determining which gpu-optimized build to use on Windows

If you are using an NVIDIA GPU supported by CUDA 12.4, you should use the `cuda124_v*` variants, you can confirm / view your GPU information using the `nvidia-smi.exe` 
command if you already have the CUDA drivers/utilities installed or using `wmic`:

```
wmic path win32_VideoController get name
```

Example output:
```
wmic path win32_VideoController get name
Name
Microsoft Basic Display Adapter
NVIDIA Tesla T4
```

For this Windows machine, the GPU is a NVIDIA Tesla T4 and the CUDA version is 12.4, it supports SSE2, AVX, AVX2, and AVX512, 
therefore the we would install the `cuda124_v3` variant to enable both GPU acceleration and AVX2 instructions.

Example install command for the `cuda124_v3` build:
```bash
conda install -c ai-staging llama.cpp=0.0.3989=cuda124_v3*
```

## Why not use the upstream package(s) directly?

Within llama.cpp upstream there are several pyproject.toml files:

```
llama.cpp/
  pyproject.toml
  gguf-py/
    pyproject.toml
```

These packages are not well-mainted within upstream:

- The PyPI version of `gguf` is frequently out of date with vs what is in the llama.cpp repo. 
- Changes to the underlying `gguf` format exposed by the `gguf` package are not always reflected in the pyproject.toml file as a new version.
- `llama.cpp-tools` has a hard dependency on the `gguf` package and they must be pinned/built and updated together. 
  - The version of the tools and gguf packages determine what AI models we can support across our portfolio, so we must ensure they are in lock-step from the same release tag.
- The main `pyproject.toml` defines a `llama.cpp-scripts` package that is out of date vs the rest of the repository.
  - It is missing dependencies and the ranges for the dependencies are not aligned with what is needed.
  - It omits several useful scripts that are present in the repository.

`llama.cpp` has evolved over time to more of a monorepo structure trying to minimize the entry points into the project, setting up environments and resolving dependencies. 
The recommended way to install the python tooling is to:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt
```

The `pip install -r requirements.txt` then installs all requirements listed in the /requirements/ directory including:

```
requirements-all.txt
requirements-compare-llama-bench.txt
requirements-convert_hf_to_gguf.txt
requirements-convert_hf_to_gguf_update.txt
requirements-convert_legacy_llama.txt
requirements-convert_llama_ggml_to_gguf.txt
requirements-convert_lora_to_gguf.txt
requirements-pydantic.txt
requirements-test-tokenizer-random.txt
```

This installs **all** dependencies for all python code within the llama.cpp repo including gguf, the conversion tools and everything in the /examples and tests directories. 

In order to work around this, we have added the `llama.cpp-tools` and `gguf` packages to the feedstock which include the necessary 
dependencies for the python tooling and the gguf package, the `llama.cpp-tools` package ignores the upstream pyproject.toml and defines a corrected 
`-tools` package that includes the conversion tools and pins to the verion of the `gguf` package built as part of the feedstock to ensure they're in sync.

As other tools and scripts are added to the repo, the current approach will allow us to add them ad-hoc to the `llama.cpp-tools` package.

## TODO

- Update to the latest upstream tag and build - more breaking changes came in after b3989.
- Identify and root cause why the upstream make logic that injects the version number from the git repo is not matching the tag that conda-build is 
  using to check out the source.
  - For example, the build should be 3989 but for some reason the make logic is setting it to 3991.
- Add in the v4 AVX512 builds for linux and windows (may restrict build hosts to only include AVX512 capable machines).
  - On AMD chips AVX512 performance is suppose to be way better than Intel / AVX512.
- Add /server/bench and /server/html assets to the `llama.cpp-tools` package (but this would depend on llama.cpp)
  - Where do html and other assets belong, especially if the server binary is in llama.cpp?
- Investigate: Patch upstream pyproject.toml files to align them with the rest of the repository and add the missing scripts/tools.
  - This would allow us to remove the `llama.cpp-tools` package from the feedstock and just execute the the install via `pip`
  - We would have to regenerate the patch for any new tool/script we wanted to add to `-tools|-scripts`
  - It would need to pull in several directories in the /examples directory.
  - May be able to extend this to add in the html / server assets and /examples/server/bench/... assets.
