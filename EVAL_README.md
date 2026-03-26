# CoIN Evaluation Setup - Current Workflow

This document explains the current strategy and setup for running the CoIN-Bench evaluations on the ENSTA H100 SLURM cluster.

## Execution Method

The evaluation is executed by submitting the SLURM job file **`run_coin_eval_final.sh`**:

```bash
sbatch run_coin_eval_final.sh
```

## Architecture: Why 4 GPUs and 3 Python Environments?

The evaluation pipeline includes a variety of heavy Vision-Language Models (VLMs) and Large Language Models (LLMs), as well as complex older simulators (Habitat). These components have conflicting Python dependencies and require substantial VRAM, so we distribute them across 4 isolated GPUs and 3 separate conda environments.

### The Three Conda Environments

1. **`coin_vllm_clone` (Python 3.9)**
   - Used for the Habitat simulation (`vlfm.run`) and the general Vision-Language-Navigation codebase.
   - Used for the Detection Stack (GroundingDINO, BLIP2, SAM).
   - *Reason*: Habitat requires an older Python setup, and compatibility is strictly tied to Python 3.9.

2. **`vllm_py310` (Python 3.10)**
   - Used exclusively to run the vLLM server.
   - *Reason*: Modern vLLM setups require Python 3.10+ and specific PyTorch/CUDA extensions that conflict with the Habitat codebase.

3. **`llava_next_py310` (Python 3.10)**
   - Used exclusively for the LLaVA-NeXT model.
   - *Reason*: LLaVA-NeXT strictly requires `transformers >= 4.45`, which causes dependency conflicts with other models in the main `vllm` or `coin_vllm` environments.

### GPU Allocation Strategy

The script (`run_coin_eval_final.sh`) partitions the workload to avoid Out-Of-Memory (OOM) errors and environment conflicts across the 4 allocated GPUs:

| Resource | Service / Responsibility | Environment Used | Model Details |
| :--- | :--- | :--- | :--- |
| **GPU 0** | **Detection Stack** | `coin_vllm_clone` (Py 3.9) | GroundingDINO, BLIP2-ITM, Mobile SAM |
| **GPU 1** | **Vision-Language Model** | `llava_next_py310` (Py 3.10) | LLaVA-NeXT |
| **GPU 2** | **CoIN Evaluation (Habitat)** | `coin_vllm_clone` (Py 3.9) | Simulator, `vlfm.run` logic |
| **GPU 3** | **LLM Server (vLLM)** | `vllm_py310` (Py 3.10) | Local LLM e.g. `gpt-oss-20b` (bfloat16) |

## Pipeline Execution Details

When `sbatch run_coin_eval_final.sh` is submitted, it does the following:

1. **Spin up Background Servers**: 
   - Starts the vLLM Server on GPU 3 (port `8000`).
   - Starts GroundingDINO, BLIP2, and SAM in parallel on GPU 0 (ports `12181`, `12182`, `12183`).
   - Starts LLaVA-NeXT on GPU 1 (port `12189`).
2. **Health Checks**: 
   - Uses `curl` to poll the `/v1/models` endpoint of the vLLM server to ensure it is healthy and the loaded model matches the expected `LOCAL_LLM_MODEL_NAME`.
   - Uses `nc -z` to poll the detection and VLM ports until they are fully available.
3. **Execution**:
   - Once all servers are marked healthy, it kicks off `python -m vlfm.run` on GPU 2.
   - *Note*: Currently, it is executing a small sanity check (`habitat_baselines.test_episode_count=10`).
4. **Cleanup**: 
   - A bash `trap` is defined so that if the job crashes, completes, or gets canceled, all background PIDs (vLLM, SAM, BLIP, GDINO, LLaVA) are properly killed avoiding "zombie" processes taking up node memory.

Logs for every server component are heavily compartmentalized and are tracked inside the `logs/` directory. Error output for the entire job is bound to `logs/coin_eval_<JOB_ID>.err/out`.