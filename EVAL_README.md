<div align="center">
  <h1>AIUTA-VLM-R1: Efficient Instance Navigation with<br>Knowledge Graphs and a Single 3B VLM</h1>
  <p><b>AI Project — MVA + ENSTA Paris, 2025–2026</b></p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.9%2F3.10-007EC6?style=flat-square&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
    <img src="https://img.shields.io/badge/VLM--R1-QWEN2.5VL--3B-97CA00?style=flat-square" alt="VLM R1" />
    <img src="https://img.shields.io/badge/ZERO--API--COST-LOCAL%20ONLY-8A2BE2?style=flat-square" alt="Zero API Cost" />
  </p>
</div>

<hr />

## 🚀 Execution Method

The evaluation is seamlessly executed by submitting the unified SLURM job file on the ENSTA H100 cluster:

```bash
sbatch run_coin_eval_final.sh
```

<br>

## 🧩 Architecture: Why 4 GPUs and 3 Python Environments?

The evaluation pipeline orchestrated in this project connects a variety of heavy Vision-Language Models (VLMs), Large Language Models (LLMs), and complex simulators (Habitat). To gracefully handle conflicting Python dependencies and massive VRAM requirements, the pipeline distributes these workloads across **4 isolated GPUs** and **3 custom conda environments**.

### 🌿 The Three Conda Environments

1. **`coin_vllm_clone` (Python 3.9)**
   - **Scope:** Runs the Habitat simulation (`vlfm.run`), the core Vision-Language-Navigation (VLN) controller, and the entire **Detection Stack** (GroundingDINO, BLIP2, Mobile SAM).
   - **Reasoning:** Habitat relies on legacy dependencies and its compatibility is strictly mapped to Python 3.9.

2. **`vllm_py310` (Python 3.10)**
   - **Scope:** Dedicated solely to hosting the ultra-fast **vLLM Inference server**.
   - **Reasoning:** Modern vLLM optimization requires Python 3.10+ along with specialized PyTorch/CUDA extensions that conflict directly with our legacy Habitat configuration.

3. **`llava_next_py310` (Python 3.10)**
   - **Scope:** Used exclusively for invoking the **LLaVA-NeXT** model integration.
   - **Reasoning:** LLaVA-NeXT inherently requires `transformers >= 4.45`, preventing it from residing in the older `coin_vllm` stack.

<br>

### 💻 GPU Allocation Strategy

To completely bypass Out-Of-Memory (OOM) errors and avoid cross-library contamination, `run_coin_eval_final.sh` spatially partitions the inference engines optimally across 4 available GPUs:

| Resource | Service / Component | Environment Engine | Active Models |
| :---: | :--- | :--- | :--- |
| **GPU 0** | 🎯 **Detection Stack** | `coin_vllm_clone` (Py 3.9) | GroundingDINO, BLIP2-ITM, Mobile SAM |
| **GPU 1** | 👁️ **Vision Analyzer** | `llava_next_py310` (Py 3.10) | LLaVA-NeXT |
| **GPU 2** | 🕹️ **Simulation & Logic** | `coin_vllm_clone` (Py 3.9) | Habitat Simulator, `vlfm.run` evaluation |
| **GPU 3** | 🧠 **Local LLM Server** | `vllm_py310` (Py 3.10) | Local LLM Engine (e.g., `gpt-oss-20b` via bfloat16) |

<br>

## ⚙️ Pipeline Execution Lifecycle

When `sbatch run_coin_eval_final.sh` enters the queue and initializes, it strictly adheres to the following sequence:

1. **Background Server Orchestration 🌍**
   - Boots the **vLLM Server** on GPU 3 (`port 8000`).
   - Concurrently spawns **GroundingDINO**, **BLIP2**, and **Mobile SAM** onto GPU 0 (`ports 12181, 12182, 12183`).
   - Deploys **LLaVA-NeXT** mapping to GPU 1 (`port 12189`).
   
2. **Automated Health Readiness 🩺**
   - Implements automated `curl` polling towards vLLM's `/v1/models` endpoint checking model-ID consistency.
   - Triggers rapid `nc -z` network socket pings continuously verifying detection and VLM availability before starting the simulator.
   
3. **Core Evaluation Loop 🏃‍♂️**
   - Once total system health is `✓ OK`, it seamlessly delegates simulation oversight to `python -m vlfm.run` on GPU 2.
   - *Active Job Note:* Defaults to executing a 10-episode sanity check (`habitat_baselines.test_episode_count=10`).
   
4. **Guaranteed Graceful Cleanup 🧹**
   - Employs a bash `trap` safety net. Whether the execution crashes, times out, or successfully concludes, all background processes (`LLM_PID`, `GDINO_PID`, `SAM_PID`, etc.) are rigorously terminated, preventing ghost processes from hoarding node memory.

> 📝 **Logging:** Total operational chatter is thoroughly isolated. Background daemon logs are routed independently within `logs/`, while the main unified runtime errors map dynamically to `logs/coin_eval_<SLURM_JOB_ID>.err/out`.
