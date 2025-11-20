# 看进程实际用到哪一个 libnvJitLink.so.12
python - << 'PY'
import subprocess, sys
subprocess.run(
    "ldd $(python -c 'import torch, inspect; import torch._C; print(torch._C.__file__)') | grep nvJitLink",
    shell=True
)
PY


#!/usr/bin/env bash
set -e

export CONDA_PREFIX=/opt/nas/p/conda/envs/verlaa

# 1. 明确把 torch 自带的 CUDA 库放在最前面
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:\
$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib:\
$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:\
$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:\
$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:\
${LD_LIBRARY_PATH:-}"

# 2. 可选：强制预加载正确的 nvJitLink
export LD_PRELOAD="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12:${LD_PRELOAD:-}"



#!/usr/bin/env bash
set -euo pipefail

# 你的 env 路径
export CONDA_PREFIX=/opt/nas/p/conda/envs/verlaa

echo ">>> 1. 查找 conda 里 nvjitlink 目录:"
python - << 'PY'
import nvidia.nvjitlink, os, inspect
print("nvidia.nvjitlink package dir:", os.path.dirname(inspect.getfile(nvidia.nvjitlink)))
PY

# 把上一段打印出的目录记一下，比如是：
# /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/nvidia/nvjitlink

NVJIT_DIR="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib"
CUSPARSE_DIR="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib"
CUBLAS_DIR="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib"
RUNTIME_DIR="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib"
NVRTC_DIR="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib"

echo ">>> 2. 覆盖 LD_LIBRARY_PATH:"
export LD_LIBRARY_PATH="$NVJIT_DIR:$CUSPARSE_DIR:$CUBLAS_DIR:$RUNTIME_DIR:$NVRTC_DIR"

echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

echo ">>> 3. 在同一个进程里检查 torch 和 nvJitLink:"
python - << 'PY'
import torch, os, subprocess
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("LD_LIBRARY_PATH in python:", os.environ.get("LD_LIBRARY_PATH","<empty>"))
subprocess.run(
    "ldd $(python -c 'import torch,inspect; import torch._C; print(torch._C.__file__)') | grep nvJitLink || echo NO_NVJIT",
    shell=True,
)
PY



export CONDA_PREFIX=/opt/nas/p/conda/envs/verlaa

NVJIT_DIR="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib"
CUSPARSE_DIR="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib"
CUBLAS_DIR="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib"
RUNTIME_DIR="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib"
NVRTC_DIR="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib"

export LD_LIBRARY_PATH="$NVJIT_DIR:$CUSPARSE_DIR:$CUBLAS_DIR:$RUNTIME_DIR:$NVRTC_DIR:$HOME/numa-migrate:${LD_LIBRARY_PATH:-}"

python - << 'PY'
import torch, subprocess
subprocess.run(
    "ldd $(python -c 'import torch,inspect; import torch._C; print(torch._C.__file__)') | grep nvJitLink",
    shell=True,
)
PY

python - << 'PY'
from sgl_kernel import common_ops
print("sgl_kernel / libnuma OK")
PY



[36m(WorkerDict pid=26103)[0m Qwen3ForCausalLM contains 4.02B parameters
[36m(WorkerDict pid=26103)[0m wrap_policy: functools.partial(<function _or_policy at 0x7ef8338a8b80>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x7ef8338a8a40>, transformer_layer_cls={<class 'transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer'>})])
[36m(WorkerDict pid=26103)[0m Total steps: 4365, num_warmup_steps: 0
[36m(WorkerDict pid=26103)[0m Actor use_remove_padding=True
[36m(WorkerDict pid=26103)[0m Actor use_fused_kernels=False
[36m(WorkerDict pid=26108)[0m Monkey patch _flash_attention_forward in transformers.integrations.flash_attention[32m [repeated 8x across cluster][0m
[36m(WorkerDict pid=26108)[0m Skipping monkey patch for Qwen3ForCausalLM as use_fused_kernels is False or fused_kernels_backend is torch[32m [repeated 8x across cluster][0m
[33m(raylet)[0m A worker died or was killed while executing a task by an unexpected system error. To troubleshoot the problem, check the logs for the dead worker. Lease ID: 010000008be35baadbde38fcf261df5e06a41ee17e1fada328e430cd1e68a26e Worker ID: 12a64c4cf872c7f121bbeb540c41ab9aa6db76930b4fac8ca49aec8a Node ID: 0b6abfec3ad8acdb94e5c2968c71a4c4582b2f43f3d72a9a74a04f85 Worker IP address: 10.80.8.181 Worker port: 45555 Worker PID: 26103 Worker exit type: SYSTEM_ERROR Worker exit detail: Worker unexpectedly exits with a connection error code 2. End of file. There are some potential root causes. (1) The process is killed by SIGKILL by OOM killer due to high memory usage. (2) ray stop --force is called. (3) The worker is crashed unexpectedly due to SIGSEGV or other unexpected errors.
Error executing job with overrides: ['algorithm.adv_estimator=grpo', 'data.train_batch_size=8', 'data.max_prompt_length=1024', 'data.max_response_length=3072', 'data.filter_overlong_prompts=True', 'data.truncation=error', 'data.return_raw_chat=True', 'actor_rollout_ref.model.path=/opt/nas/s/enyu/verl_askActively/verl/models/Qwen3-4B-Instruct-2507-sft', 'actor_rollout_ref.model.use_remove_padding=True', 'actor_rollout_ref.model.enable_gradient_checkpointing=True', 'actor_rollout_ref.actor.optim.lr=1e-6', 'actor_rollout_ref.actor.ppo_mini_batch_size=8', 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2', 'actor_rollout_ref.actor.use_kl_loss=True', 'actor_rollout_ref.actor.kl_loss_coef=0.001', 'actor_rollout_ref.actor.kl_loss_type=low_var_kl', 'actor_rollout_ref.actor.entropy_coeff=0', 'actor_rollout_ref.actor.fsdp_config.param_offload=False', 'actor_rollout_ref.actor.fsdp_config.optimizer_offload=False', 'actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16', 'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2', 'actor_rollout_ref.rollout.tensor_model_parallel_size=2', 'actor_rollout_ref.rollout.name=sglang', 'actor_rollout_ref.rollout.gpu_memory_utilization=0.7', 'actor_rollout_ref.rollout.n=8', 'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2', 'actor_rollout_ref.ref.fsdp_config.param_offload=False', 'algorithm.use_kl_in_reward=False', 'trainer.critic_warmup=0', 'trainer.logger=["console"]', 'trainer.project_name=askActively', 'trainer.experiment_name=askActively-RL', 'trainer.n_gpus_per_node=8', 'trainer.nnodes=1', 'trainer.save_freq=50', 'trainer.test_freq=20', 'trainer.rollout_data_dir=/opt/nas/s/enyu/verl_askActively/verl/rollout_data', 'data.train_files=/opt/nas/s/enyu/verl_askActively/verl/data/rl/test_rl.parquet', 'data.val_files=/opt/nas/s/enyu/verl_askActively/verl/data/rl/test_rl.parquet', 'actor_rollout_ref.rollout.multi_turn.interaction_config_path=/opt/nas/s/enyu/verl_askActively/verl/trainer/config/interaction/askActively_interaction.yaml', 'trainer.total_epochs=15']
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/opt/nas/s/enyu/verl_askActively/verl/trainer/main_ppo.py", line 406, in <module>
    main()
  File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/hydra/main.py", line 94, in decorated_main
    _run_hydra(
  File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/hydra/_internal/utils.py", line 394, in _run_hydra
    _run_app(
  File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/hydra/_internal/utils.py", line 457, in _run_app
    run_and_report(
  File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/hydra/_internal/utils.py", line 223, in run_and_report
    raise ex
  File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/hydra/_internal/utils.py", line 220, in run_and_report
    return func()
           ^^^^^^
  File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/hydra/_internal/utils.py", line 458, in <lambda>
    lambda: hydra.run(
            ^^^^^^^^^^
  File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/hydra/_internal/hydra.py", line 132, in run
    _ = ret.return_value
        ^^^^^^^^^^^^^^^^
  File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/hydra/core/utils.py", line 260, in return_value
    raise self._return_value
  File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/hydra/core/utils.py", line 186, in run_job
    ret.return_value = task_function(task_cfg)
                       ^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/nas/s/enyu/verl_askActively/verl/trainer/main_ppo.py", line 45, in main
    run_ppo(config)
  File "/opt/nas/s/enyu/verl_askActively/verl/trainer/main_ppo.py", line 88, in run_ppo
    ray.get(runner.run.remote(config))
  File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/ray/_private/auto_init_hook.py", line 22, in auto_init_wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/ray/_private/client_mode_hook.py", line 104, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/ray/_private/worker.py", line 2961, in get
    values, debugger_breakpoint = worker.get_objects(
                                  ^^^^^^^^^^^^^^^^^^^
  File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/ray/_private/worker.py", line 1026, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(ActorDiedError): [36mray::TaskRunner.run()[39m (pid=15334, ip=10.80.8.181, actor_id=fd3e78044484d6f1fdf0392301000000, repr=<main_ppo.TaskRunner object at 0x7f6f9cff1a10>)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/nas/s/enyu/verl_askActively/verl/trainer/main_ppo.py", line 308, in run
    trainer.init_workers()
  File "/opt/nas/s/enyu/verl_askActively/verl/trainer/ppo/ray_trainer.py", line 737, in init_workers
    self.actor_rollout_wg.init_model()
  File "/opt/nas/s/enyu/verl_askActively/verl/single_controller/ray/base.py", line 52, in __call__
    output = ray.get(output)
             ^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^
                                  ^^^^^^^^^^^^^^^^^^^
ray.exceptions.ActorDiedError: The actor died unexpectedly before finishing this task.
	class_name: create_colocated_worker_cls.<locals>.WorkerDict
	actor_id: 7a7e28706e933f93da5fe49701000000
	pid: 26103
	name: WQ0HJEWorkerDict_0:0
	namespace: 00e278ee-52c1-4f18-85b6-8be135bdb1b6
	ip: 10.80.8.181
The actor is dead because its worker process has died. Worker exit type: SYSTEM_ERROR Worker exit detail: Worker unexpectedly exits with a connection error code 2. End of file. There are some potential root causes. (1) The process is killed by SIGKILL by OOM killer due to high memory usage. (2) ray stop --force is called. (3) The worker is crashed unexpectedly due to SIGSEGV or other unexpected errors.
[36m(WorkerDict pid=26110)[0m [rank7]:[W1120 01:13:20.886560089 TCPStore.cpp:125] [c10d] recvValue failed on SocketImpl(fd=81, addr=[10-80-8-181.prometheus-kube-promethe-kube-proxy.kube-system.svc.cluster.local]:51546, remote=[10-80-8-181.mllm-ppo-qwen-npsvc.mllm.svc.cluster.local]:55903): Connection reset by peer[32m [repeated 5x across cluster][0m
[36m(WorkerDict pid=26110)[0m Exception raised from recvBytes at /pytorch/torch/csrc/distributed/c10d/Utils.hpp:675 (most recent call first):[32m [repeated 5x across cluster][0m
[36m(WorkerDict pid=26110)[0m frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x98 (0x7ff2443785e8 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libc10.so)[32m [repeated 5x across cluster][0m
[36m(WorkerDict pid=26110)[0m frame #8: <unknown function> + 0x126850 (0x7ff2962a7850 in /usr/lib/x86_64-linux-gnu/libc.so.6)[32m [repeated 30x across cluster][0m
[36m(WorkerDict pid=26110)[0m frame #4: c10d::TCPStore::check(std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > const&) + 0x2a9 (0x7fd4cd6ca2a9 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)[32m [repeated 5x across cluster][0m
[36m(WorkerDict pid=26110)[0m frame #5: c10d::ProcessGroupNCCL::heartbeatMonitor() + 0x379 (0x7fd48edc99f9 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)[32m [repeated 5x across cluster][0m
[36m(WorkerDict pid=26110)[0m [rank7]:[W1120 01:13:20.892421310 ProcessGroupNCCL.cpp:1662] [PG ID 0 PG GUID 0(default_pg) Rank 7] Failed to check the "should dump" flag on TCPStore, (maybe TCPStore server has shut down too early), with error: Connection reset by peer[32m [repeated 5x across cluster][0m
[33m(raylet)[0m A worker died or was killed while executing a task by an unexpected system error. To troubleshoot the problem, check the logs for the dead worker. Lease ID: 07000000451fa06f795301a489e70b57bac528c1b6a7121965675e8c522ce795 Worker ID: 715673e9d8bb41d7827c3b7105eb8e631e7b4824ca0ee8bb6c57c326 Node ID: 0b6abfec3ad8acdb94e5c2968c71a4c4582b2f43f3d72a9a74a04f85 Worker IP address: 10.80.8.181 Worker port: 42351 Worker PID: 26109 Worker exit type: SYSTEM_ERROR Worker exit detail: Worker unexpectedly exits with a connection error code 2. End of file. There are some potential root causes. (1) The process is killed by SIGKILL by OOM killer due to high memory usage. (2) ray stop --force is called. (3) The worker is crashed unexpectedly due to SIGSEGV or other unexpected errors.



  warnings.warn(
[36m(pid=15334)[0m /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
[36m(pid=15334)[0m   import pynvml  # type: ignore[import]
[36m(TaskRunner pid=15334)[0m /opt/nas/s/enyu/verl_askActively/verl/trainer/main_ppo.py:258: UserWarning: Disabled critic as algorithm.adv_estimator != gae. If it is not intended, please set critic.enable=True
[36m(TaskRunner pid=15334)[0m   use_critic=need_critic(config),
[36m(TaskRunner pid=15334)[0m /opt/nas/s/enyu/verl_askActively/verl/utils/profiler/config.py:49: UserWarning: Torch profiler tool config is not fully supported now.
[36m(TaskRunner pid=15334)[0m   warnings.warn("Torch profiler tool config is not fully supported now.", stacklevel=1)
[36m(TaskRunner pid=15334)[0m Setting TOKENIZERS_PARALLELISM=false for forked processes.
[36m(TaskRunner pid=15334)[0m WARNING:2025-11-20 01:08:17,569:Setting TOKENIZERS_PARALLELISM=false for forked processes.
[36m(TaskRunner pid=15334)[0m 
Filtering prompts longer than 1024 tokens (num_proc=1):   0%|          | 0/2329 [00:00<?, ? examples/s]
[36m(TaskRunner pid=15334)[0m 
Filtering prompts longer than 1024 tokens (num_proc=1):  43%|████▎     | 1000/2329 [00:03<00:04, 326.59 examples/s]
[36m(TaskRunner pid=15334)[0m 
Filtering prompts longer than 1024 tokens (num_proc=1):  86%|████████▌ | 2000/2329 [00:05<00:00, 375.76 examples/s]
[36m(TaskRunner pid=15334)[0m 
Filtering prompts longer than 1024 tokens (num_proc=1): 100%|██████████| 2329/2329 [00:06<00:00, 386.28 examples/s]
[36m(TaskRunner pid=15334)[0m 
Filtering prompts longer than 1024 tokens (num_proc=1): 100%|██████████| 2329/2329 [00:07<00:00, 322.80 examples/s]
[36m(TaskRunner pid=15334)[0m Setting TOKENIZERS_PARALLELISM=false for forked processes.
[36m(TaskRunner pid=15334)[0m WARNING:2025-11-20 01:08:49,192:Setting TOKENIZERS_PARALLELISM=false for forked processes.
[36m(TaskRunner pid=15334)[0m /opt/nas/s/enyu/verl_askActively/verl/trainer/ppo/ray_trainer.py:339: UserWarning: Disabled critic as algorithm.adv_estimator != gae. If it is not intended, please set critic.enable=True
[36m(TaskRunner pid=15334)[0m   self.use_critic = need_critic(self.config)
[36m(pid=26109)[0m /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
[36m(pid=26109)[0m   import pynvml  # type: ignore[import]
[36m(WorkerDict pid=26106)[0m Flash Attention 2 only supports torch.float16 and torch.bfloat16 dtypes, but the current dype in Qwen3ForCausalLM is torch.float32. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator, or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`
[36m(WorkerDict pid=26106)[0m Flash Attention 2 only supports torch.float16 and torch.bfloat16 dtypes, but the current dype in Qwen3Model is torch.float32. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator, or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`
[36m(pid=26110)[0m /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.[32m [repeated 7x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)[0m
[36m(pid=26110)[0m   import pynvml  # type: ignore[import][32m [repeated 7x across cluster][0m
[36m(WorkerDict pid=26103)[0m 
Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]
[36m(WorkerDict pid=26110)[0m 
Loading checkpoint shards:  50%|█████     | 1/2 [00:30<00:30, 30.52s/it]
[36m(WorkerDict pid=26110)[0m Flash Attention 2 only supports torch.float16 and torch.bfloat16 dtypes, but the current dype in Qwen3Model is torch.float32. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator, or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`[32m [repeated 14x across cluster][0m
[36m(WorkerDict pid=26104)[0m 
Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s][32m [repeated 7x across cluster][0m
[36m(WorkerDict pid=26104)[0m 
Loading checkpoint shards: 100%|██████████| 2/2 [00:57<00:00, 28.25s/it]
Loading checkpoint shards: 100%|██████████| 2/2 [00:57<00:00, 28.66s/it]
[36m(WorkerDict pid=26105)[0m 
Loading checkpoint shards:  50%|█████     | 1/2 [00:30<00:30, 30.94s/it][32m [repeated 7x across cluster][0m
[36m(WorkerDict pid=26106)[0m /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
[36m(WorkerDict pid=26106)[0m   import pynvml  # type: ignore[import]
[36m(WorkerDict pid=26110)[0m 
Loading checkpoint shards: 100%|██████████| 2/2 [00:57<00:00, 28.56s/it]
Loading checkpoint shards: 100%|██████████| 2/2 [00:57<00:00, 28.85s/it][32m [repeated 7x across cluster][0m
[36m(WorkerDict pid=26106)[0m 
Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]
[36m(WorkerDict pid=26106)[0m /opt/nas/s/enyu/verl_askActively/verl/utils/profiler/config.py:49: UserWarning: Torch profiler tool config is not fully supported now.
[36m(WorkerDict pid=26106)[0m   warnings.warn("Torch profiler tool config is not fully supported now.", stacklevel=1)
[36m(WorkerDict pid=26106)[0m /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:680: FutureWarning: FSDP.state_dict_type() and FSDP.set_state_dict_type() are being deprecated. Please use APIs, get_state_dict() and set_state_dict(), which can support different parallelisms, FSDP1, FSDP2, DDP. API doc: https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict .Tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html .
[36m(WorkerDict pid=26106)[0m   warnings.warn(
[36m(WorkerDict pid=26110)[0m /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.[32m [repeated 7x across cluster][0m
[36m(WorkerDict pid=26110)[0m   import pynvml  # type: ignore[import][32m [repeated 7x across cluster][0m
[36m(WorkerDict pid=26108)[0m 
Loading checkpoint shards: 100%|██████████| 2/2 [00:00<00:00, 14.36it/s]
Loading checkpoint shards: 100%|██████████| 2/2 [00:00<00:00, 14.33it/s][32m [repeated 8x across cluster][0m
[36m(WorkerDict pid=26108)[0m 
Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s][32m [repeated 7x across cluster][0m
[36m(WorkerDict pid=26110)[0m /opt/nas/s/enyu/verl_askActively/verl/utils/profiler/config.py:49: UserWarning: Torch profiler tool config is not fully supported now.[32m [repeated 7x across cluster][0m
[36m(WorkerDict pid=26110)[0m   warnings.warn("Torch profiler tool config is not fully supported now.", stacklevel=1)[32m [repeated 7x across cluster][0m
[36m(WorkerDict pid=26107)[0m [2025-11-20 01:12:06] Downcasting torch.float32 to torch.bfloat16.
[36m(WorkerDict pid=26107)[0m [2025-11-20 01:12:06] Using default HuggingFace chat template with detected content format: string
[36m(WorkerDict pid=26110)[0m /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:680: FutureWarning: FSDP.state_dict_type() and FSDP.set_state_dict_type() are being deprecated. Please use APIs, get_state_dict() and set_state_dict(), which can support different parallelisms, FSDP1, FSDP2, DDP. API doc: https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict .Tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html .[32m [repeated 3x across cluster][0m
[36m(WorkerDict pid=26110)[0m   warnings.warn([32m [repeated 3x across cluster][0m
[36m(WorkerDict pid=26107)[0m /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
[36m(WorkerDict pid=26107)[0m   import pynvml  # type: ignore[import]
[36m(WorkerDict pid=26105)[0m [2025-11-20 01:12:06] Downcasting torch.float32 to torch.bfloat16.[32m [repeated 3x across cluster][0m
[36m(WorkerDict pid=26109)[0m [2025-11-20 01:12:06] Using default HuggingFace chat template with detected content format: string[32m [repeated 3x across cluster][0m
[36m(WorkerDict pid=26105)[0m /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
[36m(WorkerDict pid=26105)[0m   import pynvml  # type: ignore[import]
[36m(WorkerDict pid=26107)[0m /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.[32m [repeated 12x across cluster][0m
[36m(WorkerDict pid=26107)[0m   import pynvml  # type: ignore[import][32m [repeated 12x across cluster][0m
[36m(WorkerDict pid=26107)[0m [2025-11-20 01:13:05 TP0] Downcasting torch.float32 to torch.bfloat16.
[36m(WorkerDict pid=26107)[0m [2025-11-20 01:13:05 TP0] Init torch distributed begin.
[36m(WorkerDict pid=26103)[0m [2025-11-20 01:13:05 TP1] Scheduler hit an exception: Traceback (most recent call last):
[36m(WorkerDict pid=26103)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/sglang/srt/managers/scheduler.py", line 2534, in run_scheduler_process
[36m(WorkerDict pid=26103)[0m     scheduler = Scheduler(
[36m(WorkerDict pid=26103)[0m                 ^^^^^^^^^^
[36m(WorkerDict pid=26103)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/sglang/srt/managers/scheduler.py", line 313, in __init__
[36m(WorkerDict pid=26103)[0m     self.tp_worker = TpWorkerClass(
[36m(WorkerDict pid=26103)[0m                      ^^^^^^^^^^^^^^
[36m(WorkerDict pid=26103)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/sglang/srt/managers/tp_worker_overlap_thread.py", line 67, in __init__
[36m(WorkerDict pid=26103)[0m     self.worker = TpModelWorker(
[36m(WorkerDict pid=26103)[0m                   ^^^^^^^^^^^^^^
[36m(WorkerDict pid=26103)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/sglang/srt/managers/tp_worker.py", line 84, in __init__
[36m(WorkerDict pid=26103)[0m     self.model_runner = ModelRunner(
[36m(WorkerDict pid=26103)[0m                         ^^^^^^^^^^^^
[36m(WorkerDict pid=26103)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/sglang/srt/model_executor/model_runner.py", line 235, in __init__
[36m(WorkerDict pid=26103)[0m     min_per_gpu_memory = self.init_torch_distributed()
[36m(WorkerDict pid=26103)[0m                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(WorkerDict pid=26103)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/sglang/srt/model_executor/model_runner.py", line 551, in init_torch_distributed
[36m(WorkerDict pid=26103)[0m     init_distributed_environment(
[36m(WorkerDict pid=26103)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/sglang/srt/distributed/parallel_state.py", line 1255, in init_distributed_environment
[36m(WorkerDict pid=26103)[0m     torch.distributed.init_process_group(
[36m(WorkerDict pid=26103)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/distributed/c10d_logger.py", line 81, in wrapper
[36m(WorkerDict pid=26103)[0m     return func(*args, **kwargs)
[36m(WorkerDict pid=26103)[0m            ^^^^^^^^^^^^^^^^^^^^^
[36m(WorkerDict pid=26103)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/distributed/c10d_logger.py", line 95, in wrapper
[36m(WorkerDict pid=26103)[0m     func_return = func(*args, **kwargs)
[36m(WorkerDict pid=26103)[0m                   ^^^^^^^^^^^^^^^^^^^^^
[36m(WorkerDict pid=26103)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/distributed/distributed_c10d.py", line 1710, in init_process_group
[36m(WorkerDict pid=26103)[0m     store, rank, world_size = next(rendezvous_iterator)
[36m(WorkerDict pid=26103)[0m                               ^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(WorkerDict pid=26103)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/distributed/rendezvous.py", line 230, in _tcp_rendezvous_handler
[36m(WorkerDict pid=26103)[0m     store = _create_c10d_store(
[36m(WorkerDict pid=26103)[0m             ^^^^^^^^^^^^^^^^^^^
[36m(WorkerDict pid=26103)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/distributed/rendezvous.py", line 198, in _create_c10d_store
[36m(WorkerDict pid=26103)[0m     return TCPStore(
[36m(WorkerDict pid=26103)[0m            ^^^^^^^^^
[36m(WorkerDict pid=26103)[0m RuntimeError: nonce == returnedNonce INTERNAL ASSERT FAILED at "/pytorch/torch/csrc/distributed/c10d/TCPStore.cpp":418, please report a bug to PyTorch. Ping failed, invalid nonce returned
[36m(WorkerDict pid=26103)[0m 
[36m(WorkerDict pid=26103)[0m [2025-11-20 01:13:05 TP0] Scheduler hit an exception: Traceback (most recent call last):
[36m(WorkerDict pid=26103)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/sglang/srt/managers/scheduler.py", line 2534, in run_scheduler_process
[36m(WorkerDict pid=26103)[0m     scheduler = Scheduler(
[36m(WorkerDict pid=26103)[0m                 ^^^^^^^^^^
[36m(WorkerDict pid=26103)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/sglang/srt/managers/scheduler.py", line 313, in __init__
[36m(WorkerDict pid=26103)[0m     self.tp_worker = TpWorkerClass(
[36m(WorkerDict pid=26103)[0m                      ^^^^^^^^^^^^^^
[36m(WorkerDict pid=26103)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/sglang/srt/managers/tp_worker_overlap_thread.py", line 67, in __init__
[36m(WorkerDict pid=26103)[0m     self.worker = TpModelWorker(
[36m(WorkerDict pid=26103)[0m                   ^^^^^^^^^^^^^^
[36m(WorkerDict pid=26103)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/sglang/srt/managers/tp_worker.py", line 84, in __init__
[36m(WorkerDict pid=26103)[0m     self.model_runner = ModelRunner(
[36m(WorkerDict pid=26103)[0m                         ^^^^^^^^^^^^
[36m(WorkerDict pid=26103)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/sglang/srt/model_executor/model_runner.py", line 235, in __init__
[36m(WorkerDict pid=26103)[0m     min_per_gpu_memory = self.init_torch_distributed()
[36m(WorkerDict pid=26103)[0m                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(WorkerDict pid=26103)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/sglang/srt/model_executor/model_runner.py", line 551, in init_torch_distributed
[36m(WorkerDict pid=26103)[0m     init_distributed_environment(
[36m(WorkerDict pid=26103)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/sglang/srt/distributed/parallel_state.py", line 1255, in init_distributed_environment
[36m(WorkerDict pid=26103)[0m     torch.distributed.init_process_group(
[36m(WorkerDict pid=26103)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/distributed/c10d_logger.py", line 81, in wrapper
[36m(WorkerDict pid=26103)[0m     return func(*args, **kwargs)
[36m(WorkerDict pid=26103)[0m            ^^^^^^^^^^^^^^^^^^^^^
[36m(WorkerDict pid=26103)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/distributed/c10d_logger.py", line 95, in wrapper
[36m(WorkerDict pid=26103)[0m     func_return = func(*args, **kwargs)
[36m(WorkerDict pid=26103)[0m                   ^^^^^^^^^^^^^^^^^^^^^
[36m(WorkerDict pid=26103)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/distributed/distributed_c10d.py", line 1710, in init_process_group
[36m(WorkerDict pid=26103)[0m     store, rank, world_size = next(rendezvous_iterator)
[36m(WorkerDict pid=26103)[0m                               ^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(WorkerDict pid=26103)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/distributed/rendezvous.py", line 230, in _tcp_rendezvous_handler
[36m(WorkerDict pid=26103)[0m     store = _create_c10d_store(
[36m(WorkerDict pid=26103)[0m             ^^^^^^^^^^^^^^^^^^^
[36m(WorkerDict pid=26103)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/distributed/rendezvous.py", line 198, in _create_c10d_store
[36m(WorkerDict pid=26103)[0m     return TCPStore(
[36m(WorkerDict pid=26103)[0m            ^^^^^^^^^
[36m(WorkerDict pid=26103)[0m RuntimeError: nonce == returnedNonce INTERNAL ASSERT FAILED at "/pytorch/torch/csrc/distributed/c10d/TCPStore.cpp":418, please report a bug to PyTorch. Ping failed, invalid nonce returned
[36m(WorkerDict pid=26103)[0m 
[36m(WorkerDict pid=26109)[0m 
[36m(WorkerDict pid=26109)[0m 
[36m(WorkerDict pid=26106)[0m [rank3]:[W1120 01:13:20.886572352 TCPStore.cpp:125] [c10d] recvValue failed on SocketImpl(fd=81, addr=[10-80-8-181.mllm-ppo-qwen-npsvc.mllm.svc.cluster.local]:51542, remote=[10-80-8-181.mllm-ppo-qwen-npsvc.mllm.svc.cluster.local]:55903): Connection reset by peer
[36m(WorkerDict pid=26106)[0m Exception raised from recvBytes at /pytorch/torch/csrc/distributed/c10d/Utils.hpp:675 (most recent call first):
[36m(WorkerDict pid=26106)[0m frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x98 (0x7fdcfc3785e8 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libc10.so)
[36m(WorkerDict pid=26106)[0m frame #1: <unknown function> + 0x5ba8bfe (0x7fbf896cdbfe in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
[36m(WorkerDict pid=26106)[0m frame #2: <unknown function> + 0x5baafcf (0x7fbf896cffcf in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
[36m(WorkerDict pid=26106)[0m frame #3: <unknown function> + 0x5bab84a (0x7fbf896d084a in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
[36m(WorkerDict pid=26106)[0m frame #4: c10d::TCPStore::check(std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > const&) + 0x2a9 (0x7fbf896ca2a9 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
[36m(WorkerDict pid=26106)[0m frame #5: c10d::ProcessGroupNCCL::heartbeatMonitor() + 0x379 (0x7fbf4adc99f9 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
[36m(WorkerDict pid=26106)[0m frame #6: <unknown function> + 0xdf0e6 (0x7fdd501610e6 in /opt/nas/p/conda/envs/verlaa/bin/../lib/libstdc++.so.6)
[36m(WorkerDict pid=26106)[0m frame #7: <unknown function> + 0x94ac3 (0x7fdd52999ac3 in /usr/lib/x86_64-linux-gnu/libc.so.6)
[36m(WorkerDict pid=26106)[0m frame #8: <unknown function> + 0x126850 (0x7fdd52a2b850 in /usr/lib/x86_64-linux-gnu/libc.so.6)
[36m(WorkerDict pid=26106)[0m 
[36m(WorkerDict pid=26106)[0m [rank3]:[W1120 01:13:20.892446889 ProcessGroupNCCL.cpp:1662] [PG ID 0 PG GUID 0(default_pg) Rank 3] Failed to check the "should dump" flag on TCPStore, (maybe TCPStore server has shut down too early), with error: Connection reset by peer
[36m(WorkerDict pid=26105)[0m /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.[32m [repeated 10x across cluster][0m
[36m(WorkerDict pid=26105)[0m   import pynvml  # type: ignore[import][32m [repeated 10x across cluster][0m
[36m(WorkerDict pid=26105)[0m [2025-11-20 01:13:05 TP1] Downcasting torch.float32 to torch.bfloat16.[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=26103)[0m [2025-11-20 01:13:05 TP0] Init torch distributed begin.[32m [repeated 3x across cluster][0m
[36m(WorkerDict pid=26109)[0m [2025-11-20 01:13:05 TP1] Scheduler hit an exception: Traceback (most recent call last):[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=26109)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/sglang/srt/managers/scheduler.py", line 2534, in run_scheduler_process[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=26109)[0m     scheduler = Scheduler([32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=26109)[0m                 ^^^^^^^^^^[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=26109)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/sglang/srt/model_executor/model_runner.py", line 235, in __init__[32m [repeated 8x across cluster][0m
[36m(WorkerDict pid=26109)[0m     self.tp_worker = TpWorkerClass([32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=26109)[0m                   ^^^^^^^^^^^^^^[32m [repeated 4x across cluster][0m
[36m(WorkerDict pid=26109)[0m     self.worker = TpModelWorker([32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=26109)[0m     self.model_runner = ModelRunner([32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=26109)[0m                         ^^^^^^^^^^^^[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=26109)[0m     min_per_gpu_memory = self.init_torch_distributed()[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=26109)[0m                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=26109)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/sglang/srt/model_executor/model_runner.py", line 551, in init_torch_distributed[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=26109)[0m     init_distributed_environment([32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=26109)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/sglang/srt/distributed/parallel_state.py", line 1255, in init_distributed_environment[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=26109)[0m     torch.distributed.init_process_group([32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=26109)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/distributed/c10d_logger.py", line 95, in wrapper[32m [repeated 4x across cluster][0m
[36m(WorkerDict pid=26109)[0m     return func(*args, **kwargs)[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=26109)[0m                   ^^^^^^^^^^^^^^^^^^^^^[32m [repeated 4x across cluster][0m
[36m(WorkerDict pid=26109)[0m     func_return = func(*args, **kwargs)[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=26109)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/distributed/distributed_c10d.py", line 1710, in init_process_group[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=26109)[0m     store, rank, world_size = next(rendezvous_iterator)[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=26109)[0m                               ^^^^^^^^^^^^^^^^^^^^^^^^^[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=26109)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/distributed/rendezvous.py", line 230, in _tcp_rendezvous_handler[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=26109)[0m     store = _create_c10d_store([32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=26109)[0m             ^^^^^^^^^^^^^^^^^^^[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=26109)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/distributed/rendezvous.py", line 198, in _create_c10d_store[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=26109)[0m     return TCPStore([32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=26109)[0m            ^^^^^^^^^[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=26109)[0m RuntimeError: nonce == returnedNonce INTERNAL ASSERT FAILED at "/pytorch/torch/csrc/distributed/c10d/TCPStore.cpp":418, please report a bug to PyTorch. Ping failed, invalid nonce returned[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=26107)[0m 
[36m(WorkerDict pid=26108)[0m 
[36m(WorkerDict pid=26104)[0m 
[36m(WorkerDict pid=26105)[0m 
[36m(WorkerDict pid=26110)[0m 

