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








+ ulimit -n 65535
++ pwd
+ PROJECT_DIR=/opt/nas/s/enyu/verl_askActively
+ CONFIG_PATH=/opt/nas/s/enyu/verl_askActively/examples/sglang_multiturn/config
+ TRAIN_BATCH_SIZE=8
+ MICRO_BATCH_SIZE=2
+ OFFLOAD=False
+ HYDRA_FULL_ERROR=1
+ python3 -m verl.trainer.main_ppo --config-path=/opt/nas/s/enyu/verl_askActively/examples/sglang_multiturn/config --config-name=askActively algorithm.adv_estimator=grpo data.train_batch_size=8 data.max_prompt_length=1024 data.max_response_length=3072 data.filter_overlong_prompts=True data.truncation=error data.return_raw_chat=True actor_rollout_ref.model.path=/opt/nas/s/enyu/verl_askActively/verl/models/Qwen3-4B-Instruct-2507-sft actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=8 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 actor_rollout_ref.rollout.tensor_model_parallel_size=2 actor_rollout_ref.rollout.name=sglang actor_rollout_ref.rollout.gpu_memory_utilization=0.7 actor_rollout_ref.rollout.n=8 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 actor_rollout_ref.ref.fsdp_config.param_offload=False algorithm.use_kl_in_reward=False trainer.critic_warmup=0 'trainer.logger=["console"]' trainer.project_name=askActively trainer.experiment_name=askActively-RL trainer.n_gpus_per_node=2 trainer.nnodes=1 trainer.save_freq=50 trainer.test_freq=20 trainer.rollout_data_dir=/opt/nas/s/enyu/verl_askActively/verl/rollout_data data.train_files=/opt/nas/s/enyu/verl_askActively/verl/data/rl/test_rl.parquet data.val_files=/opt/nas/s/enyu/verl_askActively/verl/data/rl/test_rl.parquet actor_rollout_ref.rollout.multi_turn.interaction_config_path=/opt/nas/s/enyu/verl_askActively/verl/trainer/config/interaction/askActively_interaction.yaml trainer.total_epochs=15
/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
2025-11-20 21:37:19,422	INFO worker.py:2003 -- Started a local Ray instance. View the dashboard at [1m[32mhttp://127.0.0.1:8265 [39m[22m
/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/ray/_private/worker.py:2051: FutureWarning: Tip: In future versions of Ray, Ray will no longer override accelerator visible devices env var if num_gpus=0 or num_gpus=None (default). To enable this behavior and turn off this error message, set RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
  warnings.warn(
[36m(pid=14058)[0m /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
[36m(pid=14058)[0m   import pynvml  # type: ignore[import]
[36m(TaskRunner pid=14058)[0m /opt/nas/s/enyu/verl_askActively/verl/trainer/main_ppo.py:258: UserWarning: Disabled critic as algorithm.adv_estimator != gae. If it is not intended, please set critic.enable=True
[36m(TaskRunner pid=14058)[0m   use_critic=need_critic(config),
[36m(TaskRunner pid=14058)[0m /opt/nas/s/enyu/verl_askActively/verl/utils/profiler/config.py:49: UserWarning: Torch profiler tool config is not fully supported now.
[36m(TaskRunner pid=14058)[0m   warnings.warn("Torch profiler tool config is not fully supported now.", stacklevel=1)
[36m(TaskRunner pid=14058)[0m Setting TOKENIZERS_PARALLELISM=false for forked processes.
[36m(TaskRunner pid=14058)[0m WARNING:2025-11-20 21:39:25,290:Setting TOKENIZERS_PARALLELISM=false for forked processes.
[36m(TaskRunner pid=14058)[0m 
Filtering prompts longer than 1024 tokens (num_proc=1):   0%|          | 0/2329 [00:00<?, ? examples/s]
[36m(TaskRunner pid=14058)[0m 
Filtering prompts longer than 1024 tokens (num_proc=1):  43%|████▎     | 1000/2329 [00:03<00:04, 319.23 examples/s]
[36m(TaskRunner pid=14058)[0m 
Filtering prompts longer than 1024 tokens (num_proc=1):  86%|████████▌ | 2000/2329 [00:05<00:00, 375.19 examples/s]
[36m(TaskRunner pid=14058)[0m 
Filtering prompts longer than 1024 tokens (num_proc=1): 100%|██████████| 2329/2329 [00:06<00:00, 387.12 examples/s]
[36m(TaskRunner pid=14058)[0m 
Filtering prompts longer than 1024 tokens (num_proc=1): 100%|██████████| 2329/2329 [00:06<00:00, 366.44 examples/s]
[36m(TaskRunner pid=14058)[0m Setting TOKENIZERS_PARALLELISM=false for forked processes.
[36m(TaskRunner pid=14058)[0m WARNING:2025-11-20 21:39:56,033:Setting TOKENIZERS_PARALLELISM=false for forked processes.
[36m(TaskRunner pid=14058)[0m /opt/nas/s/enyu/verl_askActively/verl/trainer/ppo/ray_trainer.py:339: UserWarning: Disabled critic as algorithm.adv_estimator != gae. If it is not intended, please set critic.enable=True
[36m(TaskRunner pid=14058)[0m   self.use_critic = need_critic(self.config)
[36m(pid=26071)[0m /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
[36m(pid=26071)[0m   import pynvml  # type: ignore[import]
[36m(WorkerDict pid=26071)[0m Flash Attention 2 only supports torch.float16 and torch.bfloat16 dtypes, but the current dype in Qwen3ForCausalLM is torch.float32. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator, or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`
[36m(WorkerDict pid=26071)[0m Flash Attention 2 only supports torch.float16 and torch.bfloat16 dtypes, but the current dype in Qwen3Model is torch.float32. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator, or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`
[36m(pid=26070)[0m /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
[36m(pid=26070)[0m   import pynvml  # type: ignore[import]
[36m(WorkerDict pid=26071)[0m 
Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]
[36m(WorkerDict pid=26071)[0m 
Loading checkpoint shards:  50%|█████     | 1/2 [00:32<00:32, 32.83s/it]
[36m(WorkerDict pid=26070)[0m Flash Attention 2 only supports torch.float16 and torch.bfloat16 dtypes, but the current dype in Qwen3Model is torch.float32. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator, or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`[32m [repeated 2x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)[0m
[36m(WorkerDict pid=26070)[0m 
Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]
[36m(WorkerDict pid=26071)[0m 
Loading checkpoint shards: 100%|██████████| 2/2 [01:03<00:00, 31.60s/it]
Loading checkpoint shards: 100%|██████████| 2/2 [01:03<00:00, 31.79s/it]
[36m(WorkerDict pid=26070)[0m 
Loading checkpoint shards:  50%|█████     | 1/2 [00:33<00:33, 33.05s/it]
[36m(WorkerDict pid=26071)[0m /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
[36m(WorkerDict pid=26071)[0m   import pynvml  # type: ignore[import]
[36m(WorkerDict pid=26070)[0m 
Loading checkpoint shards: 100%|██████████| 2/2 [01:03<00:00, 31.64s/it]
Loading checkpoint shards: 100%|██████████| 2/2 [01:03<00:00, 31.85s/it]
[36m(WorkerDict pid=26071)[0m 
Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]
[36m(WorkerDict pid=26071)[0m 
Loading checkpoint shards: 100%|██████████| 2/2 [00:00<00:00, 15.33it/s]
Loading checkpoint shards: 100%|██████████| 2/2 [00:00<00:00, 15.30it/s]
[36m(WorkerDict pid=26071)[0m /opt/nas/s/enyu/verl_askActively/verl/utils/profiler/config.py:49: UserWarning: Torch profiler tool config is not fully supported now.
[36m(WorkerDict pid=26071)[0m   warnings.warn("Torch profiler tool config is not fully supported now.", stacklevel=1)
[36m(WorkerDict pid=26071)[0m /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:680: FutureWarning: FSDP.state_dict_type() and FSDP.set_state_dict_type() are being deprecated. Please use APIs, get_state_dict() and set_state_dict(), which can support different parallelisms, FSDP1, FSDP2, DDP. API doc: https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict .Tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html .
[36m(WorkerDict pid=26071)[0m   warnings.warn(
[36m(WorkerDict pid=26070)[0m /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
[36m(WorkerDict pid=26070)[0m   import pynvml  # type: ignore[import]
[36m(WorkerDict pid=26070)[0m 
Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]
[36m(WorkerDict pid=26070)[0m 
Loading checkpoint shards: 100%|██████████| 2/2 [00:00<00:00, 15.72it/s]
Loading checkpoint shards: 100%|██████████| 2/2 [00:00<00:00, 15.69it/s]
[36m(WorkerDict pid=26070)[0m /opt/nas/s/enyu/verl_askActively/verl/utils/profiler/config.py:49: UserWarning: Torch profiler tool config is not fully supported now.
[36m(WorkerDict pid=26070)[0m   warnings.warn("Torch profiler tool config is not fully supported now.", stacklevel=1)
[36m(WorkerDict pid=26070)[0m [2025-11-20 21:43:02] Downcasting torch.float32 to torch.bfloat16.
[36m(WorkerDict pid=26070)[0m [2025-11-20 21:43:02] Using default HuggingFace chat template with detected content format: string
[36m(WorkerDict pid=26070)[0m /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
[36m(WorkerDict pid=26070)[0m   import pynvml  # type: ignore[import]
[36m(WorkerDict pid=26070)[0m /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
[36m(WorkerDict pid=26070)[0m   import pynvml  # type: ignore[import]
[36m(WorkerDict pid=26070)[0m /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
[36m(WorkerDict pid=26070)[0m   import pynvml  # type: ignore[import]
[36m(WorkerDict pid=26070)[0m /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
[36m(WorkerDict pid=26070)[0m   import pynvml  # type: ignore[import]
[36m(WorkerDict pid=26070)[0m /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
[36m(WorkerDict pid=26070)[0m   import pynvml  # type: ignore[import]
[36m(WorkerDict pid=26070)[0m /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
[36m(WorkerDict pid=26070)[0m   import pynvml  # type: ignore[import]
[36m(WorkerDict pid=26070)[0m [2025-11-20 21:43:52 TP0] Downcasting torch.float32 to torch.bfloat16.
[36m(WorkerDict pid=26070)[0m [2025-11-20 21:43:52 TP1] Downcasting torch.float32 to torch.bfloat16.
[36m(WorkerDict pid=26070)[0m [2025-11-20 21:43:53 TP1] Downcasting torch.float32 to torch.bfloat16.
[36m(WorkerDict pid=26070)[0m [2025-11-20 21:43:53 TP0] Downcasting torch.float32 to torch.bfloat16.
[36m(WorkerDict pid=26070)[0m [2025-11-20 21:43:53 TP0] Init torch distributed begin.
[36m(WorkerDict pid=26070)[0m [E1120 21:53:06.140024850 socket.cpp:1019] [c10d] The client socket has timed out after 600000ms while trying to connect to (127.0.0.1, 30349).
[36m(WorkerDict pid=26070)[0m [W1120 21:53:06.222395820 TCPStore.cpp:343] [c10d] TCP client failed to connect/validate to host 127.0.0.1:30349 - retrying (try=0, timeout=600000ms, delay=48742ms): The client socket has timed out after 600000ms while trying to connect to (127.0.0.1, 30349).
[36m(WorkerDict pid=26070)[0m Exception raised from throwTimeoutError at /pytorch/torch/csrc/distributed/c10d/socket.cpp:1021 (most recent call first):
[36m(WorkerDict pid=26070)[0m frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x98 (0x7f8d319785e8 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libc10.so)
[36m(WorkerDict pid=26070)[0m frame #1: <unknown function> + 0x5ba8bfe (0x7f8d1b0cdbfe in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
[36m(WorkerDict pid=26070)[0m frame #2: <unknown function> + 0x136920d (0x7f8d1688e20d in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
[36m(WorkerDict pid=26070)[0m frame #3: <unknown function> + 0x5bf5791 (0x7f8d1b11a791 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
[36m(WorkerDict pid=26070)[0m frame #4: <unknown function> + 0x5bf5949 (0x7f8d1b11a949 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
[36m(WorkerDict pid=26070)[0m frame #5: <unknown function> + 0x5bf5d01 (0x7f8d1b11ad01 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
[36m(WorkerDict pid=26070)[0m frame #6: <unknown function> + 0x5ba3eeb (0x7f8d1b0c8eeb in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
[36m(WorkerDict pid=26070)[0m frame #7: c10d::TCPStore::TCPStore(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, c10d::TCPStoreOptions const&) + 0x4b5 (0x7f8d1b0cb7f5 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
[36m(WorkerDict pid=26070)[0m frame #8: <unknown function> + 0xc20a05 (0x7f8d29f6ea05 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_python.so)
[36m(WorkerDict pid=26070)[0m frame #9: <unknown function> + 0xc55324 (0x7f8d29fa3324 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_python.so)
[36m(WorkerDict pid=26070)[0m frame #10: <unknown function> + 0x38a1ac (0x7f8d296d81ac in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_python.so)
[36m(WorkerDict pid=26070)[0m frame #11: sglang::scheduler_TP1() [0x528657]
[36m(WorkerDict pid=26070)[0m frame #12: _PyObject_MakeTpCall + 0x26c (0x50400c in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #13: sglang::scheduler_TP1() [0x557a1e]
[36m(WorkerDict pid=26070)[0m frame #14: _PyObject_Call + 0x11f (0x54300f in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #15: sglang::scheduler_TP1() [0x54054e]
[36m(WorkerDict pid=26070)[0m frame #16: sglang::scheduler_TP1() [0x50440c]
[36m(WorkerDict pid=26070)[0m frame #17: <unknown function> + 0x388d2b (0x7f8d296d6d2b in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_python.so)
[36m(WorkerDict pid=26070)[0m frame #18: _PyObject_MakeTpCall + 0x26c (0x50400c in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #19: _PyEval_EvalFrameDefault + 0x6a6 (0x511616 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #20: sglang::scheduler_TP1() [0x5a31d7]
[36m(WorkerDict pid=26070)[0m frame #21: sglang::scheduler_TP1() [0x52ee4b]
[36m(WorkerDict pid=26070)[0m frame #22: PyObject_Vectorcall + 0x31 (0x51e5f1 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #23: _PyEval_EvalFrameDefault + 0x6a6 (0x511616 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #24: _PyFunction_Vectorcall + 0x173 (0x538d03 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #25: PyObject_Call + 0x12c (0x542dac in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #26: _PyEval_EvalFrameDefault + 0x484b (0x5157bb in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #27: _PyFunction_Vectorcall + 0x173 (0x538d03 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #28: PyObject_Call + 0x12c (0x542dac in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #29: _PyEval_EvalFrameDefault + 0x484b (0x5157bb in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #30: _PyFunction_Vectorcall + 0x173 (0x538d03 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #31: sglang::scheduler_TP1() [0x540284]
[36m(WorkerDict pid=26070)[0m frame #32: _PyObject_MakeTpCall + 0x233 (0x503fd3 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #33: _PyEval_EvalFrameDefault + 0x6a6 (0x511616 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #34: _PyFunction_Vectorcall + 0x173 (0x538d03 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #35: sglang::scheduler_TP1() [0x5401c9]
[36m(WorkerDict pid=26070)[0m frame #36: _PyObject_MakeTpCall + 0x233 (0x503fd3 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #37: _PyEval_EvalFrameDefault + 0x6a6 (0x511616 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #38: _PyFunction_Vectorcall + 0x173 (0x538d03 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #39: sglang::scheduler_TP1() [0x540284]
[36m(WorkerDict pid=26070)[0m frame #40: _PyObject_MakeTpCall + 0x233 (0x503fd3 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #41: _PyEval_EvalFrameDefault + 0x6a6 (0x511616 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #42: _PyFunction_Vectorcall + 0x173 (0x538d03 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #43: sglang::scheduler_TP1() [0x540284]
[36m(WorkerDict pid=26070)[0m frame #44: _PyObject_MakeTpCall + 0x233 (0x503fd3 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #45: _PyEval_EvalFrameDefault + 0x6a6 (0x511616 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #46: _PyFunction_Vectorcall + 0x173 (0x538d03 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #47: _PyEval_EvalFrameDefault + 0x484b (0x5157bb in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #48: sglang::scheduler_TP1() [0x5cc38a]
[36m(WorkerDict pid=26070)[0m frame #49: PyEval_EvalCode + 0x9f (0x5cba5f in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #50: sglang::scheduler_TP1() [0x5ec977]
[36m(WorkerDict pid=26070)[0m frame #51: sglang::scheduler_TP1() [0x5e8510]
[36m(WorkerDict pid=26070)[0m frame #52: PyRun_StringFlags + 0x5f (0x5db0bf in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #53: PyRun_SimpleStringFlags + 0x3b (0x5dae6b in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #54: Py_RunMain + 0x388 (0x5f71a8 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #55: Py_BytesMain + 0x39 (0x5bbf49 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #56: <unknown function> + 0x29d90 (0x7f8d3d022d90 in /usr/lib/x86_64-linux-gnu/libc.so.6)
[36m(WorkerDict pid=26070)[0m frame #57: __libc_start_main + 0x80 (0x7f8d3d022e40 in /usr/lib/x86_64-linux-gnu/libc.so.6)
[36m(WorkerDict pid=26070)[0m frame #58: sglang::scheduler_TP1() [0x5bbd97]
[36m(WorkerDict pid=26070)[0m 
[36m(WorkerDict pid=26070)[0m [E1120 21:53:47.484870519 socket.cpp:1019] [c10d] The client socket has timed out after 600000ms while trying to connect to (127.0.0.1, 30349).
[36m(WorkerDict pid=26070)[0m [W1120 21:53:47.485490263 TCPStore.cpp:343] [c10d] TCP client failed to connect/validate to host 127.0.0.1:30349 - retrying (try=0, timeout=600000ms, delay=78303ms): The client socket has timed out after 600000ms while trying to connect to (127.0.0.1, 30349).
[36m(WorkerDict pid=26070)[0m Exception raised from throwTimeoutError at /pytorch/torch/csrc/distributed/c10d/socket.cpp:1021 (most recent call first):
[36m(WorkerDict pid=26070)[0m frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x98 (0x7f10d3cc75e8 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libc10.so)
[36m(WorkerDict pid=26070)[0m frame #1: <unknown function> + 0x5ba8bfe (0x7f10bd4cdbfe in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
[36m(WorkerDict pid=26070)[0m frame #2: <unknown function> + 0x136920d (0x7f10b8c8e20d in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
[36m(WorkerDict pid=26070)[0m frame #3: <unknown function> + 0x5bf5791 (0x7f10bd51a791 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
[36m(WorkerDict pid=26070)[0m frame #4: <unknown function> + 0x5bf5949 (0x7f10bd51a949 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
[36m(WorkerDict pid=26070)[0m frame #5: <unknown function> + 0x5bf5d01 (0x7f10bd51ad01 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
[36m(WorkerDict pid=26070)[0m frame #6: <unknown function> + 0x5ba3eeb (0x7f10bd4c8eeb in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
[36m(WorkerDict pid=26070)[0m frame #7: c10d::TCPStore::TCPStore(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, c10d::TCPStoreOptions const&) + 0x4b5 (0x7f10bd4cb7f5 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
[36m(WorkerDict pid=26070)[0m frame #8: <unknown function> + 0xc20a05 (0x7f10cc36ea05 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_python.so)
[36m(WorkerDict pid=26070)[0m frame #9: <unknown function> + 0xc55324 (0x7f10cc3a3324 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_python.so)
[36m(WorkerDict pid=26070)[0m frame #10: <unknown function> + 0x38a1ac (0x7f10cbad81ac in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_python.so)
[36m(WorkerDict pid=26070)[0m frame #11: sglang::scheduler_TP0() [0x528657]
[36m(WorkerDict pid=26070)[0m frame #12: _PyObject_MakeTpCall + 0x26c (0x50400c in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #13: sglang::scheduler_TP0() [0x557a1e]
[36m(WorkerDict pid=26070)[0m frame #14: _PyObject_Call + 0x11f (0x54300f in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #15: sglang::scheduler_TP0() [0x54054e]
[36m(WorkerDict pid=26070)[0m frame #16: sglang::scheduler_TP0() [0x50440c]
[36m(WorkerDict pid=26070)[0m frame #17: <unknown function> + 0x388d2b (0x7f10cbad6d2b in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_python.so)
[36m(WorkerDict pid=26070)[0m frame #18: _PyObject_MakeTpCall + 0x26c (0x50400c in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #19: _PyEval_EvalFrameDefault + 0x6a6 (0x511616 in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #20: sglang::scheduler_TP0() [0x5a31d7]
[36m(WorkerDict pid=26070)[0m frame #21: sglang::scheduler_TP0() [0x52ee4b]
[36m(WorkerDict pid=26070)[0m frame #22: PyObject_Vectorcall + 0x31 (0x51e5f1 in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #23: _PyEval_EvalFrameDefault + 0x6a6 (0x511616 in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #24: _PyFunction_Vectorcall + 0x173 (0x538d03 in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #25: PyObject_Call + 0x12c (0x542dac in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #26: _PyEval_EvalFrameDefault + 0x484b (0x5157bb in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #27: _PyFunction_Vectorcall + 0x173 (0x538d03 in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #28: PyObject_Call + 0x12c (0x542dac in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #29: _PyEval_EvalFrameDefault + 0x484b (0x5157bb in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #30: _PyFunction_Vectorcall + 0x173 (0x538d03 in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #31: sglang::scheduler_TP0() [0x540284]
[36m(WorkerDict pid=26070)[0m frame #32: _PyObject_MakeTpCall + 0x233 (0x503fd3 in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #33: _PyEval_EvalFrameDefault + 0x6a6 (0x511616 in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #34: _PyFunction_Vectorcall + 0x173 (0x538d03 in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #35: sglang::scheduler_TP0() [0x5401c9]
[36m(WorkerDict pid=26070)[0m frame #36: _PyObject_MakeTpCall + 0x233 (0x503fd3 in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #37: _PyEval_EvalFrameDefault + 0x6a6 (0x511616 in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #38: _PyFunction_Vectorcall + 0x173 (0x538d03 in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #39: sglang::scheduler_TP0() [0x540284]
[36m(WorkerDict pid=26070)[0m frame #40: _PyObject_MakeTpCall + 0x233 (0x503fd3 in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #41: _PyEval_EvalFrameDefault + 0x6a6 (0x511616 in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #42: _PyFunction_Vectorcall + 0x173 (0x538d03 in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #43: sglang::scheduler_TP0() [0x540284]
[36m(WorkerDict pid=26070)[0m frame #44: _PyObject_MakeTpCall + 0x233 (0x503fd3 in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #45: _PyEval_EvalFrameDefault + 0x6a6 (0x511616 in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #46: _PyFunction_Vectorcall + 0x173 (0x538d03 in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #47: _PyEval_EvalFrameDefault + 0x484b (0x5157bb in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #48: sglang::scheduler_TP0() [0x5cc38a]
[36m(WorkerDict pid=26070)[0m frame #49: PyEval_EvalCode + 0x9f (0x5cba5f in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #50: sglang::scheduler_TP0() [0x5ec977]
[36m(WorkerDict pid=26070)[0m frame #51: sglang::scheduler_TP0() [0x5e8510]
[36m(WorkerDict pid=26070)[0m frame #52: PyRun_StringFlags + 0x5f (0x5db0bf in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #53: PyRun_SimpleStringFlags + 0x3b (0x5dae6b in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #54: Py_RunMain + 0x388 (0x5f71a8 in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #55: Py_BytesMain + 0x39 (0x5bbf49 in sglang::scheduler_TP0)
[36m(WorkerDict pid=26070)[0m frame #56: <unknown function> + 0x29d90 (0x7f10df2e5d90 in /usr/lib/x86_64-linux-gnu/libc.so.6)
[36m(WorkerDict pid=26070)[0m frame #57: __libc_start_main + 0x80 (0x7f10df2e5e40 in /usr/lib/x86_64-linux-gnu/libc.so.6)
[36m(WorkerDict pid=26070)[0m frame #58: sglang::scheduler_TP0() [0x5bbd97]
[36m(WorkerDict pid=26070)[0m 
[36m(WorkerDict pid=26070)[0m [E1120 22:03:17.533304280 socket.cpp:1019] [c10d] The client socket has timed out after 600000ms while trying to connect to (127.0.0.1, 30349).
[36m(WorkerDict pid=26070)[0m [E1120 22:03:17.533539625 TCPStore.cpp:331] [c10d] TCP client failed to connect/validate to host 127.0.0.1:30349 - timed out (try=1, timeout=600000ms): The client socket has timed out after 600000ms while trying to connect to (127.0.0.1, 30349).
[36m(WorkerDict pid=26070)[0m Exception raised from throwTimeoutError at /pytorch/torch/csrc/distributed/c10d/socket.cpp:1021 (most recent call first):
[36m(WorkerDict pid=26070)[0m frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x98 (0x7f8d319785e8 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libc10.so)
[36m(WorkerDict pid=26070)[0m frame #1: <unknown function> + 0x5ba8bfe (0x7f8d1b0cdbfe in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
[36m(WorkerDict pid=26070)[0m frame #2: <unknown function> + 0x136920d (0x7f8d1688e20d in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
[36m(WorkerDict pid=26070)[0m frame #3: <unknown function> + 0x5bf5791 (0x7f8d1b11a791 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
[36m(WorkerDict pid=26070)[0m frame #4: <unknown function> + 0x5bf5949 (0x7f8d1b11a949 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
[36m(WorkerDict pid=26070)[0m frame #5: <unknown function> + 0x5bf5d01 (0x7f8d1b11ad01 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
[36m(WorkerDict pid=26070)[0m frame #6: <unknown function> + 0x5ba3eeb (0x7f8d1b0c8eeb in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
[36m(WorkerDict pid=26070)[0m frame #7: c10d::TCPStore::TCPStore(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, c10d::TCPStoreOptions const&) + 0x4b5 (0x7f8d1b0cb7f5 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
[36m(WorkerDict pid=26070)[0m frame #8: <unknown function> + 0xc20a05 (0x7f8d29f6ea05 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_python.so)
[36m(WorkerDict pid=26070)[0m frame #9: <unknown function> + 0xc55324 (0x7f8d29fa3324 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_python.so)
[36m(WorkerDict pid=26070)[0m frame #10: <unknown function> + 0x38a1ac (0x7f8d296d81ac in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_python.so)
[36m(WorkerDict pid=26070)[0m frame #11: sglang::scheduler_TP1() [0x528657]
[36m(WorkerDict pid=26070)[0m frame #12: _PyObject_MakeTpCall + 0x26c (0x50400c in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #13: sglang::scheduler_TP1() [0x557a1e]
[36m(WorkerDict pid=26070)[0m frame #14: _PyObject_Call + 0x11f (0x54300f in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #15: sglang::scheduler_TP1() [0x54054e]
[36m(WorkerDict pid=26070)[0m frame #16: sglang::scheduler_TP1() [0x50440c]
[36m(WorkerDict pid=26070)[0m frame #17: <unknown function> + 0x388d2b (0x7f8d296d6d2b in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_python.so)
[36m(WorkerDict pid=26070)[0m frame #18: _PyObject_MakeTpCall + 0x26c (0x50400c in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #19: _PyEval_EvalFrameDefault + 0x6a6 (0x511616 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #20: sglang::scheduler_TP1() [0x5a31d7]
[36m(WorkerDict pid=26070)[0m frame #21: sglang::scheduler_TP1() [0x52ee4b]
[36m(WorkerDict pid=26070)[0m frame #22: PyObject_Vectorcall + 0x31 (0x51e5f1 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #23: _PyEval_EvalFrameDefault + 0x6a6 (0x511616 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #24: _PyFunction_Vectorcall + 0x173 (0x538d03 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #25: PyObject_Call + 0x12c (0x542dac in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #26: _PyEval_EvalFrameDefault + 0x484b (0x5157bb in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #27: _PyFunction_Vectorcall + 0x173 (0x538d03 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #28: PyObject_Call + 0x12c (0x542dac in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #29: _PyEval_EvalFrameDefault + 0x484b (0x5157bb in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #30: _PyFunction_Vectorcall + 0x173 (0x538d03 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #31: sglang::scheduler_TP1() [0x540284]
[36m(WorkerDict pid=26070)[0m frame #32: _PyObject_MakeTpCall + 0x233 (0x503fd3 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #33: _PyEval_EvalFrameDefault + 0x6a6 (0x511616 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #34: _PyFunction_Vectorcall + 0x173 (0x538d03 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #35: sglang::scheduler_TP1() [0x5401c9]
[36m(WorkerDict pid=26070)[0m frame #36: _PyObject_MakeTpCall + 0x233 (0x503fd3 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #37: _PyEval_EvalFrameDefault + 0x6a6 (0x511616 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #38: _PyFunction_Vectorcall + 0x173 (0x538d03 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #39: sglang::scheduler_TP1() [0x540284]
[36m(WorkerDict pid=26070)[0m frame #40: _PyObject_MakeTpCall + 0x233 (0x503fd3 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #41: _PyEval_EvalFrameDefault + 0x6a6 (0x511616 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #42: _PyFunction_Vectorcall + 0x173 (0x538d03 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #43: sglang::scheduler_TP1() [0x540284]
[36m(WorkerDict pid=26070)[0m frame #44: _PyObject_MakeTpCall + 0x233 (0x503fd3 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #45: _PyEval_EvalFrameDefault + 0x6a6 (0x511616 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #46: _PyFunction_Vectorcall + 0x173 (0x538d03 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #47: _PyEval_EvalFrameDefault + 0x484b (0x5157bb in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #48: sglang::scheduler_TP1() [0x5cc38a]
[36m(WorkerDict pid=26070)[0m frame #49: PyEval_EvalCode + 0x9f (0x5cba5f in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #50: sglang::scheduler_TP1() [0x5ec977]
[36m(WorkerDict pid=26070)[0m frame #51: sglang::scheduler_TP1() [0x5e8510]
[36m(WorkerDict pid=26070)[0m frame #52: PyRun_StringFlags + 0x5f (0x5db0bf in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #53: PyRun_SimpleStringFlags + 0x3b (0x5dae6b in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #54: Py_RunMain + 0x388 (0x5f71a8 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #55: Py_BytesMain + 0x39 (0x5bbf49 in sglang::scheduler_TP1)
[36m(WorkerDict pid=26070)[0m frame #56: <unknown function> + 0x29d90 (0x7f8d3d022d90 in /usr/lib/x86_64-linux-gnu/libc.so.6)
[36m(WorkerDict pid=26070)[0m frame #57: __libc_start_main + 0x80 (0x7f8d3d022e40 in /usr/lib/x86_64-linux-gnu/libc.so.6)
[36m(WorkerDict pid=26070)[0m frame #58: sglang::scheduler_TP1() [0x5bbd97]
[36m(WorkerDict pid=26070)[0m 
[36m(WorkerDict pid=26070)[0m [2025-11-20 22:03:17 TP1] Scheduler hit an exception: Traceback (most recent call last):
[36m(WorkerDict pid=26070)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/sglang/srt/managers/scheduler.py", line 2534, in run_scheduler_process
[36m(WorkerDict pid=26070)[0m     scheduler = Scheduler(
[36m(WorkerDict pid=26070)[0m                 ^^^^^^^^^^
[36m(WorkerDict pid=26070)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/sglang/srt/managers/scheduler.py", line 313, in __init__
[36m(WorkerDict pid=26070)[0m     self.tp_worker = TpWorkerClass(
[36m(WorkerDict pid=26070)[0m                      ^^^^^^^^^^^^^^
[36m(WorkerDict pid=26070)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/sglang/srt/managers/tp_worker_overlap_thread.py", line 67, in __init__
[36m(WorkerDict pid=26070)[0m     self.worker = TpModelWorker(
[36m(WorkerDict pid=26070)[0m                   ^^^^^^^^^^^^^^
[36m(WorkerDict pid=26070)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/sglang/srt/managers/tp_worker.py", line 84, in __init__
[36m(WorkerDict pid=26070)[0m     self.model_runner = ModelRunner(
[36m(WorkerDict pid=26070)[0m                         ^^^^^^^^^^^^
[36m(WorkerDict pid=26070)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/sglang/srt/model_executor/model_runner.py", line 235, in __init__
[36m(WorkerDict pid=26070)[0m     min_per_gpu_memory = self.init_torch_distributed()
[36m(WorkerDict pid=26070)[0m                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(WorkerDict pid=26070)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/sglang/srt/model_executor/model_runner.py", line 551, in init_torch_distributed
[36m(WorkerDict pid=26070)[0m     init_distributed_environment(
[36m(WorkerDict pid=26070)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/sglang/srt/distributed/parallel_state.py", line 1255, in init_distributed_environment
[36m(WorkerDict pid=26070)[0m     torch.distributed.init_process_group(
[36m(WorkerDict pid=26070)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/distributed/c10d_logger.py", line 81, in wrapper
[36m(WorkerDict pid=26070)[0m     return func(*args, **kwargs)
[36m(WorkerDict pid=26070)[0m            ^^^^^^^^^^^^^^^^^^^^^
[36m(WorkerDict pid=26070)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/distributed/c10d_logger.py", line 95, in wrapper
[36m(WorkerDict pid=26070)[0m     func_return = func(*args, **kwargs)
[36m(WorkerDict pid=26070)[0m                   ^^^^^^^^^^^^^^^^^^^^^
[36m(WorkerDict pid=26070)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/distributed/distributed_c10d.py", line 1710, in init_process_group
[36m(WorkerDict pid=26070)[0m     store, rank, world_size = next(rendezvous_iterator)
[36m(WorkerDict pid=26070)[0m                               ^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(WorkerDict pid=26070)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/distributed/rendezvous.py", line 230, in _tcp_rendezvous_handler
[36m(WorkerDict pid=26070)[0m     store = _create_c10d_store(
[36m(WorkerDict pid=26070)[0m             ^^^^^^^^^^^^^^^^^^^
[36m(WorkerDict pid=26070)[0m   File "/opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/distributed/rendezvous.py", line 198, in _create_c10d_store
[36m(WorkerDict pid=26070)[0m     return TCPStore(
[36m(WorkerDict pid=26070)[0m            ^^^^^^^^^
[36m(WorkerDict pid=26070)[0m torch.distributed.DistNetworkError: The client socket has timed out after 600000ms while trying to connect to (127.0.0.1, 30349).
[36m(WorkerDict pid=26070)[0m 
ray init kwargs: {'num_cpus': None, 'runtime_env': {'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN', 'VLLM_LOGGING_LEVEL': 'WARN', 'VLLM_ALLOW_RUNTIME_LORA_UPDATING': 'true', 'CUDA_DEVICE_MAX_CONNECTIONS': '1', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:False'}, 'working_dir': None}}
[36m(TaskRunner pid=14058)[0m TaskRunner hostname: VM-1-37-tencentos, PID: 14058
[36m(TaskRunner pid=14058)[0m {'actor_rollout_ref': {'actor': {'_target_': 'verl.workers.config.FSDPActorConfig',
[36m(TaskRunner pid=14058)[0m                                  'checkpoint': {'_target_': 'verl.trainer.config.CheckpointConfig',
[36m(TaskRunner pid=14058)[0m                                                 'async_save': False,
[36m(TaskRunner pid=14058)[0m                                                 'load_contents': ['model',
[36m(TaskRunner pid=14058)[0m                                                                   'optimizer',
[36m(TaskRunner pid=14058)[0m                                                                   'extra'],
[36m(TaskRunner pid=14058)[0m                                                 'save_contents': ['model',
[36m(TaskRunner pid=14058)[0m                                                                   'optimizer',
[36m(TaskRunner pid=14058)[0m                                                                   'extra']},
[36m(TaskRunner pid=14058)[0m                                  'clip_ratio': 0.2,
[36m(TaskRunner pid=14058)[0m                                  'clip_ratio_c': 3.0,
[36m(TaskRunner pid=14058)[0m                                  'clip_ratio_high': 0.2,
[36m(TaskRunner pid=14058)[0m                                  'clip_ratio_low': 0.2,
[36m(TaskRunner pid=14058)[0m                                  'entropy_checkpointing': False,
[36m(TaskRunner pid=14058)[0m                                  'entropy_coeff': 0,
[36m(TaskRunner pid=14058)[0m                                  'entropy_from_logits_with_chunking': False,
[36m(TaskRunner pid=14058)[0m                                  'fsdp_config': {'_target_': 'verl.workers.config.FSDPEngineConfig',
[36m(TaskRunner pid=14058)[0m                                                  'entropy_checkpointing': False,
[36m(TaskRunner pid=14058)[0m                                                  'entropy_from_logits_with_chunking': False,
[36m(TaskRunner pid=14058)[0m                                                  'forward_only': False,
[36m(TaskRunner pid=14058)[0m                                                  'forward_prefetch': False,
[36m(TaskRunner pid=14058)[0m                                                  'fsdp_size': -1,
[36m(TaskRunner pid=14058)[0m                                                  'model_dtype': 'bfloat16',
[36m(TaskRunner pid=14058)[0m                                                  'offload_policy': False,
[36m(TaskRunner pid=14058)[0m                                                  'optimizer_offload': False,
[36m(TaskRunner pid=14058)[0m                                                  'param_offload': False,
[36m(TaskRunner pid=14058)[0m                                                  'reshard_after_forward': True,
[36m(TaskRunner pid=14058)[0m                                                  'strategy': 'fsdp',
[36m(TaskRunner pid=14058)[0m                                                  'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=14058)[0m                                                  'use_orig_params': False,
[36m(TaskRunner pid=14058)[0m                                                  'use_torch_compile': True,
[36m(TaskRunner pid=14058)[0m                                                  'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=14058)[0m                                  'grad_clip': 1.0,
[36m(TaskRunner pid=14058)[0m                                  'kl_loss_coef': 0.001,
[36m(TaskRunner pid=14058)[0m                                  'kl_loss_type': 'low_var_kl',
[36m(TaskRunner pid=14058)[0m                                  'loss_agg_mode': 'token-mean',
[36m(TaskRunner pid=14058)[0m                                  'optim': {'_target_': 'verl.workers.config.FSDPOptimizerConfig',
[36m(TaskRunner pid=14058)[0m                                            'betas': [0.9, 0.999],
[36m(TaskRunner pid=14058)[0m                                            'clip_grad': 1.0,
[36m(TaskRunner pid=14058)[0m                                            'lr': 1e-06,
[36m(TaskRunner pid=14058)[0m                                            'lr_warmup_steps': -1,
[36m(TaskRunner pid=14058)[0m                                            'lr_warmup_steps_ratio': 0.0,
[36m(TaskRunner pid=14058)[0m                                            'min_lr_ratio': 0.0,
[36m(TaskRunner pid=14058)[0m                                            'num_cycles': 0.5,
[36m(TaskRunner pid=14058)[0m                                            'total_training_steps': -1,
[36m(TaskRunner pid=14058)[0m                                            'warmup_style': 'constant',
[36m(TaskRunner pid=14058)[0m                                            'weight_decay': 0.01},
[36m(TaskRunner pid=14058)[0m                                  'policy_loss': {'_target_': 'verl.workers.config.PolicyLossConfig',
[36m(TaskRunner pid=14058)[0m                                                  'clip_cov_lb': 1.0,
[36m(TaskRunner pid=14058)[0m                                                  'clip_cov_ratio': 0.0002,
[36m(TaskRunner pid=14058)[0m                                                  'clip_cov_ub': 5.0,
[36m(TaskRunner pid=14058)[0m                                                  'kl_cov_ratio': 0.0002,
[36m(TaskRunner pid=14058)[0m                                                  'loss_mode': 'vanilla',
[36m(TaskRunner pid=14058)[0m                                                  'ppo_kl_coef': 0.1},
[36m(TaskRunner pid=14058)[0m                                  'ppo_epochs': 1,
[36m(TaskRunner pid=14058)[0m                                  'ppo_max_token_len_per_gpu': 16384,
[36m(TaskRunner pid=14058)[0m                                  'ppo_micro_batch_size': None,
[36m(TaskRunner pid=14058)[0m                                  'ppo_micro_batch_size_per_gpu': 2,
[36m(TaskRunner pid=14058)[0m                                  'ppo_mini_batch_size': 8,
[36m(TaskRunner pid=14058)[0m                                  'profiler': {'_target_': 'verl.utils.profiler.ProfilerConfig',
[36m(TaskRunner pid=14058)[0m                                               'all_ranks': False,
[36m(TaskRunner pid=14058)[0m                                               'enable': False,
[36m(TaskRunner pid=14058)[0m                                               'ranks': [],
[36m(TaskRunner pid=14058)[0m                                               'save_path': 'outputs/profile',
[36m(TaskRunner pid=14058)[0m                                               'tool': None,
[36m(TaskRunner pid=14058)[0m                                               'tool_config': {'npu': {'_target_': 'verl.utils.profiler.config.NPUToolConfig',
[36m(TaskRunner pid=14058)[0m                                                                       'analysis': True,
[36m(TaskRunner pid=14058)[0m                                                                       'contents': [],
[36m(TaskRunner pid=14058)[0m                                                                       'discrete': False,
[36m(TaskRunner pid=14058)[0m                                                                       'level': 'level1'},
[36m(TaskRunner pid=14058)[0m                                                               'nsys': {'_target_': 'verl.utils.profiler.config.NsightToolConfig',
[36m(TaskRunner pid=14058)[0m                                                                        'discrete': False},
[36m(TaskRunner pid=14058)[0m                                                               'torch': {'_target_': 'verl.utils.profiler.config.TorchProfilerToolConfig',
[36m(TaskRunner pid=14058)[0m                                                                         'step_end': None,
[36m(TaskRunner pid=14058)[0m                                                                         'step_start': 0},
[36m(TaskRunner pid=14058)[0m                                                               'torch_memory': {'_target_': 'verl.utils.profiler.config.TorchMemoryToolConfig',
[36m(TaskRunner pid=14058)[0m                                                                                'stack_depth': 32,
[36m(TaskRunner pid=14058)[0m                                                                                'trace_alloc_max_entries': 100000}}},
[36m(TaskRunner pid=14058)[0m                                  'shuffle': False,
[36m(TaskRunner pid=14058)[0m                                  'strategy': 'fsdp',
[36m(TaskRunner pid=14058)[0m                                  'tis_imp_ratio_cap': -1,
[36m(TaskRunner pid=14058)[0m                                  'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=14058)[0m                                  'use_dynamic_bsz': False,
[36m(TaskRunner pid=14058)[0m                                  'use_fused_kernels': False,
[36m(TaskRunner pid=14058)[0m                                  'use_kl_loss': True,
[36m(TaskRunner pid=14058)[0m                                  'use_remove_padding': True,
[36m(TaskRunner pid=14058)[0m                                  'use_torch_compile': True},
[36m(TaskRunner pid=14058)[0m                        'hybrid_engine': True,
[36m(TaskRunner pid=14058)[0m                        'model': {'_target_': 'verl.workers.config.HFModelConfig',
[36m(TaskRunner pid=14058)[0m                                  'custom_chat_template': None,
[36m(TaskRunner pid=14058)[0m                                  'enable_activation_offload': False,
[36m(TaskRunner pid=14058)[0m                                  'enable_gradient_checkpointing': True,
[36m(TaskRunner pid=14058)[0m                                  'exclude_modules': None,
[36m(TaskRunner pid=14058)[0m                                  'external_lib': None,
[36m(TaskRunner pid=14058)[0m                                  'fused_kernel_options': {'impl_backend': 'torch'},
[36m(TaskRunner pid=14058)[0m                                  'hf_config_path': None,
[36m(TaskRunner pid=14058)[0m                                  'lora_alpha': 16,
[36m(TaskRunner pid=14058)[0m                                  'lora_rank': 0,
[36m(TaskRunner pid=14058)[0m                                  'override_config': {},
[36m(TaskRunner pid=14058)[0m                                  'path': '/opt/nas/s/enyu/verl_askActively/verl/models/Qwen3-4B-Instruct-2507-sft',
[36m(TaskRunner pid=14058)[0m                                  'target_modules': 'all-linear',
[36m(TaskRunner pid=14058)[0m                                  'tokenizer_path': None,
[36m(TaskRunner pid=14058)[0m                                  'trust_remote_code': False,
[36m(TaskRunner pid=14058)[0m                                  'use_fused_kernels': False,
[36m(TaskRunner pid=14058)[0m                                  'use_liger': False,
[36m(TaskRunner pid=14058)[0m                                  'use_remove_padding': True,
[36m(TaskRunner pid=14058)[0m                                  'use_shm': False},
[36m(TaskRunner pid=14058)[0m                        'nccl_timeout': 600,
[36m(TaskRunner pid=14058)[0m                        'ref': {'entropy_checkpointing': False,
[36m(TaskRunner pid=14058)[0m                                'entropy_from_logits_with_chunking': False,
[36m(TaskRunner pid=14058)[0m                                'fsdp_config': {'_target_': 'verl.workers.config.FSDPEngineConfig',
[36m(TaskRunner pid=14058)[0m                                                'entropy_checkpointing': False,
[36m(TaskRunner pid=14058)[0m                                                'entropy_from_logits_with_chunking': False,
[36m(TaskRunner pid=14058)[0m                                                'forward_only': False,
[36m(TaskRunner pid=14058)[0m                                                'forward_prefetch': False,
[36m(TaskRunner pid=14058)[0m                                                'fsdp_size': -1,
[36m(TaskRunner pid=14058)[0m                                                'model_dtype': 'fp32',
[36m(TaskRunner pid=14058)[0m                                                'offload_policy': False,
[36m(TaskRunner pid=14058)[0m                                                'optimizer_offload': False,
[36m(TaskRunner pid=14058)[0m                                                'param_offload': False,
[36m(TaskRunner pid=14058)[0m                                                'reshard_after_forward': True,
[36m(TaskRunner pid=14058)[0m                                                'strategy': 'fsdp',
[36m(TaskRunner pid=14058)[0m                                                'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=14058)[0m                                                'use_orig_params': False,
[36m(TaskRunner pid=14058)[0m                                                'use_torch_compile': True,
[36m(TaskRunner pid=14058)[0m                                                'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=14058)[0m                                'log_prob_max_token_len_per_gpu': 16384,
[36m(TaskRunner pid=14058)[0m                                'log_prob_micro_batch_size': None,
[36m(TaskRunner pid=14058)[0m                                'log_prob_micro_batch_size_per_gpu': 2,
[36m(TaskRunner pid=14058)[0m                                'log_prob_use_dynamic_bsz': False,
[36m(TaskRunner pid=14058)[0m                                'model': None,
[36m(TaskRunner pid=14058)[0m                                'profiler': {'_target_': 'verl.utils.profiler.ProfilerConfig',
[36m(TaskRunner pid=14058)[0m                                             'all_ranks': False,
[36m(TaskRunner pid=14058)[0m                                             'enable': False,
[36m(TaskRunner pid=14058)[0m                                             'ranks': [],
[36m(TaskRunner pid=14058)[0m                                             'save_path': 'outputs/profile',
[36m(TaskRunner pid=14058)[0m                                             'tool': None,
[36m(TaskRunner pid=14058)[0m                                             'tool_config': {'npu': {'_target_': 'verl.utils.profiler.config.NPUToolConfig',
[36m(TaskRunner pid=14058)[0m                                                                     'analysis': True,
[36m(TaskRunner pid=14058)[0m                                                                     'contents': [],
[36m(TaskRunner pid=14058)[0m                                                                     'discrete': False,
[36m(TaskRunner pid=14058)[0m                                                                     'level': 'level1'},
[36m(TaskRunner pid=14058)[0m                                                             'nsys': {'_target_': 'verl.utils.profiler.config.NsightToolConfig',
[36m(TaskRunner pid=14058)[0m                                                                      'discrete': False},
[36m(TaskRunner pid=14058)[0m                                                             'torch': {'_target_': 'verl.utils.profiler.config.TorchProfilerToolConfig',
[36m(TaskRunner pid=14058)[0m                                                                       'step_end': None,
[36m(TaskRunner pid=14058)[0m                                                                       'step_start': 0},
[36m(TaskRunner pid=14058)[0m                                                             'torch_memory': {'_target_': 'verl.utils.profiler.config.TorchMemoryToolConfig',
[36m(TaskRunner pid=14058)[0m                                                                              'stack_depth': 32,
[36m(TaskRunner pid=14058)[0m                                                                              'trace_alloc_max_entries': 100000}}},
[36m(TaskRunner pid=14058)[0m                                'strategy': 'fsdp',
[36m(TaskRunner pid=14058)[0m                                'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=14058)[0m                                'use_torch_compile': True},
[36m(TaskRunner pid=14058)[0m                        'rollout': {'_target_': 'verl.workers.config.RolloutConfig',
[36m(TaskRunner pid=14058)[0m                                    'agent': {'_target_': 'verl.workers.config.AgentLoopConfig',
[36m(TaskRunner pid=14058)[0m                                              'agent_loop_config_path': None,
[36m(TaskRunner pid=14058)[0m                                              'custom_async_server': {'_target_': 'verl.workers.config.CustomAsyncServerConfig',
[36m(TaskRunner pid=14058)[0m                                                                      'name': None,
[36m(TaskRunner pid=14058)[0m                                                                      'path': None},
[36m(TaskRunner pid=14058)[0m                                              'num_workers': 8},
[36m(TaskRunner pid=14058)[0m                                    'calculate_log_probs': False,
[36m(TaskRunner pid=14058)[0m                                    'cudagraph_capture_sizes': None,
[36m(TaskRunner pid=14058)[0m                                    'disable_log_stats': True,
[36m(TaskRunner pid=14058)[0m                                    'do_sample': True,
[36m(TaskRunner pid=14058)[0m                                    'dtype': 'bfloat16',
[36m(TaskRunner pid=14058)[0m                                    'enable_chunked_prefill': True,
[36m(TaskRunner pid=14058)[0m                                    'enforce_eager': False,
[36m(TaskRunner pid=14058)[0m                                    'engine_kwargs': {'sglang': {}, 'vllm': {}},
[36m(TaskRunner pid=14058)[0m                                    'free_cache_engine': True,
[36m(TaskRunner pid=14058)[0m                                    'gpu_memory_utilization': 0.7,
[36m(TaskRunner pid=14058)[0m                                    'ignore_eos': False,
[36m(TaskRunner pid=14058)[0m                                    'layered_summon': False,
[36m(TaskRunner pid=14058)[0m                                    'load_format': 'dummy_dtensor',
[36m(TaskRunner pid=14058)[0m                                    'log_prob_max_token_len_per_gpu': 16384,
[36m(TaskRunner pid=14058)[0m                                    'log_prob_micro_batch_size': None,
[36m(TaskRunner pid=14058)[0m                                    'log_prob_micro_batch_size_per_gpu': 2,
[36m(TaskRunner pid=14058)[0m                                    'log_prob_use_dynamic_bsz': False,
[36m(TaskRunner pid=14058)[0m                                    'max_model_len': None,
[36m(TaskRunner pid=14058)[0m                                    'max_num_batched_tokens': 8192,
[36m(TaskRunner pid=14058)[0m                                    'max_num_seqs': 1024,
[36m(TaskRunner pid=14058)[0m                                    'mode': 'sync',
[36m(TaskRunner pid=14058)[0m                                    'multi_stage_wake_up': False,
[36m(TaskRunner pid=14058)[0m                                    'multi_turn': {'_target_': 'verl.workers.config.MultiTurnConfig',
[36m(TaskRunner pid=14058)[0m                                                   'enable': True,
[36m(TaskRunner pid=14058)[0m                                                   'format': 'hermes',
[36m(TaskRunner pid=14058)[0m                                                   'interaction_config_path': '/opt/nas/s/enyu/verl_askActively/verl/trainer/config/interaction/askActively_interaction.yaml',
[36m(TaskRunner pid=14058)[0m                                                   'max_assistant_turns': None,
[36m(TaskRunner pid=14058)[0m                                                   'max_parallel_calls': 1,
[36m(TaskRunner pid=14058)[0m                                                   'max_tool_response_length': 256,
[36m(TaskRunner pid=14058)[0m                                                   'max_user_turns': 5,
[36m(TaskRunner pid=14058)[0m                                                   'tokenization_sanity_check_mode': 'strict',
[36m(TaskRunner pid=14058)[0m                                                   'tool_config_path': None,
[36m(TaskRunner pid=14058)[0m                                                   'tool_response_truncate_side': 'middle',
[36m(TaskRunner pid=14058)[0m                                                   'use_inference_chat_template': False},
[36m(TaskRunner pid=14058)[0m                                    'n': 8,
[36m(TaskRunner pid=14058)[0m                                    'name': 'sglang',
[36m(TaskRunner pid=14058)[0m                                    'over_sample_rate': 0,
[36m(TaskRunner pid=14058)[0m                                    'profiler': {'_target_': 'verl.utils.profiler.ProfilerConfig',
[36m(TaskRunner pid=14058)[0m                                                 'all_ranks': False,
[36m(TaskRunner pid=14058)[0m                                                 'enable': False,
[36m(TaskRunner pid=14058)[0m                                                 'ranks': [],
[36m(TaskRunner pid=14058)[0m                                                 'save_path': 'outputs/profile',
[36m(TaskRunner pid=14058)[0m                                                 'tool': None,
[36m(TaskRunner pid=14058)[0m                                                 'tool_config': {'npu': {'_target_': 'verl.utils.profiler.config.NPUToolConfig',
[36m(TaskRunner pid=14058)[0m                                                                         'analysis': True,
[36m(TaskRunner pid=14058)[0m                                                                         'contents': [],
[36m(TaskRunner pid=14058)[0m                                                                         'discrete': False,
[36m(TaskRunner pid=14058)[0m                                                                         'level': 'level1'},
[36m(TaskRunner pid=14058)[0m                                                                 'nsys': {'_target_': 'verl.utils.profiler.config.NsightToolConfig',
[36m(TaskRunner pid=14058)[0m                                                                          'discrete': False},
[36m(TaskRunner pid=14058)[0m                                                                 'torch': {'_target_': 'verl.utils.profiler.config.TorchProfilerToolConfig',
[36m(TaskRunner pid=14058)[0m                                                                           'step_end': None,
[36m(TaskRunner pid=14058)[0m                                                                           'step_start': 0},
[36m(TaskRunner pid=14058)[0m                                                                 'torch_memory': {'_target_': 'verl.utils.profiler.config.TorchMemoryToolConfig',
[36m(TaskRunner pid=14058)[0m                                                                                  'stack_depth': 32,
[36m(TaskRunner pid=14058)[0m                                                                                  'trace_alloc_max_entries': 100000}}},
[36m(TaskRunner pid=14058)[0m                                    'prompt_length': 1024,
[36m(TaskRunner pid=14058)[0m                                    'response_length': 3072,
[36m(TaskRunner pid=14058)[0m                                    'skip_dump_dir': '/tmp/rollout_dump',
[36m(TaskRunner pid=14058)[0m                                    'skip_rollout': False,
[36m(TaskRunner pid=14058)[0m                                    'temperature': 1.0,
[36m(TaskRunner pid=14058)[0m                                    'tensor_model_parallel_size': 2,
[36m(TaskRunner pid=14058)[0m                                    'top_k': -1,
[36m(TaskRunner pid=14058)[0m                                    'top_p': 1,
[36m(TaskRunner pid=14058)[0m                                    'trace': {'_target_': 'verl.workers.config.TraceConfig',
[36m(TaskRunner pid=14058)[0m                                              'backend': None,
[36m(TaskRunner pid=14058)[0m                                              'token2text': False},
[36m(TaskRunner pid=14058)[0m                                    'update_weights_bucket_megabytes': 512,
[36m(TaskRunner pid=14058)[0m                                    'val_kwargs': {'_target_': 'verl.workers.config.SamplingConfig',
[36m(TaskRunner pid=14058)[0m                                                   'do_sample': False,
[36m(TaskRunner pid=14058)[0m                                                   'n': 1,
[36m(TaskRunner pid=14058)[0m                                                   'temperature': 0,
[36m(TaskRunner pid=14058)[0m                                                   'top_k': -1,
[36m(TaskRunner pid=14058)[0m                                                   'top_p': 1.0}}},
[36m(TaskRunner pid=14058)[0m  'algorithm': {'_target_': 'verl.trainer.config.AlgoConfig',
[36m(TaskRunner pid=14058)[0m                'adv_estimator': 'grpo',
[36m(TaskRunner pid=14058)[0m                'gamma': 1.0,
[36m(TaskRunner pid=14058)[0m                'kl_ctrl': {'_target_': 'verl.trainer.config.KLControlConfig',
[36m(TaskRunner pid=14058)[0m                            'horizon': 10000,
[36m(TaskRunner pid=14058)[0m                            'kl_coef': 0.001,
[36m(TaskRunner pid=14058)[0m                            'target_kl': 0.1,
[36m(TaskRunner pid=14058)[0m                            'type': 'fixed'},
[36m(TaskRunner pid=14058)[0m                'kl_penalty': 'kl',
[36m(TaskRunner pid=14058)[0m                'lam': 1.0,
[36m(TaskRunner pid=14058)[0m                'norm_adv_by_std_in_grpo': True,
[36m(TaskRunner pid=14058)[0m                'pf_ppo': {'reweight_method': 'pow', 'weight_pow': 2.0},
[36m(TaskRunner pid=14058)[0m                'use_kl_in_reward': False,
[36m(TaskRunner pid=14058)[0m                'use_pf_ppo': False},
[36m(TaskRunner pid=14058)[0m  'critic': {'_target_': 'verl.workers.config.FSDPCriticConfig',
[36m(TaskRunner pid=14058)[0m             'checkpoint': {'_target_': 'verl.trainer.config.CheckpointConfig',
[36m(TaskRunner pid=14058)[0m                            'async_save': False,
[36m(TaskRunner pid=14058)[0m                            'load_contents': ['model', 'optimizer', 'extra'],
[36m(TaskRunner pid=14058)[0m                            'save_contents': ['model', 'optimizer', 'extra']},
[36m(TaskRunner pid=14058)[0m             'cliprange_value': 0.5,
[36m(TaskRunner pid=14058)[0m             'enable': None,
[36m(TaskRunner pid=14058)[0m             'forward_max_token_len_per_gpu': 32768,
[36m(TaskRunner pid=14058)[0m             'forward_micro_batch_size': None,
[36m(TaskRunner pid=14058)[0m             'forward_micro_batch_size_per_gpu': None,
[36m(TaskRunner pid=14058)[0m             'grad_clip': 1.0,
[36m(TaskRunner pid=14058)[0m             'loss_agg_mode': 'token-mean',
[36m(TaskRunner pid=14058)[0m             'model': {'_target_': 'verl.workers.config.FSDPCriticModelCfg',
[36m(TaskRunner pid=14058)[0m                       'enable_activation_offload': False,
[36m(TaskRunner pid=14058)[0m                       'enable_gradient_checkpointing': True,
[36m(TaskRunner pid=14058)[0m                       'external_lib': None,
[36m(TaskRunner pid=14058)[0m                       'fsdp_config': {'_target_': 'verl.workers.config.FSDPEngineConfig',
[36m(TaskRunner pid=14058)[0m                                       'entropy_checkpointing': False,
[36m(TaskRunner pid=14058)[0m                                       'entropy_from_logits_with_chunking': False,
[36m(TaskRunner pid=14058)[0m                                       'forward_only': False,
[36m(TaskRunner pid=14058)[0m                                       'forward_prefetch': False,
[36m(TaskRunner pid=14058)[0m                                       'fsdp_size': -1,
[36m(TaskRunner pid=14058)[0m                                       'model_dtype': 'fp32',
[36m(TaskRunner pid=14058)[0m                                       'offload_policy': False,
[36m(TaskRunner pid=14058)[0m                                       'optimizer_offload': False,
[36m(TaskRunner pid=14058)[0m                                       'param_offload': False,
[36m(TaskRunner pid=14058)[0m                                       'reshard_after_forward': True,
[36m(TaskRunner pid=14058)[0m                                       'strategy': 'fsdp',
[36m(TaskRunner pid=14058)[0m                                       'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=14058)[0m                                       'use_orig_params': False,
[36m(TaskRunner pid=14058)[0m                                       'use_torch_compile': True,
[36m(TaskRunner pid=14058)[0m                                       'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=14058)[0m                       'lora_alpha': 16,
[36m(TaskRunner pid=14058)[0m                       'lora_rank': 0,
[36m(TaskRunner pid=14058)[0m                       'override_config': {},
[36m(TaskRunner pid=14058)[0m                       'path': '~/models/deepseek-llm-7b-chat',
[36m(TaskRunner pid=14058)[0m                       'target_modules': 'all-linear',
[36m(TaskRunner pid=14058)[0m                       'tokenizer_path': '/opt/nas/s/enyu/verl_askActively/verl/models/Qwen3-4B-Instruct-2507-sft',
[36m(TaskRunner pid=14058)[0m                       'trust_remote_code': False,
[36m(TaskRunner pid=14058)[0m                       'use_remove_padding': False,
[36m(TaskRunner pid=14058)[0m                       'use_shm': False},
[36m(TaskRunner pid=14058)[0m             'optim': {'_target_': 'verl.workers.config.FSDPOptimizerConfig',
[36m(TaskRunner pid=14058)[0m                       'betas': [0.9, 0.999],
[36m(TaskRunner pid=14058)[0m                       'clip_grad': 1.0,
[36m(TaskRunner pid=14058)[0m                       'lr': 1e-05,
[36m(TaskRunner pid=14058)[0m                       'lr_warmup_steps': -1,
[36m(TaskRunner pid=14058)[0m                       'lr_warmup_steps_ratio': 0.0,
[36m(TaskRunner pid=14058)[0m                       'min_lr_ratio': 0.0,
[36m(TaskRunner pid=14058)[0m                       'num_cycles': 0.5,
[36m(TaskRunner pid=14058)[0m                       'total_training_steps': -1,
[36m(TaskRunner pid=14058)[0m                       'warmup_style': 'constant',
[36m(TaskRunner pid=14058)[0m                       'weight_decay': 0.01},
[36m(TaskRunner pid=14058)[0m             'ppo_epochs': 1,
[36m(TaskRunner pid=14058)[0m             'ppo_max_token_len_per_gpu': 32768,
[36m(TaskRunner pid=14058)[0m             'ppo_micro_batch_size': None,
[36m(TaskRunner pid=14058)[0m             'ppo_micro_batch_size_per_gpu': None,
[36m(TaskRunner pid=14058)[0m             'ppo_mini_batch_size': 8,
[36m(TaskRunner pid=14058)[0m             'profiler': {'_target_': 'verl.utils.profiler.ProfilerConfig',
[36m(TaskRunner pid=14058)[0m                          'all_ranks': False,
[36m(TaskRunner pid=14058)[0m                          'enable': False,
[36m(TaskRunner pid=14058)[0m                          'ranks': [],
[36m(TaskRunner pid=14058)[0m                          'save_path': 'outputs/profile',
[36m(TaskRunner pid=14058)[0m                          'tool': None,
[36m(TaskRunner pid=14058)[0m                          'tool_config': {'npu': {'_target_': 'verl.utils.profiler.config.NPUToolConfig',
[36m(TaskRunner pid=14058)[0m                                                  'analysis': True,
[36m(TaskRunner pid=14058)[0m                                                  'contents': [],
[36m(TaskRunner pid=14058)[0m                                                  'discrete': False,
[36m(TaskRunner pid=14058)[0m                                                  'level': 'level1'},
[36m(TaskRunner pid=14058)[0m                                          'nsys': {'_target_': 'verl.utils.profiler.config.NsightToolConfig',
[36m(TaskRunner pid=14058)[0m                                                   'discrete': False},
[36m(TaskRunner pid=14058)[0m                                          'torch': {'_target_': 'verl.utils.profiler.config.TorchProfilerToolConfig',
[36m(TaskRunner pid=14058)[0m                                                    'step_end': None,
[36m(TaskRunner pid=14058)[0m                                                    'step_start': 0},
[36m(TaskRunner pid=14058)[0m                                          'torch_memory': {'_target_': 'verl.utils.profiler.config.TorchMemoryToolConfig',
[36m(TaskRunner pid=14058)[0m                                                           'stack_depth': 32,
[36m(TaskRunner pid=14058)[0m                                                           'trace_alloc_max_entries': 100000}}},
[36m(TaskRunner pid=14058)[0m             'rollout_n': 8,
[36m(TaskRunner pid=14058)[0m             'shuffle': False,
[36m(TaskRunner pid=14058)[0m             'strategy': 'fsdp',
[36m(TaskRunner pid=14058)[0m             'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=14058)[0m             'use_dynamic_bsz': False},
[36m(TaskRunner pid=14058)[0m  'custom_reward_function': {'name': 'compute_score', 'path': None},
[36m(TaskRunner pid=14058)[0m  'data': {'apply_chat_template_kwargs': {},
[36m(TaskRunner pid=14058)[0m           'custom_cls': {'name': None, 'path': None},
[36m(TaskRunner pid=14058)[0m           'datagen': {'name': None, 'path': None},
[36m(TaskRunner pid=14058)[0m           'dataloader_num_workers': 8,
[36m(TaskRunner pid=14058)[0m           'filter_overlong_prompts': True,
[36m(TaskRunner pid=14058)[0m           'filter_overlong_prompts_workers': 1,
[36m(TaskRunner pid=14058)[0m           'image_key': 'images',
[36m(TaskRunner pid=14058)[0m           'max_prompt_length': 1024,
[36m(TaskRunner pid=14058)[0m           'max_response_length': 3072,
[36m(TaskRunner pid=14058)[0m           'prompt_key': 'prompt',
[36m(TaskRunner pid=14058)[0m           'return_full_prompt': False,
[36m(TaskRunner pid=14058)[0m           'return_multi_modal_inputs': True,
[36m(TaskRunner pid=14058)[0m           'return_raw_chat': True,
[36m(TaskRunner pid=14058)[0m           'return_raw_input_ids': False,
[36m(TaskRunner pid=14058)[0m           'reward_fn_key': 'data_source',
[36m(TaskRunner pid=14058)[0m           'sampler': {'class_name': None, 'class_path': None},
[36m(TaskRunner pid=14058)[0m           'shuffle': True,
[36m(TaskRunner pid=14058)[0m           'tokenizer': None,
[36m(TaskRunner pid=14058)[0m           'train_batch_size': 8,
[36m(TaskRunner pid=14058)[0m           'train_files': '/opt/nas/s/enyu/verl_askActively/verl/data/rl/test_rl.parquet',
[36m(TaskRunner pid=14058)[0m           'truncation': 'error',
[36m(TaskRunner pid=14058)[0m           'trust_remote_code': False,
[36m(TaskRunner pid=14058)[0m           'use_shm': False,
[36m(TaskRunner pid=14058)[0m           'val_batch_size': None,
[36m(TaskRunner pid=14058)[0m           'val_files': '/opt/nas/s/enyu/verl_askActively/verl/data/rl/test_rl.parquet',
[36m(TaskRunner pid=14058)[0m           'validation_shuffle': False,
[36m(TaskRunner pid=14058)[0m           'video_key': 'videos'},
[36m(TaskRunner pid=14058)[0m  'global_profiler': {'_target_': 'verl.utils.profiler.ProfilerConfig',
[36m(TaskRunner pid=14058)[0m                      'global_tool_config': {'nsys': {'_target_': 'verl.utils.profiler.config.NsightToolConfig',
[36m(TaskRunner pid=14058)[0m                                                      'controller_nsight_options': {'cuda-graph-trace': 'graph',
[36m(TaskRunner pid=14058)[0m                                                                                    'cuda-memory-usage': 'true',
[36m(TaskRunner pid=14058)[0m                                                                                    'trace': 'cuda,nvtx,cublas,ucx'},
[36m(TaskRunner pid=14058)[0m                                                      'discrete': False,
[36m(TaskRunner pid=14058)[0m                                                      'worker_nsight_options': {'capture-range': 'cudaProfilerApi',
[36m(TaskRunner pid=14058)[0m                                                                                'capture-range-end': None,
[36m(TaskRunner pid=14058)[0m                                                                                'cuda-graph-trace': 'graph',
[36m(TaskRunner pid=14058)[0m                                                                                'cuda-memory-usage': 'true',
[36m(TaskRunner pid=14058)[0m                                                                                'kill': 'none',
[36m(TaskRunner pid=14058)[0m                                                                                'trace': 'cuda,nvtx,cublas,ucx'}},
[36m(TaskRunner pid=14058)[0m                                             'torch_memory': {'context': 'all',
[36m(TaskRunner pid=14058)[0m                                                              'kw_args': {},
[36m(TaskRunner pid=14058)[0m                                                              'stack_depth': 32,
[36m(TaskRunner pid=14058)[0m                                                              'stacks': 'all',
[36m(TaskRunner pid=14058)[0m                                                              'trace_alloc_max_entries': 100000}},
[36m(TaskRunner pid=14058)[0m                      'profile_continuous_steps': False,
[36m(TaskRunner pid=14058)[0m                      'save_path': 'outputs/profile',
[36m(TaskRunner pid=14058)[0m                      'steps': None,
[36m(TaskRunner pid=14058)[0m                      'tool': None},
[36m(TaskRunner pid=14058)[0m  'ray_kwargs': {'ray_init': {'num_cpus': None,
[36m(TaskRunner pid=14058)[0m                              'runtime_env': {'env_vars': {'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:False'}}},
[36m(TaskRunner pid=14058)[0m                 'timeline_json_file': None},
[36m(TaskRunner pid=14058)[0m  'reward_model': {'enable': False,
[36m(TaskRunner pid=14058)[0m                   'enable_resource_pool': False,
[36m(TaskRunner pid=14058)[0m                   'forward_max_token_len_per_gpu': 32768,
[36m(TaskRunner pid=14058)[0m                   'launch_reward_fn_async': False,
[36m(TaskRunner pid=14058)[0m                   'max_length': None,
[36m(TaskRunner pid=14058)[0m                   'micro_batch_size': None,
[36m(TaskRunner pid=14058)[0m                   'micro_batch_size_per_gpu': None,
[36m(TaskRunner pid=14058)[0m                   'model': {'external_lib': None,
[36m(TaskRunner pid=14058)[0m                             'fsdp_config': {'_target_': 'verl.workers.config.FSDPEngineConfig',
[36m(TaskRunner pid=14058)[0m                                             'forward_prefetch': False,
[36m(TaskRunner pid=14058)[0m                                             'fsdp_size': -1,
[36m(TaskRunner pid=14058)[0m                                             'param_offload': False,
[36m(TaskRunner pid=14058)[0m                                             'reshard_after_forward': True,
[36m(TaskRunner pid=14058)[0m                                             'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=14058)[0m                             'input_tokenizer': '/opt/nas/s/enyu/verl_askActively/verl/models/Qwen3-4B-Instruct-2507-sft',
[36m(TaskRunner pid=14058)[0m                             'path': '~/models/FsfairX-LLaMA3-RM-v0.1',
[36m(TaskRunner pid=14058)[0m                             'trust_remote_code': False,
[36m(TaskRunner pid=14058)[0m                             'use_fused_kernels': False,
[36m(TaskRunner pid=14058)[0m                             'use_remove_padding': False,
[36m(TaskRunner pid=14058)[0m                             'use_shm': False},
[36m(TaskRunner pid=14058)[0m                   'n_gpus_per_node': 0,
[36m(TaskRunner pid=14058)[0m                   'nnodes': 0,
[36m(TaskRunner pid=14058)[0m                   'profiler': {'_target_': 'verl.utils.profiler.ProfilerConfig',
[36m(TaskRunner pid=14058)[0m                                'all_ranks': False,
[36m(TaskRunner pid=14058)[0m                                'enable': False,
[36m(TaskRunner pid=14058)[0m                                'ranks': [],
[36m(TaskRunner pid=14058)[0m                                'save_path': 'outputs/profile',
[36m(TaskRunner pid=14058)[0m                                'tool': None,
[36m(TaskRunner pid=14058)[0m                                'tool_config': {'npu': {'_target_': 'verl.utils.profiler.config.NPUToolConfig',
[36m(TaskRunner pid=14058)[0m                                                        'analysis': True,
[36m(TaskRunner pid=14058)[0m                                                        'contents': [],
[36m(TaskRunner pid=14058)[0m                                                        'discrete': False,
[36m(TaskRunner pid=14058)[0m                                                        'level': 'level1'},
[36m(TaskRunner pid=14058)[0m                                                'nsys': {'_target_': 'verl.utils.profiler.config.NsightToolConfig',
[36m(TaskRunner pid=14058)[0m                                                         'discrete': False},
[36m(TaskRunner pid=14058)[0m                                                'torch': {'_target_': 'verl.utils.profiler.config.TorchProfilerToolConfig',
[36m(TaskRunner pid=14058)[0m                                                          'step_end': None,
[36m(TaskRunner pid=14058)[0m                                                          'step_start': 0},
[36m(TaskRunner pid=14058)[0m                                                'torch_memory': {'_target_': 'verl.utils.profiler.config.TorchMemoryToolConfig',
[36m(TaskRunner pid=14058)[0m                                                                 'stack_depth': 32,
[36m(TaskRunner pid=14058)[0m                                                                 'trace_alloc_max_entries': 100000}}},
[36m(TaskRunner pid=14058)[0m                   'reward_manager': 'naive',
[36m(TaskRunner pid=14058)[0m                   'sandbox_fusion': {'max_concurrent': 64,
[36m(TaskRunner pid=14058)[0m                                      'memory_limit_mb': 1024,
[36m(TaskRunner pid=14058)[0m                                      'url': None},
[36m(TaskRunner pid=14058)[0m                   'strategy': 'fsdp',
[36m(TaskRunner pid=14058)[0m                   'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=14058)[0m                   'use_dynamic_bsz': False},
[36m(TaskRunner pid=14058)[0m  'trainer': {'balance_batch': True,
[36m(TaskRunner pid=14058)[0m              'critic_warmup': 0,
[36m(TaskRunner pid=14058)[0m              'default_hdfs_dir': None,
[36m(TaskRunner pid=14058)[0m              'default_local_dir': 'checkpoints/askActively/askActively-RL',
[36m(TaskRunner pid=14058)[0m              'del_local_ckpt_after_load': False,
[36m(TaskRunner pid=14058)[0m              'device': 'cuda',
[36m(TaskRunner pid=14058)[0m              'esi_redundant_time': 0,
[36m(TaskRunner pid=14058)[0m              'experiment_name': 'askActively-RL',
[36m(TaskRunner pid=14058)[0m              'log_val_generations': 0,
[36m(TaskRunner pid=14058)[0m              'logger': ['console'],
[36m(TaskRunner pid=14058)[0m              'max_actor_ckpt_to_keep': None,
[36m(TaskRunner pid=14058)[0m              'max_critic_ckpt_to_keep': None,
[36m(TaskRunner pid=14058)[0m              'n_gpus_per_node': 2,
[36m(TaskRunner pid=14058)[0m              'nnodes': 1,
[36m(TaskRunner pid=14058)[0m              'project_name': 'askActively',
[36m(TaskRunner pid=14058)[0m              'ray_wait_register_center_timeout': 300,
[36m(TaskRunner pid=14058)[0m              'resume_from_path': None,
[36m(TaskRunner pid=14058)[0m              'resume_mode': 'auto',
[36m(TaskRunner pid=14058)[0m              'rollout_data_dir': '/opt/nas/s/enyu/verl_askActively/verl/rollout_data',
[36m(TaskRunner pid=14058)[0m              'save_freq': 50,
[36m(TaskRunner pid=14058)[0m              'test_freq': 20,
[36m(TaskRunner pid=14058)[0m              'total_epochs': 15,
[36m(TaskRunner pid=14058)[0m              'total_training_steps': None,
[36m(TaskRunner pid=14058)[0m              'use_legacy_worker_impl': 'auto',
[36m(TaskRunner pid=14058)[0m              'val_before_train': True,
[36m(TaskRunner pid=14058)[0m              'val_only': False,
[36m(TaskRunner pid=14058)[0m              'validation_data_dir': None}}
[36m(TaskRunner pid=14058)[0m [validate_config] All configuration checks passed successfully!
[36m(TaskRunner pid=14058)[0m Using dataset class: RLHFDataset
[36m(TaskRunner pid=14058)[0m dataset len: 2329
[36m(TaskRunner pid=14058)[0m filter dataset len: 2329
[36m(TaskRunner pid=14058)[0m Using dataset class: RLHFDataset
[36m(TaskRunner pid=14058)[0m dataset len: 2329
[36m(TaskRunner pid=14058)[0m filter dataset len: 2329
[36m(TaskRunner pid=14058)[0m Size of train dataloader: 291, Size of val dataloader: 1
[36m(TaskRunner pid=14058)[0m Total training steps: 4365
[36m(TaskRunner pid=14058)[0m colocated worker base class <class 'verl.single_controller.base.worker.Worker'>
[36m(WorkerDict pid=26070)[0m reference model: /opt/nas/s/enyu/verl_askActively/verl/models/Qwen3-4B-Instruct-2507-sft
[36m(WorkerDict pid=26070)[0m Model config after override: Qwen3Config {
[36m(WorkerDict pid=26070)[0m   "architectures": [
[36m(WorkerDict pid=26070)[0m     "Qwen3ForCausalLM"
[36m(WorkerDict pid=26070)[0m   ],
[36m(WorkerDict pid=26070)[0m   "attention_bias": false,
[36m(WorkerDict pid=26070)[0m   "attention_dropout": 0.0,
[36m(WorkerDict pid=26070)[0m   "dtype": "bfloat16",
[36m(WorkerDict pid=26070)[0m   "eos_token_id": 151645,
[36m(WorkerDict pid=26070)[0m   "head_dim": 128,
[36m(WorkerDict pid=26070)[0m   "hidden_act": "silu",
[36m(WorkerDict pid=26070)[0m   "hidden_size": 2560,
[36m(WorkerDict pid=26070)[0m   "initializer_range": 0.02,
[36m(WorkerDict pid=26070)[0m   "intermediate_size": 9728,
[36m(WorkerDict pid=26070)[0m   "layer_types": [
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention"
[36m(WorkerDict pid=26070)[0m   ],
[36m(WorkerDict pid=26070)[0m   "max_position_embeddings": 262144,
[36m(WorkerDict pid=26070)[0m   "max_window_layers": 36,
[36m(WorkerDict pid=26070)[0m   "model_type": "qwen3",
[36m(WorkerDict pid=26070)[0m   "num_attention_heads": 32,
[36m(WorkerDict pid=26070)[0m   "num_hidden_layers": 36,
[36m(WorkerDict pid=26070)[0m   "num_key_value_heads": 8,
[36m(WorkerDict pid=26070)[0m   "pad_token_id": 151643,
[36m(WorkerDict pid=26070)[0m   "rms_norm_eps": 1e-06,
[36m(WorkerDict pid=26070)[0m   "rope_scaling": null,
[36m(WorkerDict pid=26070)[0m   "rope_theta": 5000000,
[36m(WorkerDict pid=26070)[0m   "sliding_window": null,
[36m(WorkerDict pid=26070)[0m   "tie_word_embeddings": true,
[36m(WorkerDict pid=26070)[0m   "transformers_version": "4.54.1",
[36m(WorkerDict pid=26070)[0m   "use_cache": true,
[36m(WorkerDict pid=26070)[0m   "use_sliding_window": false,
[36m(WorkerDict pid=26070)[0m   "vocab_size": 151936
[36m(WorkerDict pid=26070)[0m }
[36m(WorkerDict pid=26070)[0m 
[36m(WorkerDict pid=26071)[0m Monkey patch _flash_attention_forward in transformers.integrations.flash_attention
[36m(WorkerDict pid=26071)[0m Skipping monkey patch for Qwen3ForCausalLM as use_fused_kernels is False or fused_kernels_backend is torch
[36m(WorkerDict pid=26070)[0m Qwen3ForCausalLM contains 4.02B parameters
[36m(WorkerDict pid=26070)[0m wrap_policy: functools.partial(<function _or_policy at 0x7ee40c304b80>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x7ee40c304a40>, transformer_layer_cls={<class 'transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer'>})])
[36m(WorkerDict pid=26070)[0m Ref use_remove_padding=True
[36m(WorkerDict pid=26070)[0m Ref use_fused_kernels=False
[36m(WorkerDict pid=26070)[0m Monkey patch _flash_attention_forward in transformers.integrations.flash_attention
[36m(WorkerDict pid=26070)[0m Skipping monkey patch for Qwen3ForCausalLM as use_fused_kernels is False or fused_kernels_backend is torch
[36m(WorkerDict pid=26070)[0m Model config after override: Qwen3Config {
[36m(WorkerDict pid=26070)[0m   "architectures": [
[36m(WorkerDict pid=26070)[0m     "Qwen3ForCausalLM"
[36m(WorkerDict pid=26070)[0m   ],
[36m(WorkerDict pid=26070)[0m   "attention_bias": false,
[36m(WorkerDict pid=26070)[0m   "attention_dropout": 0.0,
[36m(WorkerDict pid=26070)[0m   "dtype": "bfloat16",
[36m(WorkerDict pid=26070)[0m   "eos_token_id": 151645,
[36m(WorkerDict pid=26070)[0m   "head_dim": 128,
[36m(WorkerDict pid=26070)[0m   "hidden_act": "silu",
[36m(WorkerDict pid=26070)[0m   "hidden_size": 2560,
[36m(WorkerDict pid=26070)[0m   "initializer_range": 0.02,
[36m(WorkerDict pid=26070)[0m   "intermediate_size": 9728,
[36m(WorkerDict pid=26070)[0m   "layer_types": [
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention",
[36m(WorkerDict pid=26070)[0m     "full_attention"
[36m(WorkerDict pid=26070)[0m   ],
[36m(WorkerDict pid=26070)[0m   "max_position_embeddings": 262144,
[36m(WorkerDict pid=26070)[0m   "max_window_layers": 36,
[36m(WorkerDict pid=26070)[0m   "model_type": "qwen3",
[36m(WorkerDict pid=26070)[0m   "num_attention_heads": 32,
[36m(WorkerDict pid=26070)[0m   "num_hidden_layers": 36,
[36m(WorkerDict pid=26070)[0m   "num_key_value_heads": 8,
[36m(WorkerDict pid=26070)[0m   "pad_token_id": 151643,
[36m(WorkerDict pid=26070)[0m   "rms_norm_eps": 1e-06,
[36m(WorkerDict pid=26070)[0m   "rope_scaling": null,
[36m(WorkerDict pid=26070)[0m   "rope_theta": 5000000,
[36m(WorkerDict pid=26070)[0m   "sliding_window": null,
[36m(WorkerDict pid=26070)[0m   "tie_word_embeddings": true,
[36m(WorkerDict pid=26070)[0m   "transformers_version": "4.54.1",
[36m(WorkerDict pid=26070)[0m   "use_cache": true,
[36m(WorkerDict pid=26070)[0m   "use_sliding_window": false,
[36m(WorkerDict pid=26070)[0m   "vocab_size": 151936
[36m(WorkerDict pid=26070)[0m }
[36m(WorkerDict pid=26070)[0m 
[36m(WorkerDict pid=26071)[0m Monkey patch _flash_attention_forward in transformers.integrations.flash_attention
[36m(WorkerDict pid=26071)[0m Skipping monkey patch for Qwen3ForCausalLM as use_fused_kernels is False or fused_kernels_backend is torch
[36m(WorkerDict pid=26070)[0m Qwen3ForCausalLM contains 4.02B parameters
[36m(WorkerDict pid=26070)[0m wrap_policy: functools.partial(<function _or_policy at 0x7ee40c304b80>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x7ee40c304a40>, transformer_layer_cls={<class 'transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer'>})])
[36m(WorkerDict pid=26070)[0m Total steps: 4365, num_warmup_steps: 0
[36m(WorkerDict pid=26070)[0m Actor use_remove_padding=True
[36m(WorkerDict pid=26070)[0m Actor use_fused_kernels=False
[33m(raylet)[0m A worker died or was killed while executing a task by an unexpected system error. To troubleshoot the problem, check the logs for the dead worker. Lease ID: 01000000879eaa39ac4b2f7f1fd42d7892cb201d7254e7697d4ba3e24061785d Worker ID: 639444b093314f7bf5018fac858fd410b3f6c6ad2d8a65fb2f25737f Node ID: d065ec5b9695e8a7bfcc9235e3164e4856b66cd74837f19a701ddb67 Worker IP address: 10.80.1.37 Worker port: 33173 Worker PID: 26070 Worker exit type: SYSTEM_ERROR Worker exit detail: Worker unexpectedly exits with a connection error code 2. End of file. There are some potential root causes. (1) The process is killed by SIGKILL by OOM killer due to high memory usage. (2) ray stop --force is called. (3) The worker is crashed unexpectedly due to SIGSEGV or other unexpected errors.
[36m(WorkerDict pid=26070)[0m Monkey patch _flash_attention_forward in transformers.integrations.flash_attention
[36m(WorkerDict pid=26070)[0m Skipping monkey patch for Qwen3ForCausalLM as use_fused_kernels is False or fused_kernels_backend is torch
Error executing job with overrides: ['algorithm.adv_estimator=grpo', 'data.train_batch_size=8', 'data.max_prompt_length=1024', 'data.max_response_length=3072', 'data.filter_overlong_prompts=True', 'data.truncation=error', 'data.return_raw_chat=True', 'actor_rollout_ref.model.path=/opt/nas/s/enyu/verl_askActively/verl/models/Qwen3-4B-Instruct-2507-sft', 'actor_rollout_ref.model.use_remove_padding=True', 'actor_rollout_ref.model.enable_gradient_checkpointing=True', 'actor_rollout_ref.actor.optim.lr=1e-6', 'actor_rollout_ref.actor.ppo_mini_batch_size=8', 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2', 'actor_rollout_ref.actor.use_kl_loss=True', 'actor_rollout_ref.actor.kl_loss_coef=0.001', 'actor_rollout_ref.actor.kl_loss_type=low_var_kl', 'actor_rollout_ref.actor.entropy_coeff=0', 'actor_rollout_ref.actor.fsdp_config.param_offload=False', 'actor_rollout_ref.actor.fsdp_config.optimizer_offload=False', 'actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16', 'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2', 'actor_rollout_ref.rollout.tensor_model_parallel_size=2', 'actor_rollout_ref.rollout.name=sglang', 'actor_rollout_ref.rollout.gpu_memory_utilization=0.7', 'actor_rollout_ref.rollout.n=8', 'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2', 'actor_rollout_ref.ref.fsdp_config.param_offload=False', 'algorithm.use_kl_in_reward=False', 'trainer.critic_warmup=0', 'trainer.logger=["console"]', 'trainer.project_name=askActively', 'trainer.experiment_name=askActively-RL', 'trainer.n_gpus_per_node=2', 'trainer.nnodes=1', 'trainer.save_freq=50', 'trainer.test_freq=20', 'trainer.rollout_data_dir=/opt/nas/s/enyu/verl_askActively/verl/rollout_data', 'data.train_files=/opt/nas/s/enyu/verl_askActively/verl/data/rl/test_rl.parquet', 'data.val_files=/opt/nas/s/enyu/verl_askActively/verl/data/rl/test_rl.parquet', 'actor_rollout_ref.rollout.multi_turn.interaction_config_path=/opt/nas/s/enyu/verl_askActively/verl/trainer/config/interaction/askActively_interaction.yaml', 'trainer.total_epochs=15']
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
[36m(WorkerDict pid=26071)[0m [rank1]:[W1120 22:03:52.871983384 TCPStore.cpp:125] [c10d] recvValue failed on SocketImpl(fd=81, addr=[mllm-ppo-qwen-0.mllm-ppo-qwen.mllm.svc.cluster.local]:50270, remote=[10-80-1-37.node-exporter.kube-system.svc.cluster.local]:52641): Connection reset by peer
[36m(WorkerDict pid=26071)[0m Exception raised from recvBytes at /pytorch/torch/csrc/distributed/c10d/Utils.hpp:675 (most recent call first):
[36m(WorkerDict pid=26071)[0m frame #4: c10d::TCPStore::check(std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > const&) + 0x2a9 (0x7ef9656ca2a9 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
[36m(WorkerDict pid=26071)[0m frame #5: c10d::ProcessGroupNCCL::heartbeatMonitor() + 0x379 (0x7ef926dc99f9 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
[36m(WorkerDict pid=26071)[0m 
[36m(WorkerDict pid=26071)[0m [rank1]:[W1120 22:03:52.878974796 ProcessGroupNCCL.cpp:1662] [PG ID 0 PG GUID 0(default_pg) Rank 1] Failed to check the "should dump" flag on TCPStore, (maybe TCPStore server has shut down too early), with error: Connection reset by peer
[36m(WorkerDict pid=26071)[0m frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x98 (0x7f16d83785e8 in /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/torch/lib/libc10.so)
[36m(WorkerDict pid=26071)[0m frame #8: <unknown function> + 0x126850 (0x7f172eaa3850 in /usr/lib/x86_64-linux-gnu/libc.so.6)[32m [repeated 6x across cluster][0m
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
ray.exceptions.RayTaskError(ActorDiedError): [36mray::TaskRunner.run()[39m (pid=14058, ip=10.80.1.37, actor_id=4855fd8c2c5b571e18efd95b01000000, repr=<main_ppo.TaskRunner object at 0x7fd43922f350>)
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
	actor_id: 7a837da243132d51939112db01000000
	pid: 26070
	name: Yr6ukSWorkerDict_0:0
	namespace: a175b978-4ace-41af-a34a-1ef1d9d5f2c0
	ip: 10.80.1.37
The actor is dead because its worker process has died. Worker exit type: SYSTEM_ERROR Worker exit detail: Worker unexpectedly exits with a connection error code 2. End of file. There are some potential root causes. (1) The process is killed by SIGKILL by OOM killer due to high memory usage. (2) ray stop --force is called. (3) The worker is crashed unexpectedly due to SIGSEGV or other unexpected errors.

