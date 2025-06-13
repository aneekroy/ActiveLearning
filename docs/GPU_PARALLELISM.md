# Preventing GPU 0 Out-of-Memory Errors

`run_experiment.py` uses PyTorch `DataParallel` when multiple GPUs are
available. The first device in `device_ids` keeps a full copy of the model and
also hosts the gather buffers. This often leads to out-of-memory (OOM)
exceptions that appear on GPU 0 even when other GPUs have free memory.

## Why it happens

`CUDA_VISIBLE_DEVICES` re‑indexes the visible GPUs starting from 0. When the
script is launched with `CUDA_VISIBLE_DEVICES=0,1,2,3`, PyTorch sees the devices
as `cuda:0`, `cuda:1`, `cuda:2`, `cuda:3`. `DataParallel` then
1. stores the original model on `cuda:0`,
2. replicates it on every other listed device, and
3. gathers outputs back to `cuda:0` by default.

The combination of two full model copies plus gathering buffers on the same GPU
causes the memory usage of GPU 0 to be roughly twice that of the others.
Large models such as a 1B parameter LLaMA can therefore exceed the available
memory on GPU 0 while the remaining GPUs stay mostly unused.

## Recommended solutions

1. **Limit the visible devices** so that the first device in the list is not the
   already over‑subscribed GPU:
   ```bash
   CUDA_VISIBLE_DEVICES=1,2,3 python -m src.run_experiment ...
   ```
2. **Change the gather device** if all GPUs must be used:
   ```python
   model = torch.nn.DataParallel(model, device_ids=[0,1,2,3], output_device=3)
   model = model.cuda(0)
   ```
   Here GPU 3 absorbs the gather overhead instead of GPU 0.
3. **Use `DistributedDataParallel`** for best scalability. Launch the experiment
   with `torchrun` so that each process owns exactly one GPU:
   ```bash
   torchrun --nproc_per_node=4 --master_port=29500 src/run_experiment_ddp.py \
       --task <task> --al_method <method> --model_name <path> --num_shots 8
   ```
   With DDP there is a single model copy per GPU and no central gather, which
   removes the imbalance entirely.

Reducing the batch size, shortening long prompts or enabling mixed precision
(`model.half()`) can further lower the memory footprint.
