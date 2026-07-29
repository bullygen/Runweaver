# PyTorch integration

The `pytorch` extra provides training/inference base classes, device resolution,
deterministic seed utilities, TorchMetrics wrapping and model references.
Checkpoint helpers capture model, optimizer, scheduler, AMP scaler,
Python/NumPy/CPU/CUDA RNG and a user cursor.

The user owns the model, DataLoader and training loop. Runweaver supplies
orchestration, reporting, checkpoint/artifact hooks and pruning integration. It
is intentionally not a universal Trainer. DDP/FSDP/distributed checkpointing
belong in application or future advanced adapters.

See Tutorial 4.
