import time
import os
from pathlib import Path
import json
from typing import Annotated

from typer import Typer, Option

from mini_trainer.async_structured_logger import AsyncStructuredLogger
import torch
import torch.distributed as dist

from mini_trainer.batch_metrics import BatchMetrics
from mini_trainer.sampler import get_data_loader
from mini_trainer.setup_model_for_training import setup_model, setup_training_components
from mini_trainer.utils import init_distributed_environment, log_rank_0, setup_logger
from mini_trainer.training_types import TrainingMode, LogLevelEnum
from mini_trainer.scheduler_utils import calculate_num_training_steps

app = Typer(
    pretty_exceptions_show_locals=False,  # Hide local variables in tracebacks
    pretty_exceptions_short=True   
)

def take_gradient_step(model, optimizer, lr_scheduler):
    """Scales gradients, applies clipping, and takes an optimization step."""
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    return grad_norm

def save_model(fsdp_model, samples_seen, output_dir, model_name_or_path):
    from huggingface_hub import split_torch_state_dict_into_shards
    from transformers import AutoTokenizer
    from safetensors.torch import save_file
    # Only on rank 0
    log_rank_0(f"Saving model at {samples_seen} samples")
    start = time.time()
    rank = torch.distributed.get_rank()
    save_directory = Path(output_dir) / "hf_format" / f"samples_{samples_seen}"
    os.makedirs(save_directory, exist_ok=True)
    # Get full state dict
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
    state_dict = get_model_state_dict(fsdp_model, options=StateDictOptions(full_state_dict=True))
    inner = getattr(fsdp_model, "module", fsdp_model)
    if hasattr(inner, "prepare_state_dict_for_save"):
        state_dict = inner.prepare_state_dict_for_save(state_dict)
    state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
    
    if rank == 0:
        pattern = "model{suffix}.safetensors"
        index_name = "model.safetensors.index.json"
        
        # Shard splitting
        split = split_torch_state_dict_into_shards(
            state_dict, filename_pattern=pattern, max_shard_size="5GB",
        )
        # Save shards
        for filename, tensors in split.filename_to_tensors.items():
            shard = {k: state_dict[k] for k in tensors}
            path = os.path.join(save_directory, filename)
            save_file(shard, path)
            
        # Save index if sharded
        if split.is_sharded:
            index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
            with open(os.path.join(save_directory, index_name), "w") as f:
                json.dump(index, f, indent=2, sort_keys=True)
        # Save config and tokenizer (unwrap inner module)
        inner = getattr(fsdp_model, "module", fsdp_model)
        inner.config.to_json_file(os.path.join(save_directory, "config.json"))
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.save_pretrained(save_directory)
        log_rank_0(f"\033[1;38;2;0;255;255mSaved model at\033[0m {samples_seen} samples in {time.time() - start:.2f} seconds")
    torch.distributed.barrier()

def reached_stop_condition(
    training_mode: TrainingMode, 
    current_epoch: int, 
    current_step: int, 
    tokens_seen: int,
    max_epochs: int = 0, 
    max_steps: int = 0,
    max_tokens: int = 0
) -> bool:
    """
    Convenience function which determines whether or not training has reached
    a stopping condition based on the training mode.

    Args:
        training_mode: The training mode (EPOCH, STEP, TOKEN, or INFINITE)
        current_epoch: Current epoch number
        current_step: Current step number
        tokens_seen: Total number of loss-counted tokens processed so far
        max_epochs: Maximum epochs (for EPOCH mode)
        max_steps: Maximum steps (for STEP mode)
        max_tokens: Maximum tokens (for TOKEN mode)
    
    Returns:
        bool: True if stopping condition is reached, False otherwise
    """
    match training_mode:
        case TrainingMode.EPOCH:
            return max_epochs > 0 and current_epoch >= max_epochs
        case TrainingMode.STEP:
            return max_steps > 0 and current_step >= max_steps
        case TrainingMode.TOKEN:
            return max_tokens > 0 and tokens_seen >= max_tokens
        case TrainingMode.INFINITE:
            return False  # Never stop for infinite mode
        case _:
            raise ValueError(f"Unknown training mode: {training_mode}")


def validate_training_mode(
    training_mode: TrainingMode = TrainingMode.INFINITE,
    max_epochs: int = 0,
    max_steps: int = 0,
    max_tokens: int = 0,
) -> TrainingMode:
    """
    Validates that the given training mode is valid, and ensures that
    the provided options are being used as expected.

    When the training mode was provided as a string, this function
    returns the corresponding
    """
    # # Convert string training mode to TrainingMode enum if needed
    # if isinstance(training_mode, str):
    #     try:
    #         training_mode = TrainingMode(training_mode.lower())
    #     except ValueError:
    #         valid_modes = [mode.value for mode in TrainingMode]
    #         raise ValueError(f"Invalid training mode: '{training_mode}'. Valid modes are: {valid_modes}")
    
    # Validate training mode and corresponding parameters
    if training_mode == TrainingMode.EPOCH and max_epochs <= 0:
        raise ValueError("EPOCH training mode requires max_epochs > 0")
    elif training_mode == TrainingMode.STEP and max_steps <= 0:
        raise ValueError("STEP training mode requires max_steps > 0")
    elif training_mode == TrainingMode.TOKEN and max_tokens <= 0:
        raise ValueError("TOKEN training mode requires max_tokens > 0")


def train(
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler, 
        data_loader: torch.utils.data.DataLoader, 
        output_dir: str, 
        min_samples_per_checkpoint: int | None, 
        model_name_or_path: str,
        training_mode: TrainingMode = TrainingMode.INFINITE,
        max_epochs: int = 0,
        max_steps: int = 0,
        max_tokens: int = 0,
        checkpoint_at_epoch: bool = False,
        save_final_checkpoint: bool = False,
    ):
    """
    Runs the model training loop.

    Runs the model training loop with FSDP (Fully Sharded Data Parallel) support.

    This function handles the complete training process including gradient accumulation,
    checkpointing, logging, and distributed training coordination. It supports four
    different training modes: epoch-based, step-based, token-based, and infinite.

    Args:
        model (torch.nn.Module): The model to train, typically wrapped with FSDP.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        lr_scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler.
        data_loader (torch.utils.data.DataLoader): DataLoader providing training batches.
        output_dir (str): Directory path where checkpoints and logs will be saved.
        min_samples_per_checkpoint (int | None): Minimum number of samples between checkpoints. If None, sample-based checkpointing is disabled.
        model_name_or_path (str): Path or identifier of the base model for tokenizer loading.
        training_mode (Union[TrainingMode, str], optional): Training mode - EPOCH, STEP, TOKEN, or INFINITE. Can be either a TrainingMode enum or string value. Defaults to INFINITE.
        max_epochs (int, optional): Maximum number of epochs (for EPOCH mode). Defaults to 0.
        max_steps (int, optional): Maximum number of steps (for STEP mode). Defaults to 0.
        max_tokens (int, optional): Maximum number of loss-counted tokens (for TOKEN mode). Defaults to 0.
        checkpoint_at_epoch (bool, optional): Whether to save checkpoints at epoch end. Defaults to False.
        save_final_checkpoint (bool, optional): Whether to save a final checkpoint at training end. Defaults to False.

    Note:
        The training_mode can be provided as either a TrainingMode enum value or a string:
        - "epoch" or TrainingMode.EPOCH: requires max_epochs > 0
        - "step" or TrainingMode.STEP: requires max_steps > 0
        - "token" or TrainingMode.TOKEN: requires max_tokens > 0
        - "infinite" or TrainingMode.INFINITE: runs indefinitely until manually stopped

    Raises:
        RuntimeError: If distributed training is not properly initialized.
        ValueError: If training mode requirements are not met.
    """
    # just ensure that the way we are being prompted to train is correct
    validate_training_mode(training_mode, max_epochs, max_steps, max_tokens)

    # set model into training mode
    model.train()

    # control args 
    metric_logger = AsyncStructuredLogger(output_dir + f"/training_metrics_{os.environ.get('RANK')}.jsonl")
    world_size = int(os.environ["WORLD_SIZE"])
    is_main_process = dist.get_rank() == 0

    # initialize variables
    batch_totals = BatchMetrics()
    step = 0
    total_samples_accumulated = 0
    total_tokens_processed = 0  # Track total loss-counted tokens for TOKEN mode
    last_saved_samples = 0
    device = next(model.parameters()).device
    epoch = 0

    # main training loop
    while not reached_stop_condition(
        training_mode=training_mode,
        current_epoch=epoch,
        current_step=step,
        tokens_seen=total_tokens_processed,
        max_epochs=max_epochs,
        max_steps=max_steps,
        max_tokens=max_tokens
    ):
        # set the current epoch
        data_loader.sampler.set_epoch(epoch)
        data_loader_it = iter(data_loader)
        for batch in data_loader_it:
            batch_start_time = time.time()
            batch_totals.reset_batch()
            torch.cuda.reset_peak_memory_stats()
            for grad_accum, mb in enumerate(batch):

                mb_start_time = time.time()
                
                # here is everything in the minibatch

                # return {
                #     "input_ids": torch.tensor([input_ids], dtype=torch.long),
                #     "labels": torch.tensor([labels], dtype=torch.long),
                #     "position_ids": torch.tensor([position_ids], dtype=torch.long),
                #     "num_loss_counted_tokens": num_loss_counted_tokens,
                #     "num_samples": num_samples,
                #     "batch_num_loss_counted_tokens": batch_num_loss_counted_tokens,
                # }


                mb_num_loss_counted_tokens = mb['num_loss_counted_tokens']
                mb_num_samples = mb['num_samples']
                batch_num_loss_counted_tokens = mb['batch_num_loss_counted_tokens']
                
                # be explicit about what gets sent to the device
                model_inputs = {
                    'input_ids': mb['input_ids'].to(device),
                    'labels': mb['labels'].to(device),
                    'position_ids': mb['position_ids'].to(device),
                }

                # torch.distributed.breakpoint()
                output = model(**model_inputs)
                loss = output.loss.float().sum() 
                loss_metrics = loss.detach().item()
                '''multiply by world_size to account for the fact that fsdp takes the mean of the gradients across the world_size'''
                '''the loss is a sum of all cross entropy losses for all tokens in the batch, we divide by batch_num_loss_counted_tokens to get the average loss per token'''
                loss = loss * world_size / batch_num_loss_counted_tokens

                loss.backward()
                torch.cuda.empty_cache()

                batch_totals.accumulate_minibatch_metrics(
                    num_loss_counted_tokens=mb_num_loss_counted_tokens,
                    num_total_tokens=mb['input_ids'].shape[1],
                    num_samples=mb_num_samples,
                    loss=loss_metrics,
                    loss_backward=loss.detach().item()/world_size,
                    time_per_minibatch=time.time() - mb_start_time,
                )
            step += 1
            #sum the metrics from all processes
            batch_totals.reduce_batch_metrics(device)

            #use accumulated metrics to take a gradient step and logging
            bm = batch_totals.totals
            total_samples_accumulated += bm['num_samples']
            total_tokens_processed += batch_num_loss_counted_tokens  # Track tokens for TOKEN mode
            grad_norm = take_gradient_step(model, optimizer, lr_scheduler)

            if is_main_process:
                batch_time = time.time() - batch_start_time
                batch_metrics = {
                        "step": step,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "grad_norm": grad_norm.item(),
                        "loss": bm['loss']/batch_num_loss_counted_tokens,
                        "avg_loss_backward": bm['loss_backward']/(grad_accum+1),
                        "num_samples": bm['num_samples'],
                        "num_loss_counted_tokens": bm['num_loss_counted_tokens'],
                        "batch_num_loss_counted_tokens": batch_num_loss_counted_tokens,
                        "num_total_tokens": bm['num_total_tokens'],
                        "grad_accum": grad_accum+1,
                        "avg_time_per_minibatch": bm['time_per_minibatch']/(grad_accum+1)/world_size,
                        "time_per_batch": batch_time,
                        "tokens_per_second": bm['num_total_tokens']/batch_time,
                        "total_samples_accumulated": total_samples_accumulated, 
                        "samples_per_second": bm['num_samples']/batch_time,
                        "peak_memory_usage_GB": float(torch.cuda.max_memory_allocated() / 1e9),
                    }
                metric_logger.log_sync(
                    batch_metrics
                )

            torch.distributed.barrier()
            
            # sample-based saving, keep in the inner loop
            if min_samples_per_checkpoint is not None and total_samples_accumulated - last_saved_samples >= min_samples_per_checkpoint:
                save_model(model, total_samples_accumulated, output_dir, model_name_or_path)
                last_saved_samples = total_samples_accumulated
            
            # Check stopping condition after each step (for STEP and TOKEN modes)
            if reached_stop_condition(
                training_mode=training_mode,
                current_epoch=epoch,
                current_step=step,
                tokens_seen=total_tokens_processed,
                max_epochs=max_epochs,
                max_steps=max_steps,
                max_tokens=max_tokens
            ):
                break
        
        # Increment epoch counter after completing an epoch
        epoch += 1
        
        # save at the current number of samples seen. Do not record `last_saved_samples`
        # since this shouldn't interefere with frequency-based saving
        if checkpoint_at_epoch:
            save_model(model, total_samples_accumulated, output_dir, model_name_or_path)
    
    if save_final_checkpoint:
        save_model(model, total_samples_accumulated, output_dir, model_name_or_path)




@app.command()
def main(
    # the '...' is a way of defining required options/arguments without breaking Python's
    # positional vs keyword argument rules
    model_name_or_path: Annotated[str, Option(help="Model name or path")] = ...,
    data_path: Annotated[str, Option(help="Path to the training data JSONL file")] = ...,
    batch_size: Annotated[int, Option(help="Initial batch size before dynamic splitting")] = ...,
    max_tokens_per_gpu: Annotated[int, Option(help="Maximum tokens per GPU per minibatch")] = ...,
    learning_rate: Annotated[float, Option(help="Peak learning rate")] = ...,
    num_warmup_steps: Annotated[int, Option(help="Number of warmup steps for the LR scheduler")] = 0,
    lr_scheduler: Annotated[str, Option(help="Learning rate scheduler type")] = "constant_with_warmup",
    lr_scheduler_kwargs: Annotated[str, Option(help="JSON string of scheduler-specific kwargs")] = "{}",
    seed: Annotated[int, Option(help="Random seed for reproducibility")] = 67,
    use_liger_kernels: Annotated[bool, Option(help="Whether to use Liger kernels")] = False,

    # todo: currently 0.75 rank ratio means unfreezing 25% of the singular values. We want to change this into a variable for which 0 indicates no fine-tuning, and 100% means full fine-tuning
    osft: Annotated[bool, Option(help="Enable SVD based orthogonal subspace training")] = False,
    osft_rank_ratio: Annotated[float, Option(help="Ratio of ranks to use for SVD. Required when osft is True")] = None,  
    osft_target_patterns: Annotated[str, Option(
        help=("List of target modules to use for SVD. When not provided, it will try to guess the patterns based on the model. "
              "This should be a comma-separated list of patterns. "
              "For example, 'self_attn.q_proj,self_attn.k_proj,mlp.gate_proj'")
    )] = None,


    output_dir: Annotated[str, Option(help="Directory to save checkpoints and logs (required)")] = ...,
    logging_level: Annotated[LogLevelEnum, Option(help="Logging level", case_sensitive=False)] = LogLevelEnum.INFO,
    min_samples_per_checkpoint: Annotated[int | None, Option(help="Minimum number of samples processed before saving a checkpoint (required)")] = None,

    # Training mode parameters
    training_mode: Annotated[TrainingMode, Option(help="Training mode: epoch, step, token, or infinite", case_sensitive=False)] = TrainingMode.INFINITE,  # todo: change this
    max_epochs: Annotated[int, Option(help="Maximum number of epochs (for epoch mode)")] = 0,
    max_steps: Annotated[int, Option(help="Maximum number of steps (for step mode)")] = 0,
    max_tokens: Annotated[int, Option(help="Maximum number of loss-counted tokens (for token mode)")] = 0,
    checkpoint_at_epoch: Annotated[bool, Option(help="Whether to save checkpoints at the end of each epoch")] = False,
    save_final_checkpoint: Annotated[bool, Option(help="Whether to save a final checkpoint when training ends")] = False,
):
    
    
    init_distributed_environment()
    # TODO: make the path creation lazy, but confirm that we can write to the given directory
    # at this point
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # validation, do this before continuing execution flow so we don't log experiments that are invalid from
    # the get-go
    if osft:
        if osft_rank_ratio is None:
            raise ValueError("osft_rank_ratio is required when osft is True")
        if osft_target_patterns:
            osft_target_patterns = osft_target_patterns.replace("'", "").replace('"', "").replace(" ", "").split(",")
    
    # Log parameters only on rank 0
    rank = dist.get_rank()
    if rank == 0:
        params = {
            "model_name_or_path": model_name_or_path,
            "data_path": data_path,
            "batch_size": batch_size,
            "max_tokens_per_gpu": max_tokens_per_gpu,
            "learning_rate": learning_rate,
            "num_warmup_steps": num_warmup_steps,
            "lr_scheduler": lr_scheduler,
            "seed": seed,
            "use_liger_kernels": use_liger_kernels,
            "osft": osft,
            "osft_rank_ratio": osft_rank_ratio,
            "osft_target_patterns": osft_target_patterns,
            "output_dir": output_dir,
            "logging_level": logging_level.value,
            "min_samples_per_checkpoint": min_samples_per_checkpoint,
            "training_mode": training_mode.value,
            "max_epochs": max_epochs,
            "max_steps": max_steps,
            "max_tokens": max_tokens,
            "checkpoint_at_epoch": checkpoint_at_epoch,
            "save_final_checkpoint": save_final_checkpoint,
            "RANK": rank, # Include rank itself, though it will be 0 here
            "WORLD_SIZE": int(os.environ.get("WORLD_SIZE", 1))
        }
        params_path = output_path / "training_params.json"
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=4)
        # Pretty print parameters in a single line using JSON
        print(f"Training with parameters: {json.dumps(params, separators=(',', ':'), indent=4)}")
        print(f"Training parameters saved to {params_path}")

    setup_logger(level=logging_level.value)

    # Parse scheduler kwargs from JSON string
    import json as json_module
    try:
        scheduler_kwargs_dict = json_module.loads(lr_scheduler_kwargs) if lr_scheduler_kwargs else {}
    except json_module.JSONDecodeError:
        log_rank_0(f"Warning: Invalid JSON for lr_scheduler_kwargs: {lr_scheduler_kwargs}. Using empty dict.")
        scheduler_kwargs_dict = {}
    
    # grab the data loader prior to the model so we can extract the dataset length
    # and use this for calculating the number of training steps in the data loader
    data_loader = get_data_loader(
        data_path=data_path,
        batch_size=batch_size,
        max_tokens_per_gpu=max_tokens_per_gpu,
        seed=seed,
    )
    
    # Calculate number of training steps based on training mode
    num_training_steps = calculate_num_training_steps(
        training_mode=training_mode,
        data_loader=data_loader,
        max_epochs=max_epochs,
        max_steps=max_steps,
        max_tokens=max_tokens,
    )
    
    log_rank_0(f"Calculated num_training_steps: {num_training_steps}")
    
    # If Orthogonal Subspace Learning is enabled, loads a model with decomposed trainable low-rank + fixed high-rank subspace weights (see svd_utils)
    model = setup_model(
        model_name_or_path=model_name_or_path,
        use_liger_kernels=use_liger_kernels,
        osft=osft,
        rank=rank,
        osft_rank_ratio=osft_rank_ratio,
        osft_target_patterns=osft_target_patterns,
    )
    model, optimizer, lr_scheduler = setup_training_components(
        model=model,
        learning_rate=learning_rate,
        num_warmup_steps=num_warmup_steps,
        lr_scheduler=lr_scheduler,
        num_training_steps=num_training_steps,
        scheduler_kwargs=scheduler_kwargs_dict,
    )
    
    train(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        data_loader=data_loader,
        output_dir=output_dir,
        min_samples_per_checkpoint=min_samples_per_checkpoint,
        model_name_or_path=model_name_or_path,
        training_mode=training_mode,
        max_epochs=max_epochs,
        max_steps=max_steps,
        max_tokens=max_tokens,
        checkpoint_at_epoch=checkpoint_at_epoch,
        save_final_checkpoint=save_final_checkpoint
    )
    
if __name__ == "__main__":
    app()


'''
rclone copy --copy-links /new_data/experiments_rh/phi-4_limo_trainer_pipe_cleaner/hf_format/samples_8192.0 /dev/shm/phi-4_limo_trainer_pipe_cleaner_cont
        --data-path /dev/shm/knowledge_processed.jsonl \
        --data-path ./some_product_puzzle_tokenized_qwen1.5b.jsonl \
        --data-path ./mihir_prob.jsonl \
        --output-dir /new_data/experiments_rh/mihir_prob_qwen1.5b_v2     \
torchrun --nnodes=1 --nproc-per-node=8 train.py \
        --output-dir /cloud/misc/aldo/experiment/qwen32b-expert-iteration-test \
        --data-path ./tokenized_data.jsonl \
        --model-name-or-path Qwen/Qwen3-32B \
        --min-samples-per-checkpoint 3400 \
        --num-warmup-steps 20 \
        --max-tokens-per-gpu 60000              \
        --batch-size 128                       \
        --use-liger-kernels                    \
        --seed 893                               \
        --fsdp-sharding-strategy FULL_SHARD \
        --learning-rate 6e-6
'''
