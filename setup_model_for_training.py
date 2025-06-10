import math
import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from utils import log_rank_0, patch_target_module

def get_module_class_from_name(
    model: torch.nn.Module, name: str
) -> torch.nn.Module | None:
    modules_children = list(model.children())

    if model.__class__.__name__ == name:
        return model.__class__
    elif len(modules_children) == 0:
        return
    else:
        for child_module in modules_children:
            module_class = get_module_class_from_name(child_module, name)
            if module_class is not None:
                return module_class
            
def wrap_fsdp2(model: torch.nn.Module) -> torch.nn.Module:
    """
    Wrap `model` in PyTorch FSDP2 with full sharding and transformer auto-wrap policy under BF16.
    """
    # Determine the block class to auto-wrap (first no-split module)
    block_name = model._no_split_modules[0]
    block_cls = get_module_class_from_name(model, block_name)
    if block_cls is None:
        raise ValueError(f"Could not find module class named {block_name}")
    
    # Mixed-precision policy for BF16
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, 
        reduce_dtype=torch.bfloat16, 
        output_dtype=torch.bfloat16, 
        cast_forward_inputs=True)

    # FSDP2 settings: full shard, BF16, no CPU offload
    fsdp2_kwargs = {
        "mp_policy": mp_policy,
        "reshard_after_forward": True,

    }

    # Auto-wrap child modules
    for module in model.modules():
        if isinstance(module, block_cls):
            fully_shard(module, **fsdp2_kwargs)

    # Wrap the full model
    fully_shard(model, **fsdp2_kwargs)
    model = model.to(torch.float32)
    return model

def align_model_and_tokenizer(model, tokenizer):
    """
    Aligns the model's vocabulary and special tokens with the tokenizer.
    """
    if len(tokenizer) > model.config.vocab_size:
        print(
            f"WARNING: tokenizer has {len(tokenizer)} tokens but model has {model.config.vocab_size} vocab size"
        )
        model.resize_token_embeddings(
            int(8 * math.ceil(len(tokenizer) / 8.0))
        )  # make the vocab size multiple of 8 for sharding the embedding layer.

    # Fix any discrepancy between model and tokenizer
    special_tokens = {
        'pad': ('pad_token_id', 'Fixing model pad token id'),
        'bos': ('bos_token_id', 'Fixing model bos token id'),
        'eos': ('eos_token_id', 'Fixing model eos token id')
    }

    for token_type, (token_attr, message) in special_tokens.items():
        model_token = getattr(model.config, token_attr)
        tokenizer_token = getattr(tokenizer, token_attr)
        
        if (model_token is not None and tokenizer_token is not None 
            and model_token != tokenizer_token):
            log_rank_0(
                "\033[38;5;226m"
                f"WARNING: There is a mismatch between {token_type} token id of "
                f"model({model_token}) and tokenizer({tokenizer_token}). "
                f"{message} to be same as tokenizer's {token_type} token id"
                "\033[0m"
            )
            setattr(model.config, token_attr, tokenizer_token)

    return model

def setup_model(model=None, orthogonal_subspace_learning: bool = False, **kwargs):
    base_model_args = {
        "pretrained_model_name_or_path": kwargs['model_name_or_path'],
        "torch_dtype": torch.bfloat16,
    }
    base_model_args["attn_implementation"] = "flash_attention_2"

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(kwargs['model_name_or_path'])

    if kwargs['use_liger_kernels']:
        '''need to patch the loss function to not reduce, so we can reduce across all GPUs'''
        from none_reduction_losses import liger_fixed_fused_linear_cross_entropy_none_reduction
        patch_target_module(
            "liger_kernel.transformers.model.loss_utils.fixed_fused_linear_cross_entropy",
            liger_fixed_fused_linear_cross_entropy_none_reduction,
        )
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        model = AutoLigerKernelForCausalLM.from_pretrained(**base_model_args)
        model = align_model_and_tokenizer(model, tokenizer)
    else:
        from none_reduction_losses import hf_fixed_cross_entropy_none_reduction
        patch_target_module(
            "transformers.loss.loss_utils.fixed_cross_entropy",
            hf_fixed_cross_entropy_none_reduction,
        )
        if orthogonal_subspace_learning:
            from svd_utils import create_svd_model_class, auto_generate_target_svd_config
            tmp = AutoModelForCausalLM.from_pretrained(**base_model_args)
            tmp = align_model_and_tokenizer(tmp, tokenizer)
            svd_cfg = auto_generate_target_svd_config(tmp)
            svd_cls = create_svd_model_class(tmp.__class__)
            cfg = tmp.config
            del tmp
            torch.cuda.empty_cache()
            model = svd_cls.from_pretrained(
                **base_model_args,
                config=cfg,
                svd_config=svd_cfg,
                initialize_svd=False,
            )
            model = align_model_and_tokenizer(model, tokenizer)
            model.reinitialize_svd()
        else:
            model = AutoModelForCausalLM.from_pretrained(**base_model_args)
            model = align_model_and_tokenizer(model, tokenizer)

    if model.__class__.__name__ not in [
        "MistralForCausalLM",
        "GPTDolomiteForCausalLM", 
        "LlamaForCausalLM",
        "Starcoder2ForCausalLM",
        "GemmaForCausalLM",
        "MixtralForCausalLM",
        "GraniteForCausalLM",
    ]:
        log_rank_0(
            f"\033[38;2;255;255;0mWarning: Model class name: {model.__class__.__name__} is not in the list of supported models.\033[0m",
            to_print=True,
        )

    model.gradient_checkpointing_enable()
    # torch.compile(model)
    return model

def setup_training_components(model, **kwargs):
    from transformers import get_scheduler
    model = wrap_fsdp2(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=kwargs['learning_rate'],
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    lr_scheduler = get_scheduler(
        name=kwargs['lr_scheduler'],
        optimizer=optimizer,
        num_warmup_steps=kwargs['num_warmup_steps'],
    )
    lr_scheduler.split_batches = True
    lr_scheduler.step() #the scheduler starts at 0 and there's no learning.
    return model, optimizer, lr_scheduler

if __name__ == "__main__":
    from utils import init_distributed_environment
    from torch.distributed.checkpoint.state_dict import get_model_state_dict
    init_distributed_environment()
    cpu_pg = dist.new_group(backend="gloo")
    # model_name_or_path = '/dev/shm/Llama-3.1-8B-Instruct/'
    # model_name_or_path = '/dev/shm/test_save'
    model_name_or_path = 'Qwen/Qwen2.5-1.5B-instruct'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = setup_model(model_name_or_path=model_name_or_path, use_liger_kernels=True)
    model, optimizer, lr_scheduler = setup_training_components(model, 
                                                        learning_rate=1e-5,
                                                        num_warmup_steps=10,
                                                        lr_scheduler="constant_with_warmup")
    import os
    inputs = tokenizer("Hello FSDP2!", return_tensors="pt").to(int(os.environ["LOCAL_RANK"]))
    outputs = model(**inputs, labels=inputs.input_ids)
    print(f"Output logits shape: {outputs.logits.shape}")
    # state_dict = model.state_dict()
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
    state_dict = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True))
    torch.distributed.barrier()
    # from test_async_save import ModelWrapper
    # wrapper = ModelWrapper(model)
    ckpt_path = os.path.abspath("fsdp2_ckpt")
    from test_model_wrap import save_model
    save_model(model, tokenizer, ckpt_path)
    model_ = setup_model(model_name_or_path=ckpt_path, use_liger_kernels=True)

    # future = async_save(state_dict, checkpoint_id=ckpt_path, process_group=cpu_pg)
    # print(f"Async save started: {ckpt_path}")
    # future.result()
    # print(f"Async save finished: {ckpt_path}")

    if os.environ.get("RANK") == "0":
        from IPython import embed; embed()
    torch.distributed.checkpoint.load(state_dict, checkpoint_id=ckpt_path)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    # import shutil
    # output_dir = Path("/new_data/experiments_rh/llama_knowledge_mini_trainer_pipe_cleaner_v2/hf_format/test_save")
    # # output_dir = Path("/dev/shm/test_save")
    # shutil.rmtree(output_dir, ignore_errors=True)
    # output_dir.mkdir(parents=True, exist_ok=True)
    # accelerator.save_model(model,
    #                     str(output_dir),
    #                     max_shard_size="5GB",
    #                     safe_serialization=True,
    # )
    # if accelerator.is_main_process:
    #     from transformers import AutoTokenizer
    #     model.module.config.to_json_file(str(output_dir / "config.json"))
    #     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    #     tokenizer.save_pretrained(output_dir)
    #     from IPython import embed
    #     embed()
    # torch.distributed.barrier()
    # model = setup_model(model_name_or_path=output_dir, use_liger_kernels=True)

'''
torchrun --nnodes=1 --nproc-per-node=8 setup_model_for_training.py
'''