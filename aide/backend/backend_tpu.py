"""
TPU-compatible backend for AIDE agent using PyTorch/XLA
This module provides TPU support while maintaining API compatibility with backend_local.py
"""

import logging
import time
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console

logger = logging.getLogger("aide.backend.tpu")
console = Console()

# Try to import TPU/XLA support
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    TPU_AVAILABLE = True
    logger.info("PyTorch/XLA detected - TPU support enabled")
except ImportError:
    TPU_AVAILABLE = False
    logger.warning("PyTorch/XLA not found - TPU support disabled")


class TPUModelManager:
    """
    Model manager for TPU using PyTorch/XLA
    API-compatible with LocalLLMManager from backend_local.py
    """
    _cache = {}  # Cache to store loaded models

    @classmethod
    def get_device(cls):
        """Get the appropriate device (TPU, CUDA, or CPU)"""
        if TPU_AVAILABLE:
            device = xm.xla_device()
            logger.info(f"Using XLA device: {device}")
            return device
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info("Falling back to CUDA device")
            return device
        else:
            device = torch.device('cpu')
            logger.info("Falling back to CPU device")
            return device

    @classmethod
    def get_model(
        cls, 
        model_name: str, 
        use_bfloat16: bool = True,
        force_backend: Optional[str] = None
    ) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """
        Load or retrieve a model from cache for TPU
        
        Args:
            model_name: HuggingFace model identifier
            use_bfloat16: Use bfloat16 precision (recommended for TPU)
            force_backend: Force specific backend ('tpu', 'cuda', 'cpu', or None for auto)
        
        Returns:
            Tuple of (tokenizer, model)
        """
        cache_key = f"{model_name}_{force_backend or 'auto'}"
        
        if cache_key not in cls._cache:
            cls.clear_cache()  # Clear cache before loading new model
            logger.info(f"Loading model: {model_name} for TPU")
            
            try:
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    trust_remote_code=True
                )
                
                # Set padding token
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
                
                tokenizer.padding_side = "left"
                
                # Determine device
                if force_backend == 'tpu' and not TPU_AVAILABLE:
                    raise RuntimeError("TPU backend requested but PyTorch/XLA not available")
                
                device = cls.get_device()
                
                # Set dtype (bfloat16 is optimal for TPU)
                dtype = torch.bfloat16 if use_bfloat16 else torch.float32
                logger.info(f"Using dtype: {dtype}")
                
                # Load model
                # Note: For TPU, we don't use quantization_config or device_map
                # Instead, we load the model and explicitly move it to device
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    # Don't use device_map="auto" for TPU
                    # Don't use quantization_config for TPU
                )
                
                # Move model to device
                logger.info(f"Moving model to device: {device}")
                model = model.to(device)
                
                # Set model to eval mode
                model.eval()
                
                # Warm up TPU/XLA compilation (first inference triggers compilation)
                if TPU_AVAILABLE:
                    logger.info("Warming up TPU/XLA compilation...")
                    try:
                        dummy_input = tokenizer("Warmup", return_tensors="pt")
                        dummy_input = {k: v.to(device) for k, v in dummy_input.items()}
                        
                        with torch.no_grad():
                            _ = model.generate(
                                **dummy_input,
                                max_new_tokens=10,
                                do_sample=False
                            )
                        xm.mark_step()  # Synchronize TPU operations
                        logger.info("TPU warmup complete")
                    except Exception as e:
                        logger.warning(f"TPU warmup failed (non-critical): {e}")
                
                logger.info(f"Model '{model_name}' loaded successfully on {device}")
                cls._cache[cache_key] = (tokenizer, model, device)
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise
        
        return cls._cache[cache_key]

    @classmethod
    def clear_cache(cls, model_name: Optional[str] = None) -> None:
        """Clear specific model or entire cache to free memory"""
        if model_name:
            # Clear specific model
            keys_to_remove = [k for k in cls._cache.keys() if k.startswith(model_name)]
            for key in keys_to_remove:
                cls._cache.pop(key, None)
            logger.info(f"Cleared cache for model: {model_name}")
        else:
            cls._cache.clear()
            logger.info("Cleared entire model cache")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # For TPU, also clear XLA cache
        if TPU_AVAILABLE:
            try:
                # This helps free TPU memory
                import torch_xla.core.xla_model as xm
                xm.mark_step()
            except Exception as e:
                logger.debug(f"Could not clear XLA cache: {e}")

    @classmethod
    def generate_response(
        cls,
        model_name: str,
        prompt: str,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        device: torch.device,
        system_message: Optional[str] = None,
        user_message: Optional[str] = None,
        num_responses: int = 1,
        **gen_kwargs: Any,
    ) -> Tuple[str, int, int, float]:
        """
        Generate response using TPU
        
        Returns:
            Tuple of (response_text, prompt_length, output_length, latency)
        """
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        prompt_length = input_ids.shape[1]
        
        # Set generation parameters
        gen_kwargs = {
            "temperature": gen_kwargs.get("temperature", 0.6),
            "max_new_tokens": gen_kwargs.get("max_new_tokens", 2048),
            "top_p": gen_kwargs.get("top_p", 0.9),
            "top_k": gen_kwargs.get("top_k", 50),
            "repetition_penalty": gen_kwargs.get("repetition_penalty", 1.1),
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": gen_kwargs.get("do_sample", True),
            "num_return_sequences": num_responses,
        }
        
        # Filter out None values
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        
        logger.info(f"Generating {num_responses} response(s) on TPU...")
        t0 = time.time()
        
        try:
            # Generate
            with torch.no_grad():
                generated_outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
            
            # Synchronize TPU operations
            if TPU_AVAILABLE:
                xm.mark_step()
            
            # Decode outputs
            outputs = []
            for i in range(num_responses):
                output_ids = generated_outputs[i, prompt_length:]
                output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                outputs.append(output_text.strip())
            
            # Use first response as primary output
            output = outputs[0] if outputs else ""
            output_length = len(generated_outputs[0]) - prompt_length
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise ValueError(f"Failed to generate response: {str(e)}")
        
        t1 = time.time()
        latency = t1 - t0
        
        logger.info(
            f"Generated {num_responses} response(s) "
            f"({output_length} tokens) in {latency:.2f}s"
        )
        
        return output, prompt_length, output_length, latency


def query(
    system_message: Optional[str] = None,
    user_message: Optional[str] = None,
    model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    func_spec=None,
    excute: bool = False,
    planner_flag: bool = False,
    num_responses: int = 1,
    output_dir: Optional[Path] = None,
    step_identifier: str = "step",
    force_backend: Optional[str] = None,
    **model_kwargs: Any,
) -> Tuple[Optional[List[str]], float, int, int, Optional[Dict[str, Any]]]:
    """
    Query the model with TPU support
    
    API-compatible with backend_local.query()
    
    Args:
        system_message: Optional system prompt
        user_message: Optional user prompt
        model: Model name (HuggingFace identifier)
        num_responses: Number of responses to generate
        force_backend: Force specific backend ('tpu', 'cuda', 'cpu', or None)
        **model_kwargs: Additional generation arguments
    
    Returns:
        Tuple: (responses, latency, input_tokens, output_tokens, metadata)
    """
    t0 = time.time()
    raw_responses: Optional[List[str]] = None
    info: Optional[Dict[str, Any]] = {"model_name": model}
    input_token_count = 0
    output_token_count = 0
    
    try:
        # Load model
        tokenizer, model_instance, device = TPUModelManager.get_model(
            model, 
            force_backend=force_backend
        )
        
        # Format prompt
        console.rule(f"[bold red]System Prompt for {step_identifier}")
        logger.info(f"{system_message or 'None'}", extra={"verbose": True})
        console.rule(f"[bold red]User Prompt for {step_identifier}")
        logger.info(f"{user_message or 'None'}", extra={"verbose": True})
        
        # Create messages
        from aide.backend.utils import opt_messages_to_list
        messages = opt_messages_to_list(
            system_message,
            user_message,
            convert_system_to_user=model_kwargs.pop("convert_system_to_user", False),
        )
        
        # Apply chat template
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                logger.info("Applied chat template to prompt")
            except Exception as e:
                logger.warning(f"Could not apply chat template ({e}). Using concatenation.")
                prompt_text = (system_message or "") + "\n\n" + (user_message or "")
        else:
            prompt_text = (system_message or "") + "\n\n" + (user_message or "")
        
        # Generate
        logger.debug(f"Generating with params: num_responses={num_responses}, {model_kwargs}")
        
        raw_response, input_len, output_len, latency_gen = (
            TPUModelManager.generate_response(
                model_name=model,
                tokenizer=tokenizer,
                model=model_instance,
                device=device,
                prompt=prompt_text,
                num_responses=num_responses,
                **model_kwargs,
            )
        )
        
        raw_responses = raw_response
        input_token_count = input_len
        output_token_count = output_len
        
        # Handle execution if requested
        if excute:
            from aide.backend.backend_local import process_and_execute_responses
            exec_timeout = model_kwargs.get("exec_timeout", 20)
            info = process_and_execute_responses(
                responses=[raw_responses] if isinstance(raw_responses, str) else raw_responses,
                output_base_dir=output_dir or Path("./outputs"),
                interpreter_timeout=exec_timeout,
                step_identifier=step_identifier,
            )
        
    except Exception as e:
        logger.error(f"Query failed for model {model}: {e}", exc_info=True)
        info["error"] = str(e)
        raw_responses = "None"
    
    finally:
        latency = time.time() - t0
        logger.info(f"Total query latency: {latency:.2f}s")
    
    return raw_responses, latency, input_token_count, output_token_count, info


# Convenience functions for backend detection
def is_tpu_available() -> bool:
    """Check if TPU is available"""
    return TPU_AVAILABLE


def get_backend_info() -> Dict[str, Any]:
    """Get information about available backends"""
    info = {
        "tpu_available": TPU_AVAILABLE,
        "cuda_available": torch.cuda.is_available(),
        "recommended_backend": "tpu" if TPU_AVAILABLE else ("cuda" if torch.cuda.is_available() else "cpu"),
    }
    
    if TPU_AVAILABLE:
        try:
            device = xm.xla_device()
            info["tpu_device"] = str(device)
            info["tpu_ordinal"] = xm.get_ordinal()
        except Exception as e:
            info["tpu_error"] = str(e)
    
    return info


if __name__ == "__main__":
    # Test script
    print("=" * 60)
    print("TPU Backend Test")
    print("=" * 60)
    
    backend_info = get_backend_info()
    print("\nBackend Information:")
    for key, value in backend_info.items():
        print(f"  {key}: {value}")
    
    if TPU_AVAILABLE:
        print("\n✓ TPU backend ready!")
        print("\nTesting model loading...")
        try:
            tokenizer, model, device = TPUModelManager.get_model(
                "HuggingFaceTB/SmolLM-135M-Instruct"
            )
            print(f"✓ Model loaded successfully on {device}")
        except Exception as e:
            print(f"✗ Model loading failed: {e}")
    else:
        print("\n✗ TPU not available")
        print("Install PyTorch/XLA: pip install torch torch_xla")




