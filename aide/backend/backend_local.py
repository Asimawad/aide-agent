import re
import logging
import torch
from typing import Optional, Dict, Any, Tuple, List # Add List
import sys
import shutil # Import shutil for file operations
from pprint import pprint
import traceback  # Import traceback for error handling
from pathlib import Path
import time

# --- Add necessary imports ---
try:
    # Assuming aide-ds is the parent directory structure you provided earlier
    script_dir_backend = Path(__file__).resolve().parent
    aide_ds_root = script_dir_backend.parent.parent # Go up two levels from backend

    if aide_ds_root.is_dir() and (aide_ds_root / 'aide').is_dir():
         sys.path.insert(0, str(aide_ds_root))
    else:
         # Fallback if structure is different - might need manual adjustment
         print(f"Warning: Could not reliably find aide-ds root from {script_dir_backend}. Imports might fail.")
         # Assuming aide is installed in the environment
         pass

    from aide.interpreter import Interpreter, ExecutionResult
    from aide.utils.response import extract_code, format_code, wrap_code
    from aide.utils import serialize # For saving ExecutionResult
    from aide.backend.utils import opt_messages_to_list # Keep opt_messages_to_list if needed here
    from aide.utils.config import load_cfg
    cfg = load_cfg()

except ImportError as e:
    print(f"Error importing AIDE modules: {e}")
    print("Please ensure AIDE is installed correctly (e.g., `pip install -e .` from aide-ds root) or paths are set up.")
    # sys.exit(1) # Don't exit if just using as a library potentially

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from rich.console import Console
from rich.syntax import Syntax


logger = logging.getLogger("aide") # More specific logger name
console = Console()

class LocalLLMManager:
    _cache = {}  # Cache to store loaded models

    @classmethod
    def get_model(cls, model_name: str,load_in_4bit:bool = True) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        
        """Load or retrieve a model from cache with/without 4-bit quantization."""
        if model_name not in cls._cache:
            cls.clear_cache()
            logger.info(f"Loading local model: {model_name}", extra={"verbose": True})
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
                # Set padding token to avoid attention mask issues
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}", extra={"verbose": True})

                tokenizer.padding_side = "left" # Important for batch generation if ever implemented
                quantization_config = None
                if load_in_4bit and torch.cuda.is_available():
                    quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    )
                    logger.info("Using 4-bit quantization (BitsAndBytesConfig).", extra={"verbose": True})

                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config, #load unquantized model
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if quantization_config else None,
                    trust_remote_code=True,
                )
                if load_in_4bit:
                    logger.info(f"Quantized (4-bit) local model '{model_name}' loaded successfully.")
                else:
                    logger.info(f"Unquantized local model '{model_name}' loaded successfully.")
                # Move model to GPU if available
                cls._cache[model_name] = (tokenizer, model)
            except Exception as e:
                logger.error(f"Failed to load local model {model_name}: {e}")
                raise
        return cls._cache[model_name]

    @classmethod
    def clear_cache(cls, model_name: Optional[str] = None) -> None:
        """Clear specific model or entire cache to free memory."""
        if model_name:
            cls._cache.pop(model_name, None)
            logger.info(f"Cleared cache for model: {model_name}", extra={"verbose": True})
        else:
            cls._cache.clear()
            logger.info("Cleared entire model cache", extra={"verbose": True})
    @classmethod
    def generate_response(
        cls,
        model_name: str,
        prompt:str,
        tokenizer :AutoTokenizer ,
        model:AutoModelForCausalLM,
        system_message: Optional[str] = None,
        user_message: Optional[str] = None,
        num_responses:int = 1,
        **gen_kwargs: Any,
    ) -> Optional[str]:
        """Generate response with proper system/user prompt handling."""
        
        # Check prompt length against model's max context
        inputs = tokenizer(prompt, return_tensors="pt",return_attention_mask=True)
        input_ids= inputs['input_ids'].to(model.device)
        attention_mask = inputs['attention_mask'].to(model.device)
        prompt_length = input_ids.shape[1]        
        
        gen_kwargs = {
            "temperature": gen_kwargs.get("temperature", 0.6),
            "max_new_tokens": gen_kwargs.get("max_new_tokens", 2048),
            "top_p": gen_kwargs.get("top_p",0.9),
            "top_k": gen_kwargs.get("top_k",50),
            "repetition_penalty": gen_kwargs.get("repetition_penalty"),
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": gen_kwargs.get("do_sample", True) ,
            "num_return_sequences": num_responses,
        }
        # Filter out None values
        gen_kwargs      = {k: v for k, v in gen_kwargs.items() if v is not None}
        
        t0 = time.time()
        outputs = []
        console.rule(f"[bold blue]Generating Responses")

        try:
            # Generate with attention mask
            with torch.no_grad():
                generated_outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
            logger.info("finished generation", extra={"verbose": True})
            # Decode each generated sequence, removing the prompt part
            for i in range(num_responses):
                # generated_outputs shape is (num_responses, seq_len)
                output_ids = generated_outputs[i,prompt_length:]
                output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                outputs.append(output_text.strip())
                output = output_text.strip()

        except Exception as e:
            logger.error(f"Error generating response for {model_name}: {e}")
            raise ValueError(f"Failed to generate response: {str(e)}")

        t1 = time.time()
        latency = t1-t0
        logger.info(f"Batch generation of {num_responses} responses and {len(output_ids)} tokens took {t1-t0:.2f} seconds.")
        return output,prompt_length,len(output_ids),latency
    
def query(
    system_message: Optional[str] = None,
    user_message: Optional[str] = None,
    model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" ,# "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", # Example HF model ID
    func_spec=None,
    excute :bool = False,
    planner = False,
    num_responses: int = 1,
    output_dir: Optional[Path] = None,
    step_identifier: str = "step",
    **model_kwargs: Any,) -> Tuple[Optional[List[str]], float, int, int, Optional[Dict[str, Any]]]:
    """
    Query the local model with system and user messages.

    Args:
        system_message: Optional system prompt (e.g., role instructions).
        user_message: Optional user prompt (e.g., coding task).
        model: Model name (e.g., 'HuggingFaceTB/SmolLM-135M-Instruct').
        temperature: Sampling temperature (default: 0.7).
        max_new_tokens: Maximum new tokens to generate (default: 200).
        **model_kwargs: Additional generation arguments.

    Returns:
        Tuple: (response, latency, input_tokens, output_tokens, metadata)
    """
    
    t0 = time.time()
    raw_responses: Optional[List[str]] = None
    info: Optional[Dict[str, Any]] = None
    input_token_count = 0
    output_token_count = 0 # Represents tokens in the first response for consistency, or total
    info = {"model_name": model}

    if output_dir is None:
        output_base = aide_ds_root / "playground_outputs"
        # Use cfg.exp_name if available, otherwise use a generic name
        exp_name = cfg.exp_name if cfg and hasattr(cfg, 'exp_name') else "default_exp"
        output_dir = output_base / f"exp_{exp_name}" / f"{step_identifier}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        tokenizer, model_instance = LocalLLMManager.get_model(model)

          # 2. Format Prompt
        console.rule(f"[bold red]System Prompt for {step_identifier}")
        logger.info(f"{system_message or 'None'} ", extra={"verbose": True})
        console.rule(f"[bold red]User Prompt for {step_identifier}")
        logger.info(f" {user_message or 'None'} ", extra={"verbose": True})

        messages = opt_messages_to_list(system_message, user_message, convert_system_to_user=model_kwargs.pop('convert_system_to_user', False))
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                logger.info("Applied chat template to prompt.", extra={"verbose": True})
            except Exception as e:
                 logger.warning(f"Could not apply chat template ({e}). Using basic concatenation.")
                 prompt_text = (system_message or "") + "\n\n" + (user_message or "")
       
        logger.debug(f"Generating with params:  num_return_sequences = {num_responses}, {model_kwargs}", extra={"verbose": True})
        
        raw_responses, input_len, output_len_first, latency_gen = LocalLLMManager.generate_response(
            model_name=model,
            tokenizer=tokenizer, 
            model=model_instance,
            prompt=prompt_text,
            num_responses =num_responses,
            do_sample=False,
            **model_kwargs,
        )
        input_token_count = input_len
        # For consistency, report output tokens of the first response, or sum if needed elsewhere
        output_token_count = output_len_first
        
        if excute:   
            exec_timeout = model_kwargs.get("exec_timeout", 20) # Default 20s
            info = process_and_execute_responses(
                responses=raw_responses,
                output_base_dir=output_dir, # Use the determined/passed output dir
                interpreter_timeout=exec_timeout,
                step_identifier=step_identifier
            )

    except Exception as e:
        logger.error(f"Query failed for model {model}: {e}", exc_info=True)
        info["error"] = str(e)
        # Ensure return types match even on failure
        raw_responses = "None"
        latency = time.time() - t0 # Total query time including processing/execution
        logger.info(f"Total query latency (incl. processing/exec): {latency:.2f}s")
        return raw_responses,latency, input_token_count, output_token_count, info

    finally:
        latency = time.time() - t0 # Total query time including processing/execution
        logger.info(f"Total query latency (incl. processing/exec): {latency:.2f}s")

    # Return all collected results
    return raw_responses,latency, input_token_count, output_token_count, info

def process_and_execute_responses(
        responses: List[str],
        output_base_dir: Path,
        interpreter_timeout: int = 20,  # Default timeout
        main_workspace_dir: Path = ".",
        step_identifier: str = "step"  # Identifier for subdirs
    ) -> Dict[str, Any]:

    """
    Processes a list of raw LLM responses: saves raw, extracts/formats/saves code,
    executes code, and saves execution results.

    Args:
        responses: List of raw text responses from the LLM.
        output_base_dir: The base directory to save outputs for this query.
        interpreter_timeout: Timeout in seconds for code execution.
        step_identifier: String to identify this processing step (used in subdir names).

    Returns:
        Tuple containing:
            - List of extracted/formatted code strings (None if extraction failed).
            - List of ExecutionResult objects.
    """
    extracted_codes: List[Optional[str]] = []
    execution_results: List[ExecutionResult] = []
    execution_summaries: List[str] = []

    output_base_dir.mkdir(parents=True, exist_ok=True)

    for i, response in enumerate(responses):
        console.rule(f"[bold blue]Processing Response {i+1}/{len(responses)} for {step_identifier}")
        response_dir = output_base_dir / f"{step_identifier}_response_{i}"
        response_dir.mkdir(exist_ok=True)

        # Save Raw Response
        raw_response_path = response_dir / "raw_response.txt"
        raw_response_path.write_text(response or "<Response was None>") # Handle None case
        logger.info(f"Raw response saved to: {raw_response_path}", extra={"verbose": True})
        console.print(f"[bold cyan]Raw Response {i+1}:[/bold cyan]")
        console.print((response or "<Response was None>")[:1000] + ("..." if response and len(response) > 1000 else ""))
        console.print("-" * 20)

        # Extract and Save Code
        extracted_code = None
        formatted_extracted_code = None
        code_path = response_dir / f"response_{i}_extracted_code.py"
        exec_result = None

        if response: # Only process if response is not None
            extracted_code = extract_code(response)
            if not extracted_code:
                logger.error(f"Response {i+1}: Could not extract valid Python code.")
                console.print(f"[bold red]Response {i+1}: Code extraction FAILED.[/bold red]")
                exec_result = ExecutionResult(term_out=["Code extraction failed."], exec_time=0, exc_type="ExtractionError", exc_info={}, exc_stack=[])
            else:
                try:
                    formatted_extracted_code = format_code(extracted_code)
                    code_path.write_text(formatted_extracted_code)
                    logger.info(f"Extracted code saved to: {code_path}", extra={"verbose": True})
                    console.print(f"[bold green]Extracted Code {i+1}:[/bold green]")
                    console.print(Syntax(formatted_extracted_code, "python", theme="default", line_numbers=True))
                    console.print("-" * 20)
                except Exception as format_e:
                    logger.error(f"Failed to format code for response {i+1}: {format_e}")
                    # Save unformatted if formatting fails
                    code_path.write_text(extracted_code)
                    formatted_extracted_code = extracted_code # Use unformatted for execution attempt
                    console.print(f"[bold yellow]Warning: Saved unformatted code for response {i+1}.[/bold yellow]")



        # Execute code (only if code was extracted)
        if formatted_extracted_code:
            logger.info(f"Executing code for response {i+1} in Workspace :{main_workspace_dir}..")
            interpreter = Interpreter(
                working_dir=main_workspace_dir, #data_dir.parent.parent/ "workspaces" / cfg.exp_name, # Use the dedicated workspace
                timeout=interpreter_timeout
            )
            try:
                exec_result = interpreter.run(formatted_extracted_code, reset_session=True)
            except Exception as e:
                logger.error(f"Interpreter failed for response {i+1}: {e}", exc_info=True)
                # Create a more complete ExecutionResult for interpreter errors
                exec_result = ExecutionResult(
                    term_out=[f"Interpreter Error: {e}", traceback.format_exc()],
                    exec_time=0,
                    exc_type=type(e).__name__,
                    exc_info={'error': str(e)},
                    exc_stack=traceback.extract_tb(e.__traceback__)
                    )
            # finally:
            #     # try:
            #     #     interpreter.cleanup_session()
            #     except Exception as e_clean:
            #         logger.error(f"Error cleaning up interpreter for response {i+1}: {e_clean}")
        elif not extracted_code and response:
             # Already created error result for extraction failure
             pass
        elif not response:
             # Handle case where input response was None
             exec_result = ExecutionResult(term_out=["Input response was None."], exec_time=0, exc_type="InputError", exc_info={}, exc_stack=[])


        # Ensure exec_result is always an ExecutionResult object
        if exec_result is None:
             # This case shouldn't ideally happen if logic above is correct, but as a safeguard:
             logger.error(f"ExecutionResult was unexpectedly None for response {i+1}. Creating error result.")
             exec_result = ExecutionResult(term_out=["Unknown processing error."], exec_time=0, exc_type="ProcessingError", exc_info={}, exc_stack=[])

        # Save execution result
        exec_log_path = response_dir / "execution_log.json"
        try:
            serialize.dump_json(exec_result, exec_log_path)
            logger.info(f"Execution log saved to: {exec_log_path}")
        except Exception as e:
            logger.error(f"Failed to save execution log as JSON for response {i+1}: {e}")
            # Fallback to saving as text
            try:
                exec_log_path.with_suffix(".txt").write_text(str(exec_result.to_dict() if hasattr(exec_result, 'to_dict') else exec_result))
            except Exception as e_txt:
                logger.error(f"Failed to save execution log as text for response {i+1}: {e_txt}")

        
        summary_lines = []
        summary_lines.append(f"[bold magenta]Execution Result {i+1}:[/bold magenta]")
        success = exec_result.exc_type is None
        summary_lines.append(f"  Success: {success}")
        summary_lines.append(f"  Execution Time: {exec_result.exec_time:.2f}s")
        console.print(f"[bold magenta]Execution Result {i+1}:[/bold magenta]")
        console.print(f"  Success: {exec_result.exc_type is None}")
        console.print(f"  Execution Time: {exec_result.exec_time:.2f}s")
        if exec_result.exc_type:
            console.print(f"  Error Type: [bold red]{exec_result.exc_type}[/bold red]")

            summary_lines.append(f"  Error Type: [bold red]{exec_result.exc_type}[/bold red]")
            error_args = exec_result.exc_info.get('args', []) if exec_result.exc_info else []
            if error_args:
                console.print(f"  Error Args: {error_args}")
                summary_lines.append(f"  Error Args: {error_args}")
        console.print(f"  Terminal Output (preview):")
        summary_lines.append(f"  Terminal Output (preview):")
        term_out_list = exec_result.term_out if isinstance(exec_result.term_out, list) else [str(exec_result.term_out)]
        term_out_str = "\n".join(term_out_list)
        summary_lines.append("[dim]" + term_out_str[:500] + ("\n..." if len(term_out_str) > 500 else "") + "[/dim]")

        console.print("[dim]" + term_out_str[:500] + ("\n..." if len(term_out_str) > 500 else "") + "[/dim]")
        console.print(f"  Full output/logs saved in: {response_dir}") # Use relative path
        console.print("-" * 20)

        # Join lines for console print and store the raw string version
        console_summary = "\n".join(summary_lines)
        # Remove rich markup for the stored string summary
        plain_summary = "\n".join([re.sub(r'\[/?.*?\]', '', line) for line in summary_lines])

        console.print(console_summary)
        execution_summaries.append(plain_summary)


        # Append results for this response
        extracted_codes.append(formatted_extracted_code)
        execution_results.append(exec_result)

    return {"extracted_codes" : extracted_codes, "execution_results" : execution_results, "execution_summaries":execution_summaries}