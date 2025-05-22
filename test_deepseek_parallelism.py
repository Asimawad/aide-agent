import asyncio
import time
import os
import logging
from typing import Dict, Any, List, Tuple, Optional
from dotenv import load_dotenv
import openai

load_dotenv()  # Load environment variables from .env file
# --- Load Environment Variables ---
# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("deepseek_concurrency_test")

# --- Configuration for DeepSeek API ---
HOSTED_DEEPSEEK_API_KEY = os.getenv("HOSTED_DEEPSEEK_API_KEY")
if not HOSTED_DEEPSEEK_API_KEY:
    logger.error(
        "HOSTED_DEEPSEEK_API_KEY environment variable not set. "
        "Please set it to your DeepSeek API key."
    )
    exit(1)

# Global async client instance to be initialized once
_async_deepseek_client: Optional[openai.AsyncOpenAI] = None


# --- Asynchronous Client Setup ---
def get_async_deepseek_client() -> openai.AsyncOpenAI:
    """
    Initializes and returns a singleton async OpenAI client for DeepSeek.
    """
    global _async_deepseek_client
    if _async_deepseek_client is None:
        logger.info("Initializing async Hosted DeepSeek client...")
        try:
            _async_deepseek_client = openai.AsyncOpenAI(
                base_url="https://api.deepseek.com",  # Official DeepSeek API base URL
                api_key=HOSTED_DEEPSEEK_API_KEY,
                timeout=60.0,  # Set a generous timeout for API calls
            )
        except Exception as e:
            logger.critical(
                f"Failed to setup Hosted DeepSeek client: {e}", exc_info=True
            )
            raise
    return _async_deepseek_client


async def query_deepseek_async(
    prompt: str,
    model: str,
    temperature: float = 0.1,  # Low temperature for more deterministic results
    max_new_tokens: int = 300,  # Limit token generation for faster responses
    request_idx: int = 0,
) -> Tuple[str, float, int, int, Dict[str, Any]]:
    """
    Sends a single asynchronous query to the hosted DeepSeek model.
    Returns: (output_text, request_time, input_tokens, output_tokens, info_dict)
    """
    client = get_async_deepseek_client()

    messages = [{"role": "user", "content": prompt}]

    api_params = {
        "temperature": temperature,
        "max_tokens": max_new_tokens,
        "model": model,
        # Add other parameters here if desired, e.g., "top_p": 0.9, "stop": ["```"]
    }

    logger.debug(
        f"Request {request_idx}: Calling DeepSeek API with params: {api_params}"
    )

    t0 = time.time()
    completion = None
    try:
        completion = await client.chat.completions.create(
            messages=messages,
            **api_params,
        )
    except openai.APIConnectionError as e:
        logger.error(f"Request {request_idx}: API Connection Error: {e}")
        return (
            f"ERROR: API Connection Failed: {e}",
            time.time() - t0,
            0,
            0,
            {"error": str(e), "status": "connection_error"},
        )
    except openai.RateLimitError as e:
        logger.warning(f"Request {request_idx}: Rate Limit Exceeded: {e}")
        return (
            f"ERROR: Rate Limit Exceeded: {e}",
            time.time() - t0,
            0,
            0,
            {"error": str(e), "status": "rate_limit_error"},
        )
    except openai.APIStatusError as e:
        logger.error(
            f"Request {request_idx}: API Status Error {e.status_code}: {e.response} - {e.message}"
        )
        return (
            f"ERROR: API Status Error: {e.status_code} - {e.message}",
            time.time() - t0,
            0,
            0,
            {"error": str(e), "status_code": e.status_code, "response": e.response},
        )
    except openai.APITimeoutError as e:
        logger.error(f"Request {request_idx}: API Timeout Error: {e}")
        return (
            f"ERROR: API Timeout Error: {e}",
            time.time() - t0,
            0,
            0,
            {"error": str(e), "status": "timeout_error"},
        )
    except Exception as e:
        logger.error(
            f"Request {request_idx}: Unexpected DeepSeek query failed: {e}",
            exc_info=True,
        )
        return (
            f"ERROR: API call failed: {e}",
            time.time() - t0,
            0,
            0,
            {"error": str(e), "status": "unexpected_error"},
        )

    req_time = time.time() - t0

    if not completion or not completion.choices:
        err_msg = "DeepSeek API call returned empty or invalid completion object."
        logger.error(f"Request {request_idx}: {err_msg}")
        return (
            f"ERROR: {err_msg}",
            req_time,
            0,
            0,
            {"error": err_msg, "status": "no_completion"},
        )

    choice = completion.choices[0]
    output = choice.message.content or ""

    input_tokens = completion.usage.prompt_tokens if completion.usage else 0
    output_tokens = completion.usage.completion_tokens if completion.usage else 0

    info = {
        "model_used_by_provider": completion.model,
        "finish_reason": choice.finish_reason,
        "id": completion.id,
        "created": completion.created,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "request_time_sec": req_time,
        "status": "success",
    }
    logger.debug(
        f"Request {request_idx}: Response tokens: In={input_tokens}, Out={output_tokens}, Time={req_time:.2f}s"
    )
    return output, req_time, input_tokens, output_tokens, info


async def main():
    """
    Main function to orchestrate concurrent DeepSeek API calls.
    """
    # --- Test Parameters ---
    num_concurrent_requests = 10  # Number of concurrent requests to send
    # Use 'deepseek-chat' or 'deepseek-coder' or other models as per DeepSeek's offerings
    model_to_test = os.getenv("DEEPSEEK_MODEL_ID", "deepseek-chat")

    logger.info(
        f"Starting DeepSeek API concurrency test: "
        f"{num_concurrent_requests} requests for model '{model_to_test}'"
    )

    sample_prompts = [
        "Write a Python function to reverse a string.",
        "Explain the concept of recursion in simple terms.",
        "Generate a simple HTML paragraph with bold text.",
        "What is the time complexity of bubble sort?",
        "Provide a JavaScript example of an anonymous function.",
        "Write a SQL query to get the count of users in a 'users' table.",
        "Describe the purpose of a 'Dockerfile'.",
        "How do you declare a constant in C++?",
        "Give a concise definition of 'polymorphism' in OOP.",
        "Create a CSS rule to change the background color of a button on hover.",
        "Write a Python list comprehension to square numbers from 1 to 5.",
        "What are HTTP status codes? List a few common ones.",
        "Explain the 'git rebase' command.",
        "How to handle errors in Go language?",
        "Define 'RESTful API' in one sentence.",
    ]

    # Create a list of prompts, recycling if num_concurrent_requests > len(sample_prompts)
    prompts_for_tasks = [
        sample_prompts[i % len(sample_prompts)] for i in range(num_concurrent_requests)
    ]

    start_total_time = time.time()

    # Create a list of coroutine tasks
    tasks = []
    for i in range(3):
        for i, prompt in enumerate(prompts_for_tasks):
            tasks.append(
                query_deepseek_async(
                    prompt=prompt, model=model_to_test, request_idx=i + 1
                )
            )

    # Run all tasks concurrently and wait for them to complete
    # return_exceptions=True ensures that even if some tasks fail, others complete
    results = await asyncio.gather(*tasks, return_exceptions=True)

    end_total_time = time.time()
    total_duration = end_total_time - start_total_time

    # --- Process Results ---
    successful_requests = 0
    total_input_tokens = 0
    total_output_tokens = 0
    errors = []
    individual_request_times = []

    logger.info("\n--- Test Results Summary ---")
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Request {i+1} failed with an unhandled exception: {result}")
            errors.append(f"Request {i+1} failed: {type(result).__name__} - {result}")
        else:
            output, req_time, input_tokens, output_tokens, info = result
            if "ERROR" in output:
                logger.warning(
                    f"Request {i+1} completed with API error: {output.strip()} (Time: {req_time:.2f}s)"
                )
                errors.append(f"Request {i+1} API Error: {output.strip()}")
            else:
                successful_requests += 1
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                individual_request_times.append(req_time)
                logger.info(
                    f"Request {i+1} (Success): Time={req_time:.2f}s, "
                    f"In={input_tokens}, Out={output_tokens}, "
                    f"Finish Reason: {info.get('finish_reason', 'N/A')}"
                )
                # Optional: print a snippet of the output for debugging
                # logger.debug(f"  Output snippet: {output[:150].replace('\n', ' ')}...")

    logger.info(f"\n--- Overall Statistics ---")
    logger.info(f"Total Requests Sent: {num_concurrent_requests}")
    logger.info(f"Successful Requests: {successful_requests}")
    logger.info(f"Failed Requests: {len(errors)}")
    logger.info(
        f"Total Wall Clock Time (concurrent execution): {total_duration:.2f} seconds"
    )

    if successful_requests > 0:
        logger.info(
            f"Average Request Time (successful individual calls): {sum(individual_request_times) / successful_requests:.2f} seconds"
        )
        logger.info(
            f"Min Request Time (successful individual calls): {min(individual_request_times):.2f} seconds"
        )
        logger.info(
            f"Max Request Time (successful individual calls): {max(individual_request_times):.2f} seconds"
        )
        logger.info(f"Total Input Tokens (successful): {total_input_tokens}")
        logger.info(f"Total Output Tokens (successful): {total_output_tokens}")
        if total_output_tokens > 0:
            logger.info(
                f"Tokens per second (overall successful): {total_output_tokens / total_duration:.2f} tokens/sec"
            )
    else:
        logger.info("No successful requests to calculate average times or token rates.")

    if errors:
        logger.warning("\n--- Errors Details ---")
        for error in errors:
            logger.warning(f"- {error}")

    # Close the client session to release resources
    if _async_deepseek_client:
        await _async_deepseek_client.close()
        logger.info("DeepSeek client session closed.")


if __name__ == "__main__":
    asyncio.run(main())
