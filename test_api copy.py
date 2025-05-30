# test_vllm.py
"""
Minimal health-check for a local vLLM server exposed through the
OpenAI-compatible REST API (default route: http://localhost:8000/v1).

Requirements:
  pip install openai==1.*   # official library, works with vLLM’s adapter
"""

import os
import openai
from dotenv import load_dotenv

load_dotenv()

# vLLM does NOT look at the key, but OpenAI’s client expects one.
os.environ["OPENAI_API_KEY"] = "local-testing"         

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",   # note the /v1 suffix
    api_key=os.environ["OPENAI_API_KEY"],
)

try:
    response = client.chat.completions.create(
        model= "Qwen/Qwen2.5-1.5B""RedHatAI/DeepSeek-R1-Distill-Qwen-7B-FP8-dynamic",  # put the exact name you loaded
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=16,   # short reply is enough for a smoke test
    )
    print("Status  : OK")
    print("Response:", response.choices[0].message.content.strip())
except Exception as err:
    # Any openai.*Error (connection, 404, etc.) is caught here.
    print("Health-check FAILED ❌")
    print(type(err).__name__, "-", err)

# # File: test_vllm_n_standalone.py
# import os
# import openai
# import json
# import logging
# import time

# # --- Configure Logging (simple version for testing) ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger("test_vllm_n_standalone")

# # --- Test Parameters ---
# # These should point to your vLLM server that's running the CODER model
# # (as started by your entrypoint.sh on port 8000)
# VLLM_ENDPOINT_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
# VLLM_API_KEY = os.getenv("VLLM_API_KEY", "") # Often empty for local vLLM, use "EMPTY" or "" if so

# # The model name here might be ignored by vLLM if the endpoint is dedicated to one model,
# # but it's good practice to include it as per OpenAI API spec.
# # Use the model name your vLLM server on port 8000 is configured to serve.
# MODEL_ID_FOR_VLLM = os.getenv("CODER_MODEL", "RedHatAI/DeepSeek-R1-Distill-Qwen-14B-FP8-dynamic") # Defaulting to a common small one

# NUMBER_OF_COMPLETIONS_TO_REQUEST = 3
# SYSTEM_MESSAGE_TEST = "You are a Python coding assistant."
# USER_MESSAGE_TEST = "Generate three different one-line Python print statements."
# TEMPERATURE_TEST = 0.7
# MAX_TOKENS_TEST = 30

# # Initialize OpenAI client to talk to the vLLM endpoint
# try:
#     if not VLLM_API_KEY: # Handle common case of empty API key for local vLLM
#         logger.info("VLLM_API_KEY is empty or not set, using 'EMPTY' as default for local vLLM.")
#         vllm_client_api_key = "EMPTY" 
#     else:
#         vllm_client_api_key = VLLM_API_KEY

#     client = openai.OpenAI(
#         base_url=VLLM_ENDPOINT_URL,
#         api_key=vllm_client_api_key, # vLLM often ignores this if not configured for auth
#         max_retries=2 # Allow a couple of retries for network hiccups
#     )
#     logger.info(f"OpenAI client configured for vLLM endpoint: {VLLM_ENDPOINT_URL}")
# except Exception as e:
#     logger.error(f"Error initializing OpenAI client for vLLM: {e}")
#     exit()

# def run_vllm_n_test():
#     logger.info(f"--- Testing vLLM Endpoint: {VLLM_ENDPOINT_URL} ---")
#     logger.info(f"Model hint (may be overridden by endpoint): {MODEL_ID_FOR_VLLM}")
#     logger.info(f"Requesting n = {NUMBER_OF_COMPLETIONS_TO_REQUEST} completions.\n")

#     api_call_params = {
#         "model": MODEL_ID_FOR_VLLM,
#         "messages": [
#             {"role": "system", "content": SYSTEM_MESSAGE_TEST},
#             {"role": "user", "content": USER_MESSAGE_TEST},
#         ],
#         "temperature": TEMPERATURE_TEST,
#         "max_tokens": MAX_TOKENS_TEST,
#         "n": NUMBER_OF_COMPLETIONS_TO_REQUEST,
#         # "top_p": 1, # Optional: vLLM supports top_p
#         # "stop": None, # Optional: vLLM supports stop sequences
#     }
#     logger.debug(f"API call parameters: {api_call_params}")

#     try:
#         start_time = time.time()
#         response = client.chat.completions.create(**api_call_params)
#         end_time = time.time()
#         logger.info(f"API call took {end_time - start_time:.2f} seconds.")

#         logger.info("\n--- Full API Response (raw object) ---")
#         try:
#             # Pretty print the response object if model_dump() is available
#             logger.info(json.dumps(response.model_dump(), indent=2))
#         except AttributeError: # Fallback for older openai versions or different response objects
#             logger.info(str(response))
#         except Exception as e_dump:
#             logger.warning(f"Could not model_dump response, printing raw: {response}")
#             logger.warning(f"(Error during model_dump: {e_dump})")


#         logger.info("\n--- Analysis ---")
#         if response.choices:
#             logger.info(f"Number of choices returned by API: {len(response.choices)}")
#             if len(response.choices) == NUMBER_OF_COMPLETIONS_TO_REQUEST:
#                 logger.info(f"SUCCESS: API returned the requested {NUMBER_OF_COMPLETIONS_TO_REQUEST} choices.")
#             elif len(response.choices) == 1 and NUMBER_OF_COMPLETIONS_TO_REQUEST > 1:
#                 logger.warning(f"INFO: API returned 1 choice, even though n={NUMBER_OF_COMPLETIONS_TO_REQUEST} was requested.")
#                 logger.warning("This suggests the vLLM server/model at this endpoint might not fully support 'n > 1' or is configured to return only one.")
#             else:
#                 logger.warning(f"WARNING: API returned {len(response.choices)} choices, but {NUMBER_OF_COMPLETIONS_TO_REQUEST} were requested.")

#             logger.info("\n--- Generated Completions (from choices) ---")
#             for i, choice in enumerate(response.choices):
#                 logger.info(f"Completion {i+1}:")
#                 if choice.message and choice.message.content:
#                     logger.info(f"  Content: \"{choice.message.content.strip()}\"")
#                 else:
#                     logger.info(f"  Content: None or not available")
#                 logger.info(f"  Finish Reason: {choice.finish_reason}")
#                 logger.info(f"  Index: {choice.index}") # Should correspond to 0, 1, 2... for n=3
#                 logger.info("-" * 20)
#         else:
#             logger.error("ERROR: No choices found in the API response.")

#         if response.usage:
#             logger.info("\n--- Token Usage (if provided by vLLM) ---")
#             logger.info(f"Prompt Tokens: {response.usage.prompt_tokens}")
#             logger.info(f"Completion Tokens: {response.usage.completion_tokens} (often total for all choices)")
#             logger.info(f"Total Tokens: {response.usage.total_tokens}")
#         else:
#             logger.info("\nToken usage information not available in response from this vLLM endpoint.")

#     except openai.APIError as e:
#         logger.error(f"\n--- OpenAI API Error (from vLLM endpoint) ---")
#         logger.error(f"Error Type: {type(e)}")
#         if hasattr(e, 'status_code'): logger.error(f"Status Code: {e.status_code}")
#         if hasattr(e, 'code'): logger.error(f"Error Code (from body): {e.code}")
#         logger.error(f"Message: {e.message}")
#         if e.body: logger.error(f"Error Body: {e.body}")
#         if "param" in str(e).lower() or (e.body and "param" in str(e.body).lower()):
#             logger.error("This error MIGHT indicate that the 'n' parameter (or another parameter) is not supported or invalid for this vLLM model/endpoint configuration.")
#     except Exception as e:
#         logger.error(f"\n--- An Unexpected Error Occurred ---")
#         logger.error(f"Error Type: {type(e)}", exc_info=True)

#     logger.info("\n--- Standalone vLLM 'n' Parameter Test Complete ---")

# if __name__ == "__main__":
#     # Basic check for vLLM server health before running the actual test
#     VLLM_HEALTH_URL = VLLM_ENDPOINT_URL.replace("/v1", "/health") # Construct health URL
    
#     # If VLLM_ENDPOINT_URL doesn't end with /v1, this might need adjustment
#     if not VLLM_ENDPOINT_URL.endswith("/v1"):
#         # A simple heuristic if /v1 is not present
#         base_url_for_health = VLLM_ENDPOINT_URL.split('/openai')[0] if '/openai' in VLLM_ENDPOINT_URL else VLLM_ENDPOINT_URL
#         if base_url_for_health.endswith('/'): base_url_for_health = base_url_for_health[:-1]
#         VLLM_HEALTH_URL = f"{base_url_for_health}/health"
#         logger.info(f"Adjusted health check URL to: {VLLM_HEALTH_URL} based on endpoint: {VLLM_ENDPOINT_URL}")


#     logger.info(f"Checking vLLM server health at: {VLLM_HEALTH_URL}")
#     try:
#         import requests
#         response = requests.get(VLLM_HEALTH_URL, timeout=10) # Increased timeout slightly
#         if response.status_code == 200:
#             logger.info(f"vLLM server at {VLLM_HEALTH_URL} reported healthy (status {response.status_code}).")
#             run_vllm_n_test()
#         else:
#             logger.error(f"vLLM server at {VLLM_HEALTH_URL} responded with status {response.status_code}. Output: {response.text[:200]}")
#             logger.error("Cannot run test if server is not healthy or health check URL is incorrect.")
#     except ImportError:
#         logger.warning("`requests` library not found. Skipping vLLM health check. Make sure the server is running and accessible at the specified VLLM_ENDPOINT_URL.")
#         run_vllm_n_test() # Attempt test anyway
#     except requests.exceptions.RequestException as req_e:
#         logger.error(f"Could not connect to vLLM server for health check at {VLLM_HEALTH_URL}: {req_e}")
#         logger.error("Please ensure your vLLM server (for the CODER model on port 8000 by default) is running and accessible.")
#     except Exception as he:
#         logger.error(f"An unexpected error occurred during vLLM health check: {he}", exc_info=True)
# # import os
# # import openai
# # import json
# # from dotenv import load_dotenv

# # # --- Configuration ---
# # # Load environment variables (especially OPENAI_API_KEY)
# # load_dotenv()

# # # !! IMPORTANT: Set your API key directly if not using .env !!
# # # os.environ["OPENAI_API_KEY"] = "sk-your-actual-api-key"

# # if not os.getenv("OPENAI_API_KEY"):
# #     print("Error: OPENAI_API_KEY environment variable not set.")
# #     print("Please set it or add it to a .env file in the same directory.")
# #     exit()

# # # --- Model to Test ---
# # # Replace with the exact model ID you want to test (e.g., "gpt-3.5-turbo", "gpt-4o", or your "o3-..." model)
# # # For Anthropic models via OpenAI, the model ID might look different or might not be directly available.
# # # If you are using a specific Azure endpoint or other proxy, you might need to configure the client base_url.
# # MODEL_ID_TO_TEST = "o4-mini-2025-04-16" # GOOD: Supports 'n'
# # # MODEL_ID_TO_TEST = "gpt-4o" # GOOD: Supports 'n'
# # # MODEL_ID_TO_TEST = "text-davinci-003" # OLDER, might behave differently, not chat/completions
# # # MODEL_ID_TO_TEST = "claude-3-opus-20240229" # If you access Claude via Anthropic's API or a proxy that maps it to OpenAI like interface
# #                                             # This specific ID is for Anthropic's API.
# #                                             # If you have an "o3-claude-opus" ID for an OpenAI-compatible endpoint, use that.


# # # --- Test Parameters ---
# # NUMBER_OF_COMPLETIONS_TO_REQUEST = 3 # The 'n' parameter value
# # TEST_PROMPT_MESSAGES = [
# #     {"role": "system", "content": "You are a helpful assistant."},
# #     {"role": "user", "content": "Generate a very short, one-sentence creative story idea."},
# # ]
# # TEST_TEMPERATURE = 0.7
# # TEST_MAX_TOKENS = 50

# # # Initialize OpenAI client
# # try:
# #     client = openai.OpenAI()
# # except Exception as e:
# #     print(f"Error initializing OpenAI client: {e}")
# #     exit()

# # print(f"--- Testing Model: {MODEL_ID_TO_TEST} ---")
# # print(f"Requesting n = {NUMBER_OF_COMPLETIONS_TO_REQUEST} completions.\n")

# # try:
# #     response = client.chat.completions.create(
# #         model=MODEL_ID_TO_TEST,
# #         messages=TEST_PROMPT_MESSAGES,
# #         # temperature=TEST_TEMPERATURE,
# #         # max_tokens=TEST_MAX_TOKENS,
# #         n=NUMBER_OF_COMPLETIONS_TO_REQUEST,  # Key parameter to test
# #     )

# #     print("--- Full API Response ---")
# #     try:
# #         # Pretty print the response object
# #         print(json.dumps(response.model_dump(), indent=2))
# #     except Exception as e_dump:
# #         print(f"Could not model_dump response, printing raw: {response}")
# #         print(f"(Error during model_dump: {e_dump})")


# #     print("\n--- Analysis ---")
# #     if response.choices:
# #         print(f"Number of choices returned by API: {len(response.choices)}")
# #         if len(response.choices) == NUMBER_OF_COMPLETIONS_TO_REQUEST:
# #             print(f"SUCCESS: API returned the requested {NUMBER_OF_COMPLETIONS_TO_REQUEST} choices.")
# #         elif len(response.choices) == 1 and NUMBER_OF_COMPLETIONS_TO_REQUEST > 1:
# #             print(f"INFO: API returned 1 choice, even though n={NUMBER_OF_COMPLETIONS_TO_REQUEST} was requested.")
# #             print("This suggests the model or endpoint might not fully support the 'n' parameter for multiple distinct completions, or it's configured to return only one despite 'n'.")
# #         else:
# #             print(f"WARNING: API returned {len(response.choices)} choices, but {NUMBER_OF_COMPLETIONS_TO_REQUEST} were requested.")

# #         print("\n--- Generated Completions ---")
# #         for i, choice in enumerate(response.choices):
# #             print(f"Completion {i+1}:")
# #             if choice.message and choice.message.content:
# #                 print(f"  Content: \"{choice.message.content.strip()}\"")
# #             else:
# #                 print(f"  Content: None or not available")
# #             print(f"  Finish Reason: {choice.finish_reason}")
# #             # print(f"  Logprobs: {choice.logprobs}") # Usually None unless requested
# #             print(f"  Index: {choice.index}")
# #             print("-" * 20)
# #     else:
# #         print("ERROR: No choices found in the API response.")

# #     if response.usage:
# #         print("\n--- Token Usage ---")
# #         print(f"Prompt Tokens: {response.usage.prompt_tokens}")
# #         print(f"Completion Tokens: {response.usage.completion_tokens} (this is often total for all choices if n > 1)")
# #         print(f"Total Tokens: {response.usage.total_tokens}")
# #     else:
# #         print("\nToken usage information not available.")

# # except openai.APIError as e:
# #     print(f"\n--- OpenAI API Error ---")
# #     print(f"Error Type: {type(e)}")
# #     print(f"Status Code: {e.status_code}")
# #     print(f"Error Code: {e.code}")
# #     print(f"Message: {e.message}")
# #     if e.body:
# #         print(f"Error Body: {e.body}")
# #     if "param" in str(e) or (e.body and "param" in e.body):
# #         print("\nThis error might indicate that the 'n' parameter (or another parameter) is not supported or invalid for this model/endpoint.")
# # except Exception as e:
# #     print(f"\n--- An Unexpected Error Occurred ---")
# #     print(f"Error Type: {type(e)}")
# #     print(f"Message: {str(e)}")

# # print("\n--- Test Complete ---")





# # # import os

# # # OUTPUT_MD = "FULL_CODEBASE.md"

# # # # Directories to scan
# # # DIRS = [
# # #     "aide",
# # #     "aide/backend",
# # #     "aide/utils",
# # #     "."
# # # ]

# # # # Helper to get all .py files in a directory (non-recursive for backend/utils, recursive for aide)
# # # def get_py_files(base_dir):
# # #     py_files = []
# # #     for root, dirs, files in os.walk(base_dir):
# # #         # Skip __pycache__
# # #         dirs[:] = [d for d in dirs if d != "__pycache__"]
# # #         dirs[:] = [d for d in dirs if d != ".venv"]
# # #         dirs[:] = [d for d in dirs if d != "data"]
# # #         dirs[:] = [d for d in dirs if d != ".aide-ds"]
# # #         for f in files:
# # #             if f.endswith(".py") and not f.startswith("."):
# # #                 rel_path = os.path.relpath(os.path.join(root, f), ".")
# # #                 py_files.append(rel_path)
# # #         # For backend/utils, don't recurse
# # #         if base_dir in ["aide/backend", "aide/utils"]:
# # #             break
# # #     return py_files

# # # all_py_files = set()
# # # for d in DIRS:
# # #     if os.path.isdir(d):
# # #         all_py_files.update(get_py_files(d))

# # # # Remove duplicates and sort
# # # all_py_files = sorted(all_py_files)
# # # failed_files = []
# # # with open(OUTPUT_MD, "w") as out:
# # #     out.write("# Full aide-agent Codebase\n\n")
# # #     for path in all_py_files:
# # #         out.write(f"## {path}\n\n")
# # #         out.write("```python\n")
# # #         try:
# # #             with open(path, "r") as f:
# # #                 out.write(f.read())
# # #         except Exception as e:
# # #             print(f"Failed to read {path}: {e}")
# # #             failed_files.append(path)
# # #         out.write("\n```")
# # #         out.write("\n\n")

# # # print(f"Wrote {len(all_py_files)} files to {OUTPUT_MD}") 
# # # print(f"Failed to read {len(failed_files)} files")