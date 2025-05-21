import asyncio
import aiohttp
import time
import random
import json

# --- Configuration ---
SERVER_URL_1 = "http://localhost:8000/v1/chat/completions"
MODEL_NAME_1 = (
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Match the model loaded on port 8000
)

SERVER_URL_2 = "http://localhost:8001/v1/chat/completions"
MODEL_NAME_2 = (
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Match the model loaded on port 8001
)

# If you were using a single vLLM server with --served-model-name, you'd have one URL
# and change MODEL_NAME_1 and MODEL_NAME_2 to your aliases like "planner-model", "coder-model"

NUM_REQUESTS_PER_SERVER = 5000  # Number of concurrent requests to send to EACH server
MAX_NEW_TOKENS = 20  # Keep generated responses short
TEMPERATURE = 0.1  # For deterministic-like short responses

# Sample prompts - keep them short
PROMPTS = [
    "What is the capital of France?",
    "Briefly explain black holes.",
    "Who painted the Mona Lisa?",
    "What is 2 + 2?",
    "Define artificial intelligence in one sentence.",
    "What is the main purpose of a CPU?",
    "Translate 'hello' to Spanish.",
    "Name a famous scientist.",
    "What is the largest planet in our solar system?",
    "Summarize the theory of relativity quickly.",
]


async def send_request(session, server_url, model_name, prompt_text, request_num):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
    }
    start_time = time.time()
    server_identifier = "Server1(8000)" if "8000" in server_url else "Server2(8001)"
    try:
        # print(f"[{server_identifier}] Sending request {request_num}: {prompt_text[:30]}...")
        async with session.post(
            server_url, json=payload, timeout=30
        ) as response:  # 30-second timeout per request
            response_json = await response.json()
            end_time = time.time()
            duration = end_time - start_time
            if response.status == 200 and response_json.get("choices"):
                content = (
                    response_json["choices"][0].get("message", {}).get("content", "")
                )
                print(
                    f"[{server_identifier}] Req {request_num:02d} OK ({duration:.2f}s): Prompt='{prompt_text[:20]}...' -> Resp='{content[:30].strip()}...'"
                )
                return {"status": "success", "duration": duration, "response": content}
            else:
                error_detail = response_json.get("detail") or response_json.get(
                    "error", {}
                ).get("message", "Unknown error")
                print(
                    f"[{server_identifier}] Req {request_num:02d} ERR ({duration:.2f}s): Status {response.status}, Detail: {error_detail}"
                )
                return {
                    "status": "error",
                    "duration": duration,
                    "error": f"Status {response.status}, Detail: {error_detail}",
                }
    except asyncio.TimeoutError:
        end_time = time.time()
        duration = end_time - start_time
        print(f"[{server_identifier}] Req {request_num:02d} TIMEOUT ({duration:.2f}s)")
        return {"status": "timeout", "duration": duration}
    except aiohttp.ClientConnectorError as e:
        end_time = time.time()
        duration = end_time - start_time
        print(
            f"[{server_identifier}] Req {request_num:02d} CONNECTION_ERROR ({duration:.2f}s): {e}"
        )
        return {"status": "connection_error", "duration": duration, "error": str(e)}
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(
            f"[{server_identifier}] Req {request_num:02d} EXCEPTION ({duration:.2f}s): {type(e).__name__} - {e}"
        )
        return {"status": "exception", "duration": duration, "error": str(e)}


async def main():
    async with aiohttp.ClientSession() as session:
        tasks_server1 = []
        tasks_server2 = []

        print(
            f"--- Starting {NUM_REQUESTS_PER_SERVER} concurrent requests to Server 1 (Port 8000) ---"
        )
        for i in range(NUM_REQUESTS_PER_SERVER):
            prompt = random.choice(PROMPTS)
            tasks_server1.append(
                send_request(session, SERVER_URL_1, MODEL_NAME_1, prompt, i + 1)
            )

        print(
            f"--- Starting {NUM_REQUESTS_PER_SERVER} concurrent requests to Server 2 (Port 8001) ---"
        )
        for i in range(NUM_REQUESTS_PER_SERVER):
            prompt = random.choice(PROMPTS)
            tasks_server2.append(
                send_request(session, SERVER_URL_2, MODEL_NAME_2, prompt, i + 1)
            )

        all_tasks = tasks_server1 + tasks_server2

        start_overall_time = time.time()
        results = await asyncio.gather(
            *all_tasks, return_exceptions=False
        )  # Set to True to see exceptions if gather itself fails
        end_overall_time = time.time()

        print(f"\n--- Test Summary ---")
        print(f"Total requests sent: {len(all_tasks)}")
        print(f"Total time taken: {end_overall_time - start_overall_time:.2f} seconds")

        success_count = sum(1 for r in results if r and r.get("status") == "success")
        error_count = sum(1 for r in results if r and r.get("status") == "error")
        timeout_count = sum(1 for r in results if r and r.get("status") == "timeout")
        conn_error_count = sum(
            1 for r in results if r and r.get("status") == "connection_error"
        )
        other_exception_count = sum(
            1 for r in results if r and r.get("status") == "exception"
        )

        print(f"Successful requests: {success_count}")
        if error_count > 0:
            print(f"Errored requests (from server): {error_count}")
        if timeout_count > 0:
            print(f"Timed out requests: {timeout_count}")
        if conn_error_count > 0:
            print(f"Connection errors: {conn_error_count}")
        if other_exception_count > 0:
            print(f"Other exceptions during request: {other_exception_count}")

        if success_count > 0:
            successful_durations = [
                r["duration"] for r in results if r and r.get("status") == "success"
            ]
            print(
                f"Average duration for successful requests: {sum(successful_durations) / success_count:.2f} seconds"
            )
            print(
                f"Min duration for successful requests: {min(successful_durations):.2f} seconds"
            )
            print(
                f"Max duration for successful requests: {max(successful_durations):.2f} seconds"
            )

        # Further breakdown per server
        results_s1 = results[:NUM_REQUESTS_PER_SERVER]
        results_s2 = results[NUM_REQUESTS_PER_SERVER:]

        s1_success = sum(1 for r in results_s1 if r and r.get("status") == "success")
        s2_success = sum(1 for r in results_s2 if r and r.get("status") == "success")
        print(
            f"\nServer 1 (Port 8000) Successful: {s1_success}/{NUM_REQUESTS_PER_SERVER}"
        )
        print(
            f"Server 2 (Port 8001) Successful: {s2_success}/{NUM_REQUESTS_PER_SERVER}"
        )


if __name__ == "__main__":
    # For Python 3.7+
    asyncio.run(main())
