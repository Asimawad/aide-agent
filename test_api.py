"""

aide data_dir="data/spooky-author-identification" \
    goal="Predict the author of a sentence as one of Poe, Lovecraft, or Shelley" \
    eval="Use multi-class logarithmic loss between predicted author probabilities and the true label." \
    agent.code.model=o3-mini \
    agent.ITS_Strategy="none" \
    agent.steps=3 \
    inference_engine="vllm" \
    agent.code.planner_model="Qwen/Qwen2-0.5B-Instruct" \
    competition_name=spooky-author-identification



"""
import os
import openai

def test_vllm_connection():
    base_url = os.getenv("VLLM_BASE_URL2", "http://localhost:8001/v1")
    api_key = os.getenv("VLLM_API_KEY", "EMPTY")  # Use if your server requires it

    client = openai.OpenAI(base_url=base_url, api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2-0.5B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello in one sentence."}
            ],
            max_tokens=20,
            temperature=0.5,
        )
        print("Response from vLLM server:")
        print(response.choices[0].message.content)
    except Exception as e:
        print("Error querying vLLM server:", e)

if __name__ == "__main__":
    test_vllm_connection()
