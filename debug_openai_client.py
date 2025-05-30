import openai
import json

def test_openai_client():
    client = openai.OpenAI(
        base_url='http://localhost:8000/v1',
        api_key='',
        max_retries=0,
    )

    try:
        response = client.chat.completions.create(
            model='RedHatAI/DeepSeek-R1-Distill-Qwen-7B-FP8-dynamic',
            messages=[{'role': 'user', 'content': 'test'}],
            temperature=0.8,
            n=1,
            max_tokens=10
        )
        print('SUCCESS:', response.choices[0].message.content)
    except Exception as e:
        print('ERROR Type:', type(e).__name__)
        print('ERROR Message:', str(e))
        print('ERROR Repr:', repr(e))
        if hasattr(e, 'body'):
            print('ERROR Body:', e.body)
        if hasattr(e, 'response'):
            print('ERROR Response:', e.response)
        
        # Let's also try with requests directly
        import requests
        try:
            direct_response = requests.post(
                'http://localhost:8000/v1/chat/completions',
                headers={'Content-Type': 'application/json'},
                json={
                    'model': 'RedHatAI/DeepSeek-R1-Distill-Qwen-7B-FP8-dynamic',
                    'messages': [{'role': 'user', 'content': 'test'}],
                    'temperature': 0.8,
                    'n': 1,
                    'max_tokens': 10
                }
            )
            print('DIRECT REQUEST SUCCESS:', direct_response.status_code)
            print('DIRECT REQUEST CONTENT:', direct_response.text[:500])
        except Exception as req_e:
            print('DIRECT REQUEST ERROR:', str(req_e))

if __name__ == '__main__':
    test_openai_client() 