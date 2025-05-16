# from openai import OpenAI
# from dotenv import load_dotenv      
# import os
# load_dotenv()

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# response = client.chat.completions.create(model="gpt-4o-mini",
# messages=[
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "what model are you!, like what version exactly?"},
# ],
# max_tokens=20)
# import json
# def load_benchmarks(competition_name: str):
#     with open('benchmarks.json', 'r') as f:
#         benchmarks = json.load(f)
#     return benchmarks[competition_name]
# # print(response.choices[0].message.content)
# competition_benchmarks = load_benchmarks("nomad2018-predict-transparent-conductors")
# print(competition_benchmarks)
import wandb

run = wandb.init()
print(f"Run ID: {run.id}")
wandb.log({"test": 1})
wandb.summary["test"] = 1
wandb.summary["test2"] = 2
wandb.summary["test3"] = 3
summary = wandb.summary._as_dict()
print(type(summary))
print(summary)
# turn to pandas dataframe
import pandas as pd
df = pd.DataFrame(summary)
print(df)
# save as csv
df.to_csv("summary.csv", index=False)

wandb.finish()