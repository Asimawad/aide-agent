from typing import Any

fmt = (
"\n\n---\n"
"## Bugs Summary/Analysis: (plain text, no fences):\n"
"<your step‑by‑step reasoning abd summary of the bugs in the previous solution here>\n\n"
"## Plan: (plain text, no fences):\n"
"<your step‑by‑step reasoning and plan steps for fixing the bugs here>\n\n"
    )
plan_prompt: Any = {
            "Introduction": "v",
            "Task description": "self.task_desc,",
            "Previous (buggy) implementation": "wrap_code(parent_node.code)",
            "Execution output": "wrap_code(parent_node.term_out",
            "Instructions": {}
        }
plan_prompt["Instructions"] |= {

    "Your response for the summary should be a detailed and high quality bullet points of the bugs in the previous solution, summarizing all the information and problems(5-7 sentences), "
    "Your response for the plan should be a detailed and high quality bullet points of the steps of your proposed solution in natural language (7-10 sentences), "
    "There should be no additional headings or Code in your response. Just natural language text (summary) under ## Bugs Summary/Analysis: and natural language text (plan) under ## Plan: "
    "explicitly,structure your answer exactly like this: " + fmt
}
