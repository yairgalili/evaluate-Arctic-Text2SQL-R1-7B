import openai
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

prompt_suggested = """
System:
You are a data science expert. Below, you are provided with a database schema and a natural
language question. Your task is to understand the schema and generate a valid SQL query to
answer the question.
User:
Database Engine:
SQLite
Database Schema: {schema}
This schema describes the database’s structure, including tables, columns, primary keys,
foreign keys, and any relevant relationships or constraints.
Question:
{question}
Instructions:
- Make sure you only output the information that is asked in the question. If the question asks
for a specific column, make sure to only include that column in the SELECT clause, nothing
more.
- The generated query should return all of the information asked in the question without any
missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the
query.
Output Format:
Please provide a detailed chain-of-thought reasoning process and include your thought
process within ‘<think>‘ tags. Your final answer should be enclosed within ‘<answer>‘ tags.
Ensure that your SQL query follows the correct syntax and is formatted as follows:
```sql
– Your SQL query here
```
Example format:
<think> Step-by-step reasoning, including self-reflection and corrections if necessary.
[Limited by 4K tokens] </think>
<answer> Summary of the thought process leading to the final SQL query. [Limited by 1K
tokens]
```sql
Correct SQL query here
```
</answer>
Assistant:
Let me solve this step by step.
<think>

"""

import os
import sqlite3

def get_conn(spider_db_path:str, db_id: str):
    db_file = f"{db_id}.sqlite"
    db_path = os.path.join(spider_db_path, db_id, db_file)
    conn = sqlite3.connect(db_path)
    return conn

def get_db_schema(conn) -> str:
    res = conn.execute("SELECT * FROM sqlite_master").fetchall()
    schema = ""
    for d in res:
        if d[-1] is None:
            continue
            
        schema += f"{d[-1]}\n\n"
    return schema

def inference(tokenizer, model, question: str, schema: str) -> str:
    eos_token_id = tokenizer.eos_token_id
    inputs = tokenizer(prompt_suggested.format(question=question, schema=schema), return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        **inputs,
        num_return_sequences=1,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
        max_new_tokens=1000,
        do_sample=False,
    )
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    def postgres_to_sqlite(query: str) -> str:
        substitutions = [
            (r'ilike', 'LIKE'),
            (r'serial\s*$', 'INTEGER PRIMARY KEY AUTOINCREMENT'),
            (r'start\s+with\s+(\d+)', 'CHECK (id >= \\1)'),
        ]

        for pattern, replacement in substitutions:
            query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)

        return query

    # postgres_query = outputs[0].split("Thus, the final SQL query is:\n```sql")[-1].rstrip("```")
    # query = postgres_to_sqlite(postgres_query)
    if "Thus, the final SQL query is:" in outputs[0]:
        query = outputs[0].split("Thus, the final SQL query is:\n```sql")[-1].split("```")[0]
    else:
        query = outputs[0].split("```sql")[-1].split("```")[0]
    return query


def compare_sql(gold: str, gen: str, conn, is_ordered=False):
    try:
        gold_res = pd.read_sql(gold, conn)
    except Exception as _:
        print("[Ground Fail]", gold)
        return 1

    try:
        gen_res = pd.read_sql(gen, conn)
    except Exception as _:
        print("[Gen Fail]", gen)
        return 0

    accuracy = 0
    if (len(gold_res)) == 0:
        return 1 if len(gen_res) == 0 else 0
    
    gold_len = len(gold_res)
    gen_len = len(gen_res)
    for i in range(min(gold_len, gen_len)):
        gold_record = gold_res.values[i]
        
        if not is_ordered:
            try:
                is_match = gold_record in gen_res.values
            except:
                is_match = False
        else:
            is_match = gold_record == gen_res.values[i]
            if (type(is_match) != bool):
                is_match = is_match.all()
                
        if is_match:
            accuracy += 1

    return accuracy / len(gold_res)


# Generate prompt for GPT
def make_prompt(schema: str, question: str):
    
    prompt = f"""
    You are an expert in generating SQL queries. Based on the following question and database schema, write the correct SQL query.

    Database Schema:
    {schema}

    Question: {question}

    SQL Query:
    """
    return prompt


# Call OpenAI API
def get_sql_query(client, prompt):
    completion = client.chat.completions.create(
    model="gpt-4", # gpt-3.5-turbo
    messages=[
            {"role": "user", "content": prompt}
    ]
    )
    return completion.choices[0].message.content
