
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer
import json
import openai
from tqdm import tqdm
import os
torch.cuda.empty_cache()

from utils import compare_sql, get_conn, get_db_schema, get_sql_query, inference, make_prompt

# Set your API key
openai.api_key = ""
from openai import OpenAI
api_key = ''

client = OpenAI(api_key=api_key)


# %%
# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-0528", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Snowflake/Arctic-Text2SQL-R1-7B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Snowflake/Arctic-Text2SQL-R1-7B", trust_remote_code=True)
model = model.eval()
# tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
# tokenizer = PreTrainedTokenizer.from_pretrained("your_tokenizer_directory", use_fast=False)


# from transformers import AutoModel
# model = AutoModel.from_pretrained("Snowflake/Arctic-Text2SQL-R1-7B", trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# %%




# Load dataset
def load_dataset(data_path, table_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    with open(table_path, 'r') as f:
        tables = json.load(f)
    return data, tables



# Main function
def evaluate_openai(data, tables, output_path):
    results = []
    table_dict = {t['db_id']: t for t in tables}

    for entry in tqdm(data):
        db_id = entry['db_id']
        prompt = make_prompt(entry, table_dict[db_id])
        pred_sql = get_sql_query(prompt)

        results.append({
            "question": entry['question'],
            "gold": entry['query'],
            "pred": pred_sql,
            "db_id": db_id
        })

    # Save predictions
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Predictions saved to {output_path}")

# Run evaluation
if __name__ == "__main__":
    # data_path = "spider_data/dev.json"
    # table_path = "spider_data/tables.json"
    # output_path = "predictions.json"

    # data, tables = load_dataset(data_path, table_path)


    # evaluate_model([data[0]], tables, output_path)
    
    spider_db_path = "/home/paperspace/yair/spider_data/database"
    spider_train_path = "/home/paperspace/yair/spider_data/train_spider.json"

    with open(spider_train_path, "r") as f:
        data = json.load(f)
    
# data[0]
# cd spider/
# python evaluation.py --gold dev.json --pred predictions.json --db_root_path spider/database/

# %%
import sqlparse
NUM_EVAL = 100
total_accuracy = 0
total_accuracy_openai = 0
for idx, h in enumerate(tqdm(data)):
    if idx >= NUM_EVAL:
        break
        
    db_id = h['db_id']
    conn = get_conn(spider_db_path, db_id)
    schema = get_db_schema(conn)
    gen = inference(tokenizer, model, h['question'], schema)
    gold = h['query']
    total_accuracy += compare_sql(gold, gen, conn)

    # openai
    prompt = make_prompt(schema=schema, question=h['question'])
    sql_openai = get_sql_query(client, prompt)
    total_accuracy_openai += compare_sql(gold, sql_openai, conn)

    print(f"Example {idx}")
    print("total_accuracy", total_accuracy)
    print("total_accuracy_openai", total_accuracy_openai)
    # format_gold = sqlparse.format(gold, reident=True)
    # format_gen = sqlparse.format(gen, reident=True)
    
total_accuracy /= NUM_EVAL
print(f"Execution accuracy: {total_accuracy * 100}%")
    
        