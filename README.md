# evaluate-Arctic-Text2SQL-R1-7B


I explore the model Snowflake/Arctic-Text2SQL-R1-7B

1. They do not provide the requirements of their model.
2. After a lot of trials, I got the version transformers==4.45.0, which is suitable for tokenizer and model.

If the reasoning model does not converge to final answer, I used the last step answer as final answer. 

## Comparison for example

| generated | True | Question | Gpt-4 |
| --- | --- | --- | --- |
| '\\nSELECT COUNT(\*) \\nFROM management AS m\\nJOIN head AS h ON m.head_ID = h.head_ID\\nWHERE h.age > 56;\\n'  <br>**3** | 'SELECT count(\*) FROM head WHERE age > 56'<br><br>**5** | 'How many heads of the departments are older than 56 ?' | SELECT COUNT(\*) \\nFROM "head" \\nWHERE "age" > 56;<br><br>**5** |

## Example for 100 cases

Example 99

total_accuracy 91.0

total_accuracy_openai 90.9

1%|█▋ | 100/7000 \[54:43<62:55:33, 32.83s/it\]

Execution accuracy: 91.0%
