import os
import json
import re
import sys
import sqlvalidator
from glob import glob

jsonl_path = sys.argv[1]
output_file_path = sys.argv[2]
dirname = os.path.dirname(output_file_path)
os.makedirs(dirname, exist_ok=True)
print(f'Usage: python script.py input_jsonl_path output_file_path')
# regexes = [
#             "(SELECT .+?;)",
#             "```(.+)```",
#            "The query would be:(.+)The result would be:",
#            "The query would be:(.+)",
#            "[t|T]he answer.+is:(.+)",
#            "[q|Q]uery.*:(.+)This",
#            "[q|Q]uery.*:(.+)Assuming",
#            "[q|Q]uery.*:(.+)",
#            "sql\n(.+)",
#            "The query is:(.+?)(?=The result)",
#            ]
select_regex = r"(SELECT\s.*?FROM\s.*?(?:;|$))"



def process_unit(string_line):
    string_line = string_line.strip()
    response = string_line.split(";")[0]

    # Strip leading and trailing whitespace from the input string
    # string_line = string_line.strip()
    
    # Apply the regex pattern to find and extract the SELECT query
    matches = re.findall(select_regex, response, flags=re.DOTALL | re.IGNORECASE)
    response = matches[0].strip() if matches else ""
    
    # Remove 'SQL' or 'sql' prefixes if present
    response = re.sub(r"^\s*SQL\s*|\s*sql\s*", "", response)
    
    # Clean up extra spaces and ensure the response does not contain extraneous content
    response = response.strip()
    
    if "<|eot_id|>" in response:
        response = response.replace("<|eot_id|>", "").strip()

    if '\n' in response:
        response = response.replace('\n', ' ')

    if ' # ' in response:
        response = response.split(' # ')[0]
    
    # remove multiple spaces
    response = re.sub(r' +', ' ', response)

    
    if not response:
        response = "No response"
    
    if response.endswith(";"):
        response = response[:-1]

    
    # sql_query = sqlvalidator.parse(response)
    # if not sql_query.is_valid():
    #     print(sql_query.errors)
    # print(f"Processed response: {response}")
    return response


def post_process_sql_string(jsonfile, output_file_path=output_file_path):
    instances = json.load(open(jsonfile))
    for instance in instances:
        respone = instance["response_text"]
        respone = process_unit(respone)
        print(f"Processed response: {respone}")
        instance["response_text_processed"] = respone

    with open(output_file_path, "w") as f:
        json.dump(instances, f, indent=4)

post_process_sql_string(jsonl_path, output_file_path)
    