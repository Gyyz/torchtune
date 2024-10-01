import os
import json
import re
import sys
import sqlvalidator
from glob import glob

jsonl_path = sys.argv[1]
output_file_path = sys.argv[2]
print(jsonl_path, output_file_path)
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
#            "[q|Q]uery.*:(.+)"
#            ]
select_regex = r"(SELECT.*?FROM.*?)(?:;|(?=SELECT)|$)"
select_regex = r"(SELECT\s.*?FROM\s.*?(?:;|$))"

def remove_unbalanced_parentheses(s):
    stack = []
    indexes_to_remove = set()
    
    # First pass: Identify unbalanced closing parentheses and their positions
    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                stack.pop()
            else:
                indexes_to_remove.add(i)
    
    # Add any unbalanced opening parentheses to the removal set
    indexes_to_remove.update(stack)
    
    # Build the final string without the unbalanced parentheses
    result = ''.join([char for i, char in enumerate(s) if i not in indexes_to_remove])
    
    return result

def process_unit(string_line):
    string_line = string_line.strip()
    responses = string_line.split(";")

    # Strip leading and trailing whitespace from the input string
    # string_line = string_line.strip()
    for response in responses:
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
            response = response.split('\n')[0]

        if ' # ' in response:
            response = response.split(' # ')[0]
        if ' ### ' in response:
            response = response.split(' # ')[0]

        if response.endswith(";"):
            response = response[:-1]

        if response:
            response = remove_unbalanced_parentheses(response)
            print(f"Processed response: {response}")
            return response
        
    return "No Response"
    

    
    # sql_query = sqlvalidator.parse(response)
    # if not sql_query.is_valid():
    #     print(sql_query.errors)

    # print(f"Processed response: {response}")
    return response


def post_process_sql_string(jsonfile, output_file_path=output_file_path):
    instances = json.load(open(jsonfile))
    for instance in instances:
        respone = instance["lama_output"]
        respone = process_unit(respone)
        instance["response_text_processed"] = respone

    with open(output_file_path, "w") as f:
        json.dump(instances, f, indent=4)

post_process_sql_string(jsonl_path, output_file_path)
    