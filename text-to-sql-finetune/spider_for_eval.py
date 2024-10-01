import json
import os
import sys

input_file = sys.argv[1]

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data

def split_data(data):
    golden = []
    predicted = []
    for item in data:
        db_id = item['db_id']
        golden_sql = item['output']
        predicted_sql = item['response_text_processed']
        golden.append(f"{golden_sql}\t{db_id}")
        predicted.append(f"{predicted_sql}\t{db_id}")
    
    with open("golden.txt", "w") as f:
        f.write("\n".join(golden))
    
    with open("predicted.txt", "w") as f:
        f.write("\n".join(predicted))
if __name__ == '__main__':
    data = read_json(input_file)
    split_data(data)