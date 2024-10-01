import json
from random import randrange

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def schema_content_to_str(schema_content):
    schema_dict = {}
    table_names = list(schema_content.keys())
    for table_name in table_names:
        column_names = list(schema_content[table_name].keys())
        if table_name not in schema_dict:
            schema_dict[table_name] = column_names
        else:
            schema_dict[table_name] += column_names

    return str(schema_dict)

#  dict_keys(['schema', 'text', 'sql', 'tables', 'yc_hardness', 'ts_hardness', 'annotation', 'schema_content', 'id', 'pre_schema_content', 'pre_error_type', 
# 'prompt', 'prompt_token_number', 'gpt4_turn_1', 'gpt4_turn_2', 'gpt4_answer', 'response_token_number'])
def prompt_style_1(data, test_set=False):
    # zero-shot
    new_dataset = []
    for item in data:
        # assert item['gpt4_answer'] == "True"
        schema = item['schema']
        user_question = item['text']
        sql = item['sql']
        processed_schema_content = item['pre_schema_content']
        try:
            original_schema_content = json.loads(item['schema_content'])
        except:
            print('schema_content', item['schema_content'], user_question)
            original_schema_content = {}
            continue
        original_schema_content = schema_content_to_str(original_schema_content)
        # if processed_schema_content == {}:
        #     continue
        id = item['id']
        prompt = f'''Based on \n# the table&column(database schema) information {processed_schema_content} and \n# the user question: {user_question},\nGive me the right SQL query to retrieve the information. Only SQL query, no explaination. # SQL query:'''
        prompt_full = f'''Based on \n# the table&column(database schema) information {original_schema_content} and \n# the user question: {user_question},\nGive me the right SQL query to retrieve the information. Only SQL query, no explaination. # SQL query:'''
        new_item = {

            'id': id,
            'instruction': "You are an SQL expert. and proficient with the text-to-sql task.",
            # 'label': item['gpt4_answer'],
            'input_shorten': prompt,
            'input': prompt_full,
            'user_question': user_question,
            'output': sql,
            'schema': schema,
            'processed_schema_content': processed_schema_content,
        }
        new_dataset.append(new_item)
    print(f"Total {len(new_dataset)} instances for zero-shot data for file {test_set}")

    if test_set:
        with open(test_set.replace(".json", ".zeroshot.json"), 'w') as f:
            json.dump(new_dataset, f, indent=4, ensure_ascii=False)
    else:
        with open('prompt_zeroshot.json', 'w') as f:
            json.dump(new_dataset[:-1000], f, indent=4, ensure_ascii=False)
        with open('test_split_zeroshot.json', 'w') as f:
            json.dump(new_dataset[-1000:], f, indent=4, ensure_ascii=False)
    return new_dataset


def prompt_style_2(data, test_set=False):
    # one-shot
    data_size = len(data)
    new_dataset = []
    for item in data:
        # assert item['gpt4_answer'] == "True"
        schema = item['schema']
        user_question = item['text']
        sql = item['sql']
        processed_schema_content = item['pre_schema_content']
        try:
            original_schema_content = json.loads(item['schema_content'])
        except:
            print('schema_content', item['schema_content'], user_question)
            original_schema_content = {}
            continue
        # original_schema_content = json.loads(item['schema_content'])
        # original_schema_content = schema_content_to_str(original_schema_content)
        # if processed_schema_content == {}:
        #     continue
        id = item['id']

        sample = data[randrange(data_size)]
        sample_question = sample['text']
        sample_sql = sample['sql']
        sample_processed_schema_content = sample['pre_schema_content']
        sample_original_schema_content = json.loads(sample['schema_content'])
        sample_original_schema_content = schema_content_to_str(sample_original_schema_content)

        example_question = f'''Here is an example, based on \n#the table&column(database schema) information is: {sample_processed_schema_content}. \n# the user question is: {sample_question}. \n#SQL query: {sample_sql}\n'''
        prompt = f'''So for my task & question:\nBased on \n# the table&column(database schema) information {processed_schema_content} and \n# the user question: {user_question},\nGive me the right SQL query for the question. Only SQL query, no explaination. \n# SQL query:'''

        example_question_full = f'''Here is an example, based on \n#the table&column(database schema) information is: {sample_original_schema_content}. \n# the user question is: {sample_question}. \n#SQL query: {sample_sql}\n'''
        prompt_full = f'''So for my task & question:\nBased on \n# the table&column(database schema) information {original_schema_content} and \n# the user question: {user_question},\nGive me the right SQL query for the question. Only SQL query, no explaination. \n# SQL query:'''

        new_item = {

            'id': id,
            'instruction': "You are an SQL expert. and proficient with the text-to-sql task.",
            'input_shorten': example_question + prompt,
            'input': example_question_full + prompt_full,
            'user_question': user_question,
            'output': sql,
            'schema': schema,
            'processed_schema_content': processed_schema_content,
        }
        new_dataset.append(new_item)
    print(f"Total {len(new_dataset)} instances for one-shot data for file {test_set}")
    if test_set:
        with open(test_set.replace(".json", ".oneshot.json"), 'w') as f:
            json.dump(new_dataset, f, indent=4, ensure_ascii=False)
    
    else:
        with open('prompt_oneshot.json', 'w') as f:
            json.dump(new_dataset[:-1000], f, indent=4, ensure_ascii=False)
        with open('test_split_oneshot.json', 'w') as f:
            json.dump(new_dataset[-1000:], f, indent=4, ensure_ascii=False)
    return new_dataset


def prompt_style_3(data):
    # few-shot with in domain data
    pass



if __name__ == '__main__':
    # filename = './first_round_testset.json'
    # # data = load_data('./gpt4_true_cases.json')
    # data = load_data(filename)
    # # print(data[0])
    # data = [itm for itm in data if itm['label']['correct'] == 'Yes']
    # prompt_style_1(data, test_set=filename)
    # prompt_style_2(data, test_set=filename)


    filename = './second_round_total.json'
    # data = load_data('./gpt4_true_cases.json')
    data = load_data(filename)
    data = [itm for itm in data if itm['label']['correct'] == 'Yes']
    prompt_style_1(data, test_set=filename)
    prompt_style_2(data, test_set=filename)


    # filename = './my_finetune_dataset.json'

    # data = load_data(filename)
    # prompt_style_1(data)
    # prompt_style_2(data)


