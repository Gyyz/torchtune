import sqlparse
import json
from sqlparse.tokens import Keyword
from collections import Counter
from math import comb

def count_tables_and_columns(sql_query, table_names, column_names):
    # Normalize table and column names to lowercase for easier comparison
    table_names = set([name.lower() for name in table_names])
    column_names = set([name.lower() for name in column_names])

    # Tokenize and parse the SQL query
    parsed = sqlparse.parse(sql_query)
    table_count = {}
    column_count = {}

    for token in parsed[0].flatten():
        # Normalize the token value
        value = token.value.lower()
        if value in table_names:
            if value in table_count:
                table_count[value] += 1
            else:
                table_count[value] = 1
        elif value in column_names:
            if value in column_count:
                column_count[value] += 1
            else:
                column_count[value] = 1

    return table_count, column_count

def normalize_sql_query(sql_query):
    # Normalize the SQL query by removing comments and whitespace
    return sql_query

def normalize_table_name(table_name):
    return table_name

def normalize_column_name(column_name):
    return column_name

def count_sql_keywords(sql_query):
    # Tokenize and parse the SQL query
    parsed = sqlparse.parse(sql_query)
    keyword_count = {}

    for token in parsed[0].flatten():
        if token.ttype is Keyword:
            # Normalize the keyword to handle case sensitivity
            keyword = token.value.upper()
            if keyword in keyword_count:
                keyword_count[keyword] += 1
            else:
                keyword_count[keyword] = 1

    return keyword_count

# Example usage
def count_sql_query_complexity(file_path):
    instances = json.load(open(file_path))
    print(len(instances))

    total_table_count = 0
    total_column_count = 0

    total_involved_tables = 0
    total_involved_columns = 0

    total_keyword_count = 0

    for inst in instances:
        # dict_keys(['schema', 'text', 'sql', 'tables', 'yc_hardness', 
        # 'ts_hardness', 'annotation', 'schema_content', 'id', 'pre_schema_content', 
        # 'pre_error_type', 'prompt', 'prompt_token_number', 'gpt4_turn_1', 
        # 'gpt4_turn_2', 'gpt4_answer', 'response_token_number'])
        sql_query = inst['sql']
        schema_content = json.loads(inst['schema_content']) if inst['schema_content'] else {}
        # print(schema_content)
        # print('######tables', schema_content.keys())
        # print('######columns', schema_content.values())  
        # exit(0)
        table_names = set([normalize_table_name(name) for name in schema_content.keys()])
        column_names = set([normalize_column_name(name) for name in [col for cols in schema_content.values() for col in cols.keys()]])
        table_count, column_count = count_tables_and_columns(sql_query, table_names, column_names)
        # print(table_count, column_count)

        total_table_count += sum(table_count.values())
        total_column_count += sum(column_count.values())

        total_keyword_count += sum(count_sql_keywords(sql_query).values())

        total_involved_tables += len([' ' for itm in table_count.values() if itm > 0])
        total_involved_columns += len([' ' for itm in column_count.values() if itm > 0])

    print(total_table_count, total_column_count)
    print(total_table_count/len(instances), total_column_count/len(instances))
    print(total_involved_tables, total_involved_columns)
    print(total_involved_tables/len(instances), total_involved_columns/len(instances))
    print(total_keyword_count/len(instances))


def count_databse_schema_complexity(file_paths):
    instances = []
    for fpath in file_paths:
        instances += json.load(open(fpath))
    
    db_schema = {}
    schemas = []
    for inst in instances:
        db_id = inst['schema']
        
        schema_content = json.loads(inst['schema_content']) if inst['schema_content'] else ''
        if not schema_content:
            if db_id not in schemas:
                print(f'empty in {db_id}', inst['schema_content'])
                schemas.append(db_id)
            continue
        if db_id not in db_schema:
            db_schema[db_id] = {}
            table_names = [normalize_table_name(name) for name in schema_content.keys()]
            column_names = [list(inst.keys()) for inst in schema_content.values()]
            db_schema[db_id]['table_names'] = table_names
            db_schema[db_id]['column_names'] = column_names
        else:
            continue
    # print(len(db_schema))
    # print(len(set(schemas)))
    total_foreign_keys_num = 0
    total_primary_key_num = 0
    total_table_num = 0
    total_column_num = 0
    
    for db_id in db_schema:
        table_names = db_schema[db_id]['table_names']
        total_table_num += len(table_names)
        column_names = db_schema[db_id]['column_names']
        total_column_num += sum([len(cols) for cols in column_names])
        primary_key_num, foreign_keys_num = found_primary_key_foreign_key(table_names, column_names)
        total_primary_key_num += primary_key_num
        total_foreign_keys_num += foreign_keys_num
    print(total_table_num, total_column_num)
    print(total_table_num/len(db_schema), total_column_num/len(db_schema))
    print(total_primary_key_num, total_foreign_keys_num)
    print(total_primary_key_num/len(db_schema), total_foreign_keys_num/len(db_schema))

def found_primary_key_foreign_key(table_names, column_names):
    # table_names is a list of string
    # column_names is a list of list of string
    # assume the column name with id in it is the primary key
    # assume the column name in different table is the foreign key
    column_names_flatted = [col for cols in column_names for col in cols]
    column_count = Counter(column_names_flatted)
    foreign_keys = []
    primary_key = []

    foreign_keys_num = 0
    for col, cnt in column_count.items():
        if cnt > 1:
            foreign_keys.append(col)
            foreign_keys_num += comb(cnt, 2)



    for tab_name, col_names in zip(table_names, column_names):
        primary_key_tmp = ''
        for col in col_names:
            if col.lower().endswith('id'):
                primary_key_tmp = col
                break
        if primary_key_tmp != '':
            primary_key.append(primary_key_tmp)
    primary_key_num = len(primary_key)
    return primary_key_num, foreign_keys_num



file_path = './aigsql/my_finetune_dataset.json'
# count_sql_query_complexity(file_path)

file_paths = ['./aigsql/my_finetune_dataset.json', './aigsql/all_data_48k_gpt4_score.json', './aigsql/gpt4_true_cases.json']
count_databse_schema_complexity(file_paths)