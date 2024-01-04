import json
import pandas as pd


def read_jsonl_to_list(file_path):
    data_list = []
    with open(file_path, 'r') as file:
        for line in file:
            data_list.append(json.loads(line))
    return data_list


def load_data(num_code_data: int, num_instruction_data: int, num_p3_data: int, simple_responses: False):
    instructions = []
    responses = []
    with open('data/code_alpaca/code_alpaca_20k.json', 'r') as f:
        code_data = json.load(f)[:num_code_data]
    instructions.extend([example['instruction'] + ' ' + example['input'] for example in code_data])
    responses.extend([example['output'] for example in code_data])
    instructions_df = pd.read_pickle('data/alpaca/alpaca_instructions_simple_responses_df.pkl')[:num_instruction_data]

    instructions.extend(instructions_df['instruction'].values)
    if simple_responses:
        responses.extend(instructions_df['simple_response'].values)
    else:
        responses.extend(instructions_df['response'].values)

    p3_data_list = read_jsonl_to_list("data/combined_mini_p3/combined_mini_p3.jsonl")[:num_p3_data]
    instructions.extend([example['source'] for example in p3_data_list])
    responses.extend([example['target'] for example in p3_data_list])

    return instructions, responses


def load_bias_data():
    '''
    Load bias data. Currently we do not have instructions, so we just use responses.
    :return:
    '''
    with open('data/bias/train_100.txt', 'r') as f:
        bias_data = f.readlines()
    bias_data = [line.strip() for line in bias_data]
    instructions = ["" for _ in range(len(bias_data))]
    responses = bias_data

    return instructions, responses