import os 
import json
from collections import defaultdict
import random
import pandas as pd

id2label = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5:'F'}
label2id = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4,'F': 5}
answer_dict = defaultdict(list)
vimmrc_path = '../datasets/KALAPA_ByteBattles_2023_MEDICAL_Set1/ViMMRC'
folders = ['dev', 'test', 'train']
for folder in folders:
    folder_path = os.path.join(vimmrc_path, folder)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        # if not file.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # print(data)\
            # article: <str> -- context
            # questions: [<str>] -- multiple questions
            # options: [[<str>]] -- multiple options for each question
            # answers: [<str>] -- answer for each question
            context = "%%Corpus 1: " + data['article']
            for question, options, answer in zip(data['questions'], data['options'], data['answers']):
                answer_dict['context'].extend([context]*2)
                answer_dict['question'].extend([question]*2)
                correct_option = options[label2id[answer]]
                wrong_option = random.choice([opt for opt in options if opt != correct_option])
                answer_dict['option'].extend([correct_option, wrong_option])
                answer_dict['label'].extend([1, 0])

df = pd.DataFrame(answer_dict)
df['instruction'] = ''

print(df.head())
print(len(df))
output_path = '../datasets/KALAPA_ByteBattles_2023_MEDICAL_Set1/ViMMRC/processed_train_file.csv'
df.to_csv(output_path, index=False)

