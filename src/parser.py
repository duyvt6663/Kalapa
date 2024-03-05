import pandas as pd
import json
import re
import ahocorasick
import pickle
from collections import defaultdict
import os
from rapidfuzz import fuzz
import shutil

ontology = "../datasets/KALAPA_ByteBattles_2023_MEDICAL_Set1/disease.json"

# def fill_synonyms():
#     data = json.load(open(ontology, 'r', encoding='utf-8'))
#     pattern = r'\(.*?\)$'  # regex pattern to match any string ending with (something)

#     for i in range(len(data)):
#         disease = data[i]['synonyms'][0]
        
#         new_disease = disease[0].upper() + disease[1:]
#         data[i]['synonyms'].append(new_disease)

#         if disease.startswith('bệnh'):
#             new_diseases = list(set([disease[5].upper() + disease[6:], disease[5:]]))
#             data[i]['synonyms'].extend(new_diseases)
        
#         match = re.search(pattern, disease)
#         if match:
#             print(f'Disease "{disease}" ends with "{match.group()}"')
#             response = input(f'Do you want to add this to the dataset? (y/n/c): ')
#             old_disease = disease.replace(match.group(), '').strip()
#             if response.lower() == 'y':
#                 new_disease = match.group()[1:-1]  # remove the parentheses
#                 new_list = list(set([new_disease, new_disease[0].upper() + new_disease[1:], old_disease]))
#                 data[i]['synonyms'].extend(new_list)

#                 print(f'New diseases: {new_list}') 
#             elif response.lower() == 'c':
#                 response = input('Enter the new disease: ')
#                 new_list = list(set([response, response[0].upper() + response[1:], old_disease]))
#                 data[i]['synonyms'].extend(new_list)

#                 print(f'New diseases: {new_list}')
#             else:
#                 data[i]['synonyms'].append(old_disease)
            
#     with open(ontology, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)
#         f.write('\n')


def build_trie():
    data = json.load(open(ontology, 'r', encoding='utf-8'))
    automaton = ahocorasick.Automaton()
    for i in range(len(data)):
        for word in data[i]['synonyms']:
            automaton.add_word(word, data[i]['name'])
    
    automaton.make_automaton
    with open("../datasets/KALAPA_ByteBattles_2023_MEDICAL_Set1/automaton.pickle", "wb") as f:
        pickle.dump(automaton, f)

def load_trie():
    load_path = "../datasets/KALAPA_ByteBattles_2023_MEDICAL_Set1/automaton.pickle"
    with open(load_path, 'rb') as f:
        automaton = pickle.load(f)
    return automaton

def test_trie():
    automaton = load_trie()
    automaton.make_automaton()
    string = """level2_23,Mụn trứng cá và mụn cóc phẳng có những điểm gì giống nhau?,Cả hai đều là các bệnh lý về da,Cả hai đều gây ra những biến chứng nghiêm trọng,Cả hai đều có nguy cơ lây nhiễm từ người này sang người khác,,,"""

    for key, root_disease in automaton.iter(string):
        print(root_disease)
        # the special case is with bệnh "ho" where it can match in the middle of the string


def cluster_data():

    ans = defaultdict(list)
    diseases = json.load(open(ontology, 'r', encoding='utf-8'))
    diseases = [disease['name'] for disease in diseases]
    subsections_path = "../datasets/KALAPA_ByteBattles_2023_MEDICAL_Set1/subsections/"

    for file in os.listdir(subsections_path):
        if not file.endswith('.json'):
            continue
        data = json.load(open(subsections_path + file, 'r', encoding='utf-8'))
        title = data['document_name_accent']

        # Calculate the similarity between the title and each disease
        similarities = [(disease, fuzz.ratio(disease, title)) for disease in diseases]

        # Sort the diseases by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)

        # The most similar disease is the first one in the sorted list
        most_similar_disease = similarities[0][0]
        ans[most_similar_disease].append(file)
        print(f"{file}: {most_similar_disease}")
        # Move the file to the disease folder
        shutil.move(subsections_path + file, subsections_path + most_similar_disease + '/' + file)
    
    # print(ans)

def check_empty_folders():
    subsections_path = "../datasets/KALAPA_ByteBattles_2023_MEDICAL_Set1/subsections/"
    for folder in os.listdir(subsections_path):
        if not os.listdir(subsections_path + folder):
            print(folder)

def check_misplaced_fodlers():
    SIZE = 3
    for folder in os.listdir("../datasets/KALAPA_ByteBattles_2023_MEDICAL_Set1/subsections/"):
        folder_path = "../datasets/KALAPA_ByteBattles_2023_MEDICAL_Set1/subsections/" + folder
        files = os.listdir(folder_path)
        if not all(file[:SIZE] == files[0][:SIZE] for file in files):
            print(folder)
        

def count_files():
    subsections_path = "../datasets/KALAPA_ByteBattles_2023_MEDICAL_Set1/subsections/"
    count = 0
    for folder in os.listdir(subsections_path):
        c = len(os.listdir(subsections_path + folder))
        if c == 0:
            print(folder)
        count += c
    
    print(f'Total: {count}')

# check_empty_folders()
build_trie()
# count_files()