
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np
from langchain.vectorstores import FAISS
# import pandas as pd
import polars as pl
from search import AdvRetriever
import random
import pandas as pd

# {
#     "document_name": "buong-trung-da-nang",
#     "document_name_accent": "Buồng trứng đa nang",
#     "document_title": "Buồng trứng đa nang: Nguyên nhân, dấu hiệu, chẩn đoán và điều trị",
#     "document_category": "Sản - Phụ khoa",
#     "subsection_name": "buong-trung-da-nang_0_Buồng trứng đa nang là gì?",
#     "subsection_content": "Buồng trứng đa nang (tiếng Anh là Polycystic Ovary Syndrome – PCOS) là một dạng rối loạn nội tiết tố thường gặp ở phụ nữ trong độ tuổi sinh sản. Bệnh gây nhiều tác động lên buồng trứng khiến chu kỳ kinh nguyệt bị rối loạn, tăng nồng độ nội tiết tố nam, dẫn đến hình thành nhiều nang nhỏ bên trong buồng trứng.\nThống kê cho thấy, có khoảng 2,2 – 26,7% phụ nữ trong độ tuổi sinh sản (15 – 44 tuổi) mắc hội chứng đa nang buồng trứng. Và nhiều người trong số họ không biết mình mắc bệnh nên không điều trị sớm dẫn đến nhiều biến chứng. (1)",
#     "subsection_title": "Buồng trứng đa nang. Buồng trứng đa nang là gì?",
#     "subsection_data": "Buồng trứng đa nang. Buồng trứng đa nang là gì?\nBuồng trứng đa nang (tiếng Anh là Polycystic Ovary Syndrome – PCOS) là một dạng rối loạn nội tiết tố thường gặp ở phụ nữ trong độ tuổi sinh sản. Bệnh gây nhiều tác động lên buồng trứng khiến chu kỳ kinh nguyệt bị rối loạn, tăng nồng độ nội tiết tố nam, dẫn đến hình thành nhiều nang nhỏ bên trong buồng trứng.\nThống kê cho thấy, có khoảng 2,2 – 26,7% phụ nữ trong độ tuổi sinh sản (15 – 44 tuổi) mắc hội chứng đa nang buồng trứng. Và nhiều người trong số họ không biết mình mắc bệnh nên không điều trị sớm dẫn đến nhiều biến chứng. (1)"
# }

def chop_file():
    output_folder = '/kaggle/output/'
    os.makedirs(output_folder + '1000_chunks', exist_ok=True)
    os.makedirs(output_folder + '500_chunks', exist_ok=True)
    os.makedirs(output_folder + 'sent_chunks', exist_ok=True)

    model_name = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

    emb_func = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cuda'},
        encode_kwargs=encode_kwargs
    )

    dataset_path = '../datasets/KALAPA_ByteBattles_2023_MEDICAL_Set1/subsections/'
    folders = os.listdir(dataset_path)

    large_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    small_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
    chunks_1000, chunks_500, sent_chunks = output_folder + '1000_chunks/', output_folder + '500_chunks/', output_folder + 'sent_chunks/'

    import concurrent.futures

    for folder in folders:
        if not os.path.exists(chunks_1000 + folder):
            os.makedirs(chunks_1000 + folder)
        else:
            continue
        if not os.path.exists(chunks_500 + folder):
            os.makedirs(chunks_500 + folder)
        else:
            continue
        if not os.path.exists(sent_chunks + folder):
            os.makedirs(sent_chunks + folder)
        else:
            continue

        large_data, small_data, sentences = [], [], []
        for file in os.listdir(dataset_path + folder):
            data = json.load(open(dataset_path + folder + '/' + file, 'r', encoding='utf-8'))
            subsection_content = data['subsection_content']
            
            largechunks = large_splitter.split_text(subsection_content)
            smallchunks = small_splitter.split_text(subsection_content)
            sents = subsection_content.split('\n')
            
            doc_title = data['document_title']
            subsec_title =data['subsection_title'].split('_')[-1]
            largechunks = [f"document: {doc_title}\nsection: {subsec_title}\nsnippet: {chunk}" for chunk in largechunks]
            smallchunks = [f"document: {doc_title}\nsection: {subsec_title}\nsnippet: {chunk}" for chunk in smallchunks]
            sents = [f"document: {doc_title}\nsection: {subsec_title}\nsnippet: {sent}" for sent in sents]

            large_data.extend(largechunks)
            small_data.extend(smallchunks)
            sentences.extend(sents)
        
        embeddings_sents = emb_func.embed_documents(sentences)
        embeddings_large = emb_func.embed_documents(large_data)
        embeddings_small = emb_func.embed_documents(small_data)

        np.save(chunks_1000 + folder + '/embeddings.npy', embeddings_large)
        np.save(chunks_500 + folder + '/embeddings.npy', embeddings_small)
        np.save(sent_chunks + folder + '/embeddings.npy', embeddings_sents)

        json.dump(large_data, open(chunks_1000 + folder + '/data.json', 'w', encoding='utf-8'), ensure_ascii=False)
        json.dump(small_data, open(chunks_500 + folder + '/data.json', 'w', encoding='utf-8'), ensure_ascii=False)
        json.dump(sentences, open(sent_chunks + folder + '/data.json', 'w', encoding='utf-8'), ensure_ascii=False)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_folder, folders)



def load_embedding():
    model_name = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cuda'},
        encode_kwargs=encode_kwargs
    )

def process_chunk(chunk: pl.DataFrame, i):
    def shuffle_context(chunk):
        relevant_docs = np.random.choice(texts, (len(chunk), 2))
        new_list = np.column_stack((relevant_docs[:, 0], chunk['context'], relevant_docs[:, 1]))
        np.apply_along_axis(np.random.shuffle, axis=1, arr=new_list)
        new_list = np.vectorize(lambda contexts: np.array([f"%%Corpus {i+1}: {cont}" for i, cont in enumerate(contexts)]))(new_list)
        new_context = np.array([f'\n###\n'.join(row) for i, row in enumerate(new_list.astype(str))])
        return new_context

    chunk['context'] = shuffle_context(chunk)
    print(chunk)
    chunk.write_csv(f"MedMCQA_chunk_{i+1}.csv", index=False)
        
def chop_mcqa():
    global retriever, texts
    base_data_path = '/kaggle/input'
    medmcqa_path = os.path.join(base_data_path, "medmcqa-vi/processed_MedMCQA.csv")
    df = pl.read_csv(medmcqa_path)
    
    f = pl.col("context").str.starts_with
    df = df.filter(~(f('Trả lời') | f('Câu trả lời') | f('Phương án')))
    
    chunk_size = 20000
    num_chunks = len(df) // chunk_size
    if len(df) % chunk_size != 0:
        num_chunks += 1
    chunks = [df[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]

    # load retriever
    automaton_path = os.path.join(base_data_path, 'kalapa-keyword-search/automaton.pickle')
    document_path = os.path.join(base_data_path, 'kalapa-datasets-2/kalapa-datasets/1000_chunks')
    embedding = load_embedding()
    # retriever = AdvRetriever(automaton_path=automaton_path, document_path=document_path, embedding=embedding, k1=2, k2=1)
    texts, _ = AdvRetriever.retrieve_texts_and_embeddings(document_path)

#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         futures = executor.map(process_chunk, chunks, range(len(chunks)))
#         for future in concurrent.futures.as_completed(futures):
#             try:
#                 future.result()
#             except Exception as e:
#                 print('Process caused an exception:', e)
#                 traceback.print_exc()
    for i, chunk in enumerate(chunks):
        process_chunk(chunk, i)

def chop_public_test():
    data_path = '../datasets/KALAPA_ByteBattles_2023_MEDICAL_Set1/labeled_public_test.csv'
    output_path = '../datasets/KALAPA_ByteBattles_2023_MEDICAL_Set1/public_test_level/'
    df = pd.read_csv(data_path)
    # cut wrt levels
    df['level'] = df['id'].apply(lambda x: int(x[5]))
    # Collect dataframes based on level
    level_1_df = df[df['level'] == 1]
    level_2_df = df[df['level'] == 2]
    level_3_df = df[df['level'] == 3]
    level_4_df = df[df['level'] == 4]
    level_files = [level_1_df, level_2_df, level_3_df, level_4_df]
    for i, level_df in enumerate(level_files):
        level_df.to_csv(output_path + f'level_{i+1}.csv', index=False)

def chop_unroll_public_test():
    data_path = '../datasets/KALAPA_ByteBattles_2023_MEDICAL_Set1/labeled_public_test.csv'
    format_path = '../datasets/KALAPA_ByteBattles_2023_MEDICAL_Set1/formated_public_test.csv'
    output_path = '../datasets/KALAPA_ByteBattles_2023_MEDICAL_Set1/public_test_level/'
    df = pd.read_csv(data_path, dtype={'label':str})
    # cut wrt levels
    df['level'] = df['id'].apply(lambda x: int(x[5]))
    # unroll the level column
    lvls, ids = [], []
    for i, row in df.iterrows():
        lvls.extend([row['level']] * len(row['label']))
        ids.extend([row['id']] * len(row['label']))
    print(len(lvls))
    
    format_df = pd.read_csv(format_path)
    format_df['level'], format_df['id'] = lvls, ids
    format_df.drop(['Unnamed: 0', 'instruction'], axis=1, inplace=True)
    print(format_df.head())
    # Collect dataframes based on level
    level_1_df = format_df[format_df['level'] == 1]
    level_2_df = format_df[format_df['level'] == 2]
    level_3_df = format_df[format_df['level'] == 3]
    level_4_df = format_df[format_df['level'] == 4]
    level_files = [level_1_df, level_2_df, level_3_df, level_4_df]
    for i, level_df in enumerate(level_files):
        level_df.to_csv(output_path + f'level_{i+1}_unroll.csv', index=False)

def merge_context_translation():
    path = '../datasets/KALAPA_ByteBattles_2023_MEDICAL_Set1/PubMedQA'
    file_names = ['pubmed_a.json', 'translated_context.json']
    file_paths = [os.path.join(path, file) for file in file_names]
    raw, translated = [json.load(open(file_path, 'r', encoding='utf-8')) for file_path in file_paths]
    combined = []
    for (k, v), trans in zip(raw.items(), translated):
        combined.append(
            {
                'raw': v['CONTEXTS'],
                'translated': trans
            }
        )
    json.dump(combined, open(os.path.join(path, 'combined.json'), 'w', encoding='utf-8'), ensure_ascii=False)


if __name__ == '__main__':
    merge_context_translation()