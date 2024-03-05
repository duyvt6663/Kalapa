import os
import json

import streamlit as st
import pandas as pd
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


st.set_page_config(page_title="Labeler", page_icon="🏷️", layout="wide")


@st.cache_resource
def load_faiss_database(
    faiss_database_path,
):
    return FAISS.load_local(faiss_database_path)


@st.cache_resource
def load_bm25_docs(
    document_dir,
):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for folder in os.listdir(document_dir):
        folderpath = os.path.join(document_dir, folder)
        if not os.path.isdir(folderpath):
            continue
        for filename in sorted(os.listdir(folderpath)):
            filepath = os.path.join(folderpath, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                file_data = json.load(f)
                subs = text_splitter.split_text(file_data['subsection_content'])
                subs = [file_data['subsection_title'] + '\n' + text for text in subs]
                for i, sub in enumerate(subs):
                    docs.append(Document(
                        page_content=sub,
                        metadata={
                            "filename": filename,
                            "filepath": filepath,
                            "document_name": file_data["document_name"],
                            "document_name_accent": file_data["document_name_accent"],
                            "document_title": file_data["document_title"],
                            "document_category": file_data["document_category"],
                            "subsection_name": file_data["subsection_name"],
                            "subsection_title": file_data["subsection_title"],
                            "chunk_id": i
                        }
                    ))
    return docs


@st.cache_resource
def load_retriever(document_dir, retriever_name, k):
    if retriever_name == "BM25":
        docs = load_bm25_docs(document_dir)
        retriever = BM25Retriever.from_documents(docs)
        retriever.k = k
        return retriever
    elif retriever_name == "SBERT":
        retriever = load_faiss_database(saved_database_path)
        retriever = retriever.as_retriever(search_kwargs=dict(k=k))
        return retriever
    else:
        raise ValueError(f"Unknown retriever name: {retriever_name}")


# sidebar elements
retriever_method = st.sidebar.selectbox("Select your retriever:", ["BM25", "SBERT"])
st.sidebar.write("")
num_retrie_docs = st.sidebar.slider("Number of retrieved documents:", 1, 10, 5)
hide_labeled_samples = st.sidebar.checkbox("Hide labeled samples", value=True)

if retriever_method == "SBERT":
    sbert_model_name = st.sidebar.selectbox(
        "Select your SBERT model:",
        [
            "distiluse-base-multilingual-cased-v1",
            "paraphrase-xlm-r-multilingual-v1",
            "paraphrase-multilingual-mpnet-base-v2",
        ],
    )
    saved_database_path = st.sidebar.text_input(
        "Enter the path to the saved database:",
        value="./saved_database",
        key="saved_database_path",
    )
elif retriever_method == "BM25":
    document_dir = st.sidebar.text_input(
        "Enter the path to the document directory:",
        value="./datasets/KALAPA_ByteBattles_2023_MEDICAL_Set1/subsections",
        key="document_dir",
    )


retriever = load_retriever(
    document_dir=document_dir if retriever_method == "BM25" else saved_database_path,
    retriever_name=retriever_method,
    k=num_retrie_docs,
)


@st.cache_resource
def load_dataframe(df_path):
    try:
        df = pd.read_csv(df_path, index_col=0, dtype={'label': str})
        return df
    except Exception as e:
        print(e)
        st.error(f"Cannot load dataframe from {df_path}")
        return None


@st.cache_resource
def load_sentence_transformer_model(sbert_model_name):
    return SentenceTransformerEmbeddings(
        model_name=sbert_model_name,
        # model_kwargs={},
        encode_kwargs={
            "batch_size": 32,
            "normalize_embeddings": True,
            # "device": device,
        },
        # cache_folder=cache_dir,
    )


# main page elements
st.title("KALAPA Dataset Labeling Tool")
raw_df = None
labeled_df = None

with st.expander("Path and Dir Setting", expanded=True) as path_and_dir_setting:
    df_path = st.text_input(
        "Enter the path to the dataset.csv:",
        value="./datasets/KALAPA_ByteBattles_2023_MEDICAL_Set1/public_test.csv",
    )
    label_df_path = st.text_input(
        "Enter the path to the labeled_dataset.csv:",
        value="./datasets/KALAPA_ByteBattles_2023_MEDICAL_Set1/labeled_public_test.csv",
    )

    raw_df = load_dataframe(df_path)
    # if not os.path.exists(label_df_path):
    #     # create a new file with the same df structure and content, then save
    #     labeled_df = raw_df.copy()
    #     # add a new column for storing labels
    #     labeled_df["label"] = ""
    #     labeled_df.to_csv(label_df_path)
    labeled_df = load_dataframe(label_df_path)

    load_df_button = st.button("Save", key="load_df_button")
    if load_df_button:
        raw_df = load_dataframe(df_path)
        if not os.path.exists(label_df_path):
            # create a new file with the same df structure and content, then save
            labeled_df = raw_df.copy(deep=True)
            # add a new column for storing labels
            labeled_df["label"] = ""
            labeled_df.to_csv(label_df_path)
            del labeled_df
        labeled_df = pd.read_csv(label_df_path, index_col=0)


with st.expander("Show/Hide Raw Dataset"):
    st.dataframe(raw_df)


if raw_df is None:
    st.error("Please load the dataset first.")
    st.stop()


st.header("Labeling Workspace")

# print all sample id without labels
ids_without_labels = []
for idx, (i, row) in enumerate(labeled_df.iterrows()):
    if pd.isna(row["label"]):
        ids_without_labels.append(idx)
with st.expander("Sample IDs without labels"):
    st.markdown(f"{json.dumps(ids_without_labels, indent=2, ensure_ascii=False)}")

sample_id = st.number_input(
    "Enter the sample id:", min_value=0, max_value=len(raw_df) - 1, value=0
)

cur_sample = raw_df.iloc[sample_id].to_dict()

cur_question = cur_sample["question"]
option_1 = cur_sample["option_1"]
option_2 = cur_sample["option_2"]
option_3 = cur_sample["option_3"]
option_4 = cur_sample["option_4"]
option_5 = cur_sample["option_5"]
option_6 = cur_sample["option_6"]
cur_label = labeled_df.iloc[sample_id]["label"]

st.markdown(f"###### Question: {cur_question}")
max_option_len = max(
    [
        len(str(option))
        for option in [option_1, option_2, option_3, option_4, option_5, option_6]
        if option is not pd.isnull(option)
    ]
)
print(max_option_len)
if max_option_len >= 32:
    option_col_weights = [1, 1]
elif max_option_len >= 16:
    option_col_weights = [1, 1, 1]
elif max_option_len >= 8:
    option_col_weights = [1, 1, 2]
else:
    option_col_weights = [1, 1, 3]
option_col12 = st.columns(option_col_weights)
with option_col12[0]:
    st.markdown(f"Option 1: {option_1}")
with option_col12[1]:
    st.markdown(f"Option 2: {option_2}")
option_col34 = st.columns(option_col_weights)
with option_col34[0]:
    st.markdown(f"Option 3: {option_3}")
with option_col34[1]:
    st.markdown(f"Option 4: {option_4}")
option_col56 = st.columns(option_col_weights)
with option_col56[0]:
    st.markdown(f"Option 5: {option_5}")
with option_col56[1]:
    st.markdown(f"Option 6: {option_6}")

if cur_label:
    st.markdown(f"###### Label: {cur_label}")

if hide_labeled_samples and not pd.isna(cur_label):
    st.info(
        "This sample is labeled, please uncheck the `Hide labeled samples` checkbox to show it."
    )
    st.stop()


cur_input_label = st.text_input(
    "Enter the label:",
    key=f"input_label_{sample_id}",
)
if st.button("Save", key=f"save_button_{sample_id}"):
    num_available_options = sum(
        [
            1 if not pd.isna(o) else 0
            for o in [option_1, option_2, option_3, option_4, option_5, option_6]
        ]
    )
    if len(cur_input_label) != num_available_options:
        st.error(
            f"Please enter the label with {num_available_options} characters, e.g. {'1' * num_available_options}"
        )
    else:
        labeled_df["label"][sample_id] = cur_input_label
        labeled_df.to_csv(label_df_path)

        st.success(
            f"Saved the label {labeled_df.iloc[sample_id]['label']} for sample {sample_id}"
        )

related_docs = retriever.get_relevant_documents(cur_question + '\n' + '\n'.join([option for option in [option_1, option_2, option_3, option_4, option_5, option_6] if str(option) != 'nan']))
if len(related_docs) > 0:
    st.markdown("###### Related Documents:")
    for i, doc in enumerate(related_docs):
        with st.expander(f"Document {i}: {doc.metadata['document_name']}"):
            st.markdown(f"**{doc.metadata['subsection_title']}**")
            st.markdown(f"{doc.page_content}")
