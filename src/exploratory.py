
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
file = 'MEDICAL/public_test.csv'

df = pd.read_csv(file)
fields = ['id', 'question', 'option_1', 'option_2', 'option_3', 'option_4', 'option_5', 'option_6']
for field in fields:
    df[field] = df[field].apply(lambda x: x.lower() if pd.notnull(x) else x)

# for i, row in df.iterrows():
def plot_top_kgrams(df, n, k):
    words_freq = []
    for field in fields:
        column = df[field].dropna()
        if column.empty:
            continue
        vec = CountVectorizer(ngram_range=(n, n)).fit(column)
        bag_of_words = vec.transform(column)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq += [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    top_df = pd.DataFrame(words_freq[:k], columns=['ngram', 'frequency'])

    plt.figure(figsize=(10,5))
    plt.bar(top_df['ngram'], top_df['frequency'])
    plt.title(f'Top {k} {n}-grams')
    plt.xticks(rotation=45)
    plt.show()

for n in range(2, 6):
    plot_top_kgrams(df, n, 10)


