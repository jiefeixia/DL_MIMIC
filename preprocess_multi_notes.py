import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from loader import check_sys_path
from sklearn.feature_extraction.text import CountVectorizer
import argparse
import re

NOTES_COL = ["allergy", "chief complaint", "history of present illness", "past medical history", "past procedure",
             "social history", "family history", "initial exam", "admission medications", "pertinent results"]
MEDICINE_COL = ["metoprolol", "furosemide", "lisinopril", "amlodipine", "atenolol", "hydrochlorothiazide", "diltiazem",
                "carvedilol"]

parser = argparse.ArgumentParser(description='embedding model')
args = parser.parse_args()

if __name__ == '__main__':
    # read data
    df = pd.read_csv(os.path.join(check_sys_path(), "discharge_notes_with_medication.csv"))

    print("cleaning word...")
    df["admission_notes"] = ""
    for col in NOTES_COL:
        df[col] = df[col].str.lower().replace(r"\d\. ", " ordernum ", regex=True)
        df[col] = df[col].str.replace(r"\d\d:\d\d", " hourtime ", regex=True)
        df[col] = df[col].str.replace(r"\d+", " num ", regex=True)
        df[col] = df[col].str.replace("_", " ", regex=True)
        df[col] = df[col].str.replace(r"\. ", " <eos> ", regex=True)
        df["admission_notes"] = df["admission_notes"] + df[col]

    df = df.dropna(subset=["admission_notes"])
    admission_notes = df["admission_notes"].dropna().tolist()

    print("training nlp model...")
    vectorizer = CountVectorizer(stop_words="english")
    vectorizer.fit(df["admission_notes"])

    word2idx = vectorizer.vocabulary_
    idx2word = [None] * len(word2idx)
    for word, idx in word2idx.items():
        idx2word[idx] = word
    vocab = word2idx.keys()

    # reserve 0 for padding value
    idx2word.append(idx2word[0])
    word2idx[idx2word[0]] = len(word2idx) + 1
    idx2word[0] = " "
    word2idx[" "] = 0

    freq_stop_words = vectorizer.stop_words_
    tokenizer = vectorizer.build_tokenizer()

    print("transforming word to idx...")
    df[NOTES_COL] = df[NOTES_COL].applymap(lambda x: [word2idx[token] for token in tokenizer(x) if token in vocab])

    df = df.dropna(thresh=3)  # if one row has more than three nan columns, then drop it

    # set random indexing for splitting training and validation data
    print("random splitting data into training and validation data set...")
    df = df.sample(frac=1).reset_index(drop=True)

    def save(file, idx, label):
        file.write(label + "\n")
        freq = df.loc[idx, MEDICINE_COL].sum() / len(idx)
        for i, med in enumerate(MEDICINE_COL):
            f.write("%s:%.2f(%d)\n" % (med, freq[i], df[med][idx].sum()))
        f.write("total#: %d" % len(idx))

        df.loc[idx, NOTES_COL].to_csv("%s_idx.csv" % label, index=False)
        df.loc[idx, MEDICINE_COL].to_csv("%s_label.csv" % label, index=False)

    with open("data_stat.txt", "w") as f:
        save(f, df.index[:len(df)//10*8], "train")
        save(f, df.index[len(df)//10*8:len(df)//10*9],  "val")
        save(f, df.index[len(df)//10*9:], "test")

    with open(os.path.join(check_sys_path(), "word2idx.txt"), "w") as f:
        f.write("\n".join(["%s:%d" % (word, idx) for word, idx in word2idx.items()]))
    with open(os.path.join(check_sys_path(), "med2idx.txt"), "w") as f:
        for idx, medicine in enumerate(MEDICINE_COL):
            f.write("%s:%s\n" % (medicine, idx))
