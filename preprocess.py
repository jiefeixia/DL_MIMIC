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
parser.add_argument('--min_freq', default=10, type=int, help='learning rate')  # wrong descriptions?
parser.add_argument('--max_df', default=0.8, type=float, help='batch size')  # wrong descriptions?
args = parser.parse_args()

if __name__ == '__main__':
    # read data
    df = pd.read_csv(os.path.join(check_sys_path(), "discharge_notes_with_medication_full_text_18APR.csv"))
    df = df[df["admission_notes"].notna()]

    # set random indexing for splitting training and validation data
    print("random splitting data into training and validation data set...")
    random_idx = np.random.permutation(np.arange(df.shape[0]))
    train_idx = random_idx[0:int(0.8 * len(random_idx))]  # select first 80% as training data
    val_idx = random_idx[int(0.8 * len(random_idx)):int(0.9 * len(random_idx))]  # random select 10% as validation data
    test_idx = random_idx[int(0.9 * len(random_idx)):-1]  # random select 10% as test data

    # calculate relative freq for each label
    print("relative freq for each label in training data")
    medicines = list(df.iloc[train_idx, -8:].columns)
    freq = df.iloc[train_idx, -8:].sum() / train_idx.shape[0]
    for i, medicine in enumerate(medicines):
        print("%20s:%.2f" % (medicine, freq[i]))

    print("cleaning word...")
    # discharge_notes = df["discharge_notes"][train_idx].fillna("").tolist()
    admission_notes = df["admission_notes"].fillna("").tolist()
    for i in range(len(admission_notes)):
        admission_notes[i] = re.sub(r"\d\. ", " ordernum ", admission_notes[i])
        admission_notes[i] = re.sub(r"\d\d:\d\d", " hourtime ", admission_notes[i])
        admission_notes[i] = re.sub(r"\d+", " num ", admission_notes[i])
        admission_notes[i] = re.sub("_", " ", admission_notes[i])
        admission_notes[i] = re.sub(r"\. ", " eos ", admission_notes[i])

    print("training nlp model...")
    vectorizer = CountVectorizer(min_df=args.min_freq,
                                 stop_words="english",
                                 max_df=args.max_df)
    vectorizer.fit(admission_notes)
    # vectorizer.fit(discharge_notes)

    word2idx = vectorizer.vocabulary_
    idx2word = [word for word, idx in word2idx.items()]
    vocab = word2idx.keys()
    # reserve 0 for padding value
    idx2word.append(idx2word[0])
    word2idx[idx2word[0]] = len(word2idx) + 1
    idx2word[0] = " "
    word2idx[" "] = 0

    freq_stop_words = vectorizer.stop_words_
    tokenizer = vectorizer.build_tokenizer()

    # transform word to idx
    print("transforming word to idx...")
    # discharge_notes_idx = np.array([np.array([word2idx[token] for token in tokenizer(note) if token in vocab])
    #                                 for note in tqdm(discharge_notes)])
    admission_notes_idx = np.array([np.array([word2idx[token] for token in tokenizer(note) if token in vocab])
                                    for note in admission_notes])

    # np.save(os.path.join(check_sys_path(), "embedding_train_idx.npy"), discharge_notes_idx)
    np.save(os.path.join(check_sys_path(), "train_idx.npy"), admission_notes_idx[train_idx])
    np.save(os.path.join(check_sys_path(), "train_label.npy"), np.array(df.loc[train_idx, MEDICINE_COL]))
    np.save(os.path.join(check_sys_path(), "val_idx.npy"), admission_notes_idx[val_idx])
    np.save(os.path.join(check_sys_path(), "val_label.npy"), np.array(df.loc[val_idx, MEDICINE_COL]))
    np.save(os.path.join(check_sys_path(), "test_idx.npy"), admission_notes_idx[test_idx])
    np.save(os.path.join(check_sys_path(), "test_label.npy"), np.array(df.loc[test_idx, MEDICINE_COL]))

    # save dict
    with open(os.path.join(check_sys_path(), "word2idx.txt"), "w") as f:
        f.write("\n".join(["%s:%d" % (word, idx) for word, idx in word2idx.items()]))
    with open(os.path.join(check_sys_path(), "med2idx.txt"), "w") as f:
        for idx, medicine in enumerate(MEDICINE_COL):
            f.write("%s:%s\n" % (medicine, idx))
