import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from loader import check_sys_path
from sklearn.feature_extraction.text import CountVectorizer
import argparse

NOTES_COL = ["allergy", "chief complaint", "history of present illness", "past medical history", "past procedure",
             "social history", "family history", "initial exam", "admission medications", "pertinent results"]
MEDICINE_COL = ["metoprolol", "furosemide", "lisinopril", "amlodipine", "atenolol", "hydrochlorothiazide", "diltiazem",
                "carvedilol"]

parser = argparse.ArgumentParser(description='embedding model')
parser.add_argument('--min_freq', default=10, type=int, help='learning rate')#wrong descriptions?
parser.add_argument('--max_df', default=0.8, type=float, help='batch size')#wrong descriptions?
args = parser.parse_args()

if __name__ == '__main__':
    # read data
    df = pd.read_csv(os.path.join(check_sys_path(), "discharge_notes_with_medication_full_textv2.csv"))
    df = df[df["admission_notes"].notna()]

    # set random indexing for splitting training and validation data
    print("random splitting data into training and validation data set...")
    random_idx = np.random.permutation(np.arange(df.shape[0]))
    train_idx = random_idx[0:int(0.8 * len(random_idx))]  # select first 80% as training data
    val_idx = random_idx[int(0.8 * len(random_idx)):-1]  # random select 20% as validation data

    discharge_notes = df["discharge_notes"][train_idx].fillna("").tolist()
    admission_notes = df["admission_notes"].fillna("").tolist()
    # calculate relative freq for each label
    print("relative freq for each label in training data")
    medicines = list(df.iloc[train_idx, -8:].columns)
    freq = df.iloc[train_idx, -8:].sum()/train_idx.shape[0]
    for i, medicine in enumerate(medicines):
        print("%20s:%.2f" % (medicine, freq[i]))

    # train nlp model
    print("training nlp model...")
    vectorizer = CountVectorizer(min_df=args.min_freq,
                                 stop_words="english",
                                 max_df=args.max_df)
    vectorizer.fit(discharge_notes)

    word_idx = vectorizer.vocabulary_
    idx_word = {idx: word for word, idx in word_idx.items()}
    vocab = word_idx.keys()
    freq_stop_words = vectorizer.stop_words_
    tokenizer = vectorizer.build_tokenizer()

    # transform word to idx
    print("transforming word to idx...")
    discharge_notes_idx = np.array([np.array([word_idx[token] for token in tokenizer(note) if token in vocab])
                                    for note in tqdm(discharge_notes)])
    admission_notes_idx = np.array([np.array([word_idx[token] for token in tokenizer(note) if token in vocab])
                                    for note in admission_notes])

    np.save(os.path.join(check_sys_path(), "embedding_train_idx.npy"), discharge_notes_idx)
    np.save(os.path.join(check_sys_path(), "train_idx.npy"), admission_notes_idx[train_idx])
    np.save(os.path.join(check_sys_path(), "train_label.npy"), np.array(df.loc[train_idx, MEDICINE_COL]))#.to_numpy())
    np.save(os.path.join(check_sys_path(), "val_idx.npy"), admission_notes_idx[val_idx])
    np.save(os.path.join(check_sys_path(), "val_label.npy"), np.array(df.loc[val_idx, MEDICINE_COL]))#.to_numpy())

    # random oversample
    x_train = np.load(os.path.join(check_sys_path(), "train_idx.npy")
    y_train = np.load(os.path.join(check_sys_path(), "train_label.npy")
    # oversample lowest 3 drugs
    minority_indices = np.where(y_train[:,5:] == 1)[0]
    num_minority_instances = np.sum(y_train[:,5:] == 1)
    train_length = len(y_train)
    num_samples = 2000
    # sample from minority indices num_samples times with replacement
    y_trainn = list(y_train)
    x_trainn = list(x_train)
    for idx, i in enumerate(range(num_samples)):
        random_sample = np.random.randint(0,num_minority_instances)
        index_to_add = minority_indices[random_sample]
        y_trainn.append(y_trainn[index_to_add])
        x_trainn.append(x_trainn[index_to_add])

    y_train_oversampled = np.array(y_trainn)
    x_train_oversampled = np.array(x_trainn)

    print("After oversampling")
    for i in range(len(MEDICINE_COL)):
    print(f'{MEDICINE_COL[i]}: {np.sum(y_train_oversampled[:,i] == 1) / len(y_train_oversampled)}\n')

    # save dict
    with open(os.path.join(check_sys_path(), "word_idx.txt"), "w") as f:
        f.write("\n".join(["%s:%d" % (word, idx) for word, idx in word_idx.items()]))
    with open(os.path.join(check_sys_path(), "med_idx.txt"), "w") as f:
        for idx, medicine in enumerate(MEDICINE_COL):
            f.write("%s:%s\n" % (medicine, idx))
