from model import *
import torch.optim as optim
import numpy as np
from loader import check_sys_path
import os
from torch.nn.utils.rnn import pack_sequence
from loader import *
from torch.utils.data import DataLoader
import torch
import argparse
from sklearn.neighbors import NearestNeighbors


parser = argparse.ArgumentParser(description='embedding model')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--context_size', default=4, type=int, help='length of content after target')
parser.add_argument('--embedding_dim', default=50, type=int, help='word embedding dimension')
parser.add_argument('--epoch', "-e", default=5, type=int, help='max epoch')
parser.add_argument('--predict', type=str, help='transform idx data to embedding data')
args = parser.parse_args()

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    net = NGramLanguageModeler(vocab_size=IdxData.get_vacab_size(),
                               embedding_dim=args.embedding_dim,
                               context_size=args.context_size)
    net.cuda()
    if args.predict:
        args.epoch = 0

    else:
        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        train_dataset = IdxData(args.context_size, "embedding_train_idx.npy")
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=4,
                                  shuffle=True)

        # train embedding layer
        print("training embedding model")
    for epoch in range(args.epoch):
        net.train()
        running_loss = 0.
        with tqdm(total=int(len(train_loader)), ascii=True) as pbar:
            for batch_idx, words in enumerate(train_loader):
                optimizer.zero_grad()

                words = words.long().cuda()
                context = words[:, 0:args.context_size]
                target = words[:, args.context_size]
                log_probs = net(context)
                loss = loss_function(log_probs, target)

                running_loss += loss.item()
                loss.backward()
                optimizer.step()

                # update progress bar
                if batch_idx % 10 == 0:
                    niter = epoch * len(train_loader) + batch_idx
                    pbar.set_postfix(T_current_loss=round(loss.item(), 4))

                    pbar.update(10 if pbar.n + 50 <= pbar.total else pbar.total - pbar.n)

            running_loss /= len(train_loader)

        print("\re %3d: |Train avg loss: %.3f" % (epoch, running_loss))

    print("saving embedding layer in")
    torch.save(net.state_dict(), 'word_embedding%d.pth' % args.embedding_dim)

    # transform idx data to embedding data
    if args.predict:
        print("loading exist model from %s" % args.predict)
        net.load_state_dict(torch.load(args.predict))


    def transform(infile, outfile):
        """
        :param infile: npy filename, shape=(num_notes, num_words)
        :param outfile: npy filename, shape=(num_notes, num_words, embedding_dim)
        """
        print("transforming idx data %s to embedding data %s..." % (infile, outfile))
        transform_dataset = IdxData(context_size=-1, file=infile)
        transform_loader = DataLoader(transform_dataset,
                                      batch_size=1,  # because the num of words in different note is different
                                      num_workers=1,
                                      shuffle=False)
        with torch.no_grad():
            net.eval()
            embeddings = []
            for note in transform_loader:
                note = note.long().cuda()
                embedding = net.embeddings(note)

                embedding = embedding.cpu().numpy().astype("float32")
                embeddings.append(embedding[0])  # batch size = 1, select the item

        notes_embedding = np.array(embeddings)
        np.save(os.path.join(check_sys_path(), outfile), notes_embedding)


    transform("train_idx.npy", "train_%dembedding.npy" % args.embedding_dim)
    transform("val_idx.npy", "val_%dembedding.npy" % args.embedding_dim)

    # find most closed words to medicines on embedding space to evaluate
    print("finding most relevant words of medicine (nearest neighbour on embedding space...")
    # TODO: test performance on T-SNE low dimension space

    idx_med = dict()
    with open(os.path.join(check_sys_path(), "med_idx.txt")) as f:
        for line in f:
            med, idx = line.split(":")
            idx_med[int(idx)] = med

    idx_word = dict()
    with open(os.path.join(check_sys_path(), "word_idx.txt")) as f:
        for line in f:
            word, idx = line.split(":")
            idx_word[int(idx)] = word

    print("calculating embeddings for all words and medicines")
    # get embedding of all words
    for med in idx_med.keys():  # remove medicines idx from words idx
        idx_word.pop(med, None)
    words = np.array(list(idx_word.keys()))
    words_embedding = net.embeddings(torch.from_numpy(words).long().cuda())
    words_embedding = words_embedding.cpu().detach().numpy()

    # get embedding of all medicines
    meds = np.array(list(idx_med.keys()))
    meds_embedding = net.embeddings(torch.from_numpy(meds).long().cuda())
    meds_embedding = meds_embedding.cpu().detach().numpy()

    n_neighbors = 5
    print("calculating top %d nearest neighbours for medicines" % n_neighbors)
    neigh = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    neigh.fit(words_embedding)
    dist, ind = neigh.kneighbors(meds_embedding)

    with open("result/word_embedding%d.txt" % args.embedding_dim, "w") as f:
        for i, med in enumerate(list(idx_med.values())):
            s = med + ": " + str([(idx_word[words[ind[i, j]]], round(dist[i, j], 2)) for j in range(n_neighbors)])
            print(s)
            f.write(s + "\n")


