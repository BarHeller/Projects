import os
from data import *
from collections import defaultdict

def most_frequent_train(train_data):
    """
    Gets training data that includes tagged sentences.
    Returns a dictionary that maps every word in the training set to its most frequent tag.
    The dictionary should have a default value.
    """
    ### YOUR CODE HERE
    tags = {}
    for sentence in train_data:
        for word, tag in sentence:
            if word not in tags:
                tags[word] = {}
            tags[word][tag] = 1 if tag not in tags[word] else tags[word][tag] + 1
    frequent_tags = {word: (sorted(tags[word].items(), key= lambda x: x[1], reverse=True)[0][0]) for word in tags}
    return frequent_tags
    ### END YOUR CODE

def most_frequent_eval(test_set, pred_tags):
    """
    Gets test data and tag prediction map.
    Returns an evaluation of the accuracy of the most frequent tagger.
    """
    gold_tag_seqs = []
    pred_tag_seqs = []
    for sent in test_set:
        words, true_tags = zip(*sent)
        gold_tag_seqs.append(true_tags)

        ### YOUR CODE HERE
        pred_tag_seqs.append(tuple(pred_tags[word] if word in pred_tags else pred_tags['UNK'] for word,_ in sent))
        ### END YOUR CODE

    return evaluate_ner(gold_tag_seqs, pred_tag_seqs)

if __name__ == "__main__":
    train_sents = read_conll_ner_file("data/train.conll")
    dev_sents = read_conll_ner_file("data/dev.conll")
    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)
    model = most_frequent_train(train_sents)
    most_frequent_eval(dev_sents, model)
