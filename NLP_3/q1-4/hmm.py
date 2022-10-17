import os
import time
from data import *
from collections import defaultdict, Counter
import math

def hmm_train(sents):
    """
        sents: list of tagged sentences
        Returns: the q-counts and e-counts of the sentences' tags, total number of tokens in the sentences
    """

    print("Start training")
    total_tokens = 0
    # YOU MAY OVERWRITE THE TYPES FOR THE VARIABLES BELOW IN ANY WAY YOU SEE FIT
    q_tri_counts, q_bi_counts, q_uni_counts, e_tag_counts = [defaultdict(int) for i in range(4)]
    e_word_tag_counts = defaultdict(lambda: defaultdict(int))
    ### YOUR CODE HERE
    for sentence in sents:
        total_tokens += len(sentence)
        padded_sentence = [("*", "*"),("*", "*")] + sentence + [("STOP", "STOP")]
        for i in range(2, len(sentence)):
            current_word,   current_tag   = padded_sentence[i]
            prev_word,      prev_tag      = padded_sentence[i-1]
            prev_prev_word, prev_prev_tag = padded_sentence[i-2]
            q_uni_counts[current_tag] += 1
            q_bi_counts [(prev_tag, current_tag)] += 1
            q_tri_counts[(prev_prev_tag, prev_tag, current_tag)] += 1
            e_word_tag_counts[current_word][current_tag] +=1
    e_tag_counts = dict(q_uni_counts)

    ### END YOUR CODE
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts


def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                e_word_tag_counts, e_tag_counts, lambda1, lambda2):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))
    ### YOUR CODE HERE
    lambda3 = 1 - (lambda1 + lambda2)
    def get_tags(i):
        return ['*'] if i < 0 else list(e_word_tag_counts[sent[i][0]].keys())

    def calc_q(w, u , v):
        q_uni = 0
        q_bi = 0
        q_tri = 0
        if q_bi_counts[(w, u)] > 0:
            q_tri = q_tri_counts[(w, u, v)]/float(q_bi_counts[(w, u)])
        if q_uni_counts[u] > 0:
            q_bi = q_bi_counts[(u, v)]/float(q_uni_counts[u])
        if q_uni_counts[v] > 0:
            q_uni = q_uni_counts[v]/float(total_tokens)
        
        return lambda3 * q_uni + lambda2 * q_bi + lambda1 * q_tri

    def calc_e(word, tag):
        if word in e_word_tag_counts.keys() and tag in e_tag_counts:
            return float(e_word_tag_counts[word][tag]/e_tag_counts[tag])
        return 0

    pi = [dict() for i in range(len(sent) + 1)]
    pi[0] = {('*', '*'): [1, '']}

    for i in range(len(sent)):
        curr_word = sent[i][0]
        for v in get_tags(i):
            curr_e = calc_e(curr_word, v)
            if curr_e > 0:
                for u in get_tags(i - 1):
                    pi[i + 1][(u, v)] = [-math.inf, 'O']
                    for w in get_tags(i - 2):
                        curr_prob = pi[i][(w, u)][0] * calc_q(w, u, v) * calc_e(curr_word, v)
                        if curr_prob > pi[i + 1][(u, v)][0]:
                            pi[i + 1][(u, v)][0] = curr_prob
                            pi[i + 1][(u, v)][1] = w

    if len(pi[-1].items()) == 0:
        pi[-1] = {('O', 'O'): [1, 'O']}

    u,v = max(pi[-1].items(), key=lambda x:x[1])[0]     
    if len(sent) == 1:
        predicted_tags[-1] = v
    else:
        predicted_tags[-1] = v
        predicted_tags[-2] = u
        for i in reversed(range(len(predicted_tags) - 2)):
            u = predicted_tags[i + 1]
            v = predicted_tags[i + 2]
            if (u,v) not in pi[i + 3]:
                continue
            else:      
                predicted_tags[i] = pi[i + 3][(u, v)][1]
    
    ### END YOUR CODE
    return predicted_tags

def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    print("Start evaluation")
    gold_tag_seqs = []
    pred_tag_seqs = []

    for sent in test_data:
        words, true_tags = zip(*sent)
        gold_tag_seqs.append(true_tags)

        ### YOUR CODE HERE
        pred_tag_seqs.append(hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                e_word_tag_counts, e_tag_counts, 0.3, 0.5))
        ### END YOUR CODE

    return evaluate_ner(gold_tag_seqs, pred_tag_seqs)

if __name__ == "__main__":
    start_time = time.time()
    train_sents = read_conll_ner_file("data/train.conll")
    dev_sents = read_conll_ner_file("data/dev.conll")
    vocab = compute_vocab_count(train_sents)

    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)

    hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
             e_word_tag_counts, e_tag_counts)

    train_dev_end_time = time.time()
    print("Train and dev evaluation elapsed: " + str(train_dev_end_time - start_time) + " seconds")
