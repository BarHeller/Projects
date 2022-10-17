from data import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import time
import os
import numpy as np
from collections import defaultdict

def build_extra_decoding_arguments(train_sents):
    """
    Receives: all sentences from training set
    Returns: all extra arguments which your decoding procedures requires
    """

    extra_decoding_arguments = {}
    ### YOUR CODE HERE
    extra_decoding_arguments['word_to_label'] = {}
    word_to_label = extra_decoding_arguments['word_to_label']
    for sent in train_sents:
        for word, label in sent:
            if word not in word_to_label:
                word_to_label[word] = set()
            word_to_label[word].add(label)

    ### END YOUR CODE

    return extra_decoding_arguments


def extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag):
    """
        Receives: a word's local information
        Returns: The word's features.
    """
    features = {}
    features['word'] = curr_word
    ### YOUR CODE HERE

    # unigrams
    features['next_word'] = next_word
    features['prev_word'] = prev_word
    features['prevprev_word'] = prevprev_word
    features['prev_tag'] = prev_tag
    features['prevprev_tag'] = prevprev_tag
    features['prev_word_tag'] = prev_word + "-" + prev_tag
    features['prevprev_word_tag'] = prevprev_word + "-" + prevprev_tag

    # bigrams
    features['prev_words_bigram'] = prevprev_word + "-" + prev_word
    features['curr_prev_word_bigram'] = prev_word + "-" + curr_word
    features['curr_next_word_bigram'] = curr_word + "-" + next_word
    features['tag_bigram'] = prevprev_tag + "-" + prev_tag

    # trigrams
    # tags-trigrams are impossible with the given data
    features['word_trigram'] = prevprev_word + "-" + prev_word + "-" + curr_word

    # prefix and suffix 
    for i in range(1, min(5, len(curr_word))):
        index_suffix = 'suffix_'+str(i)
        index_prefix = 'prefix_'+str(i)
        features[index_suffix] = curr_word[-i:]
        features[index_prefix] = curr_word[:i]

    # others
    # features['is_capital'] = 1 if curr_word[0].isupper() else 0
    # features['short_word'] = 1 if len(curr_word) < 3 else 0
    # features['special'] = 1 if curr_word[0].isalpha() else 0


    ### END YOUR CODE
    return features

def extract_features(sentence, i):
    curr_word = sentence[i][0]
    prev_token = sentence[i - 1] if i > 0 else ('<st>', '*')
    prevprev_token = sentence[i - 2] if i > 1 else ('<st>', '*')
    next_token = sentence[i + 1] if i < (len(sentence) - 1) else ('</s>', 'STOP')
    return extract_features_base(curr_word, next_token[0], prev_token[0], prevprev_token[0], prev_token[1], prevprev_token[1])

def vectorize_features(vec, features):
    """
        Receives: feature dictionary
        Returns: feature vector

        Note: use this function only if you chose to use the sklearn solver!
        This function prepares the feature vector for the sklearn solver,
        use it for tags prediction.
    """
    example = [features]
    return vec.transform(example)

def create_examples(sents, tag_to_idx_dict):
    examples = []
    labels = []
    num_of_sents = 0
    for sent in sents:
        num_of_sents += 1
        for i in range(len(sent)):
            features = extract_features(sent, i)
            examples.append(features)
            labels.append(tag_to_idx_dict[sent[i][1]])

    return examples, labels


def memm_greedy(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))
    ### YOUR CODE HERE
    sent_len = len(sent)
    for i in range(sent_len):
        features = extract_features(sent, i)
        vectorized_features = vectorize_features(vec, features)
        prediction = logreg.predict(vectorized_features)[0]
        predicted_tags[i] = index_to_tag_dict[prediction]
    ### END YOUR CODE
    return predicted_tags

def memm_viterbi(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))
    ### YOUR CODE HERE
    def get_tags(i):
        return ['*'] if i < 0 else extra_decoding_arguments['word_to_label'][sent[i][0]]

    def calc_q_dict(sent):
        features = []
        q = dict()
        indices = dict()
        j = 0
        for i in range(len(sent)):
            word = sent[i]
            word_features = extract_features(sent, i)
            tags_prev_prev = get_tags(i - 2)
            tags_prev = get_tags(i - 1)
            for t in tags_prev_prev:
                for u in tags_prev:
                    current_features = dict(word_features)
                    current_features['prev_tag'] = u
                    current_features['prevprev_tag'] = t
                    current_features['tag_bigram'] = t + "-" + u
                    features.append(current_features)#features[j] = current_features
                    indices[j] = (i,t,u)
                    j += 1
                
        feature_vec = vec.transform(features)
        predictions = logreg.predict_proba(feature_vec)
        
        for j in indices.keys():           
            q[indices[j]] = predictions[j]
        return q

    pi = [dict() for i in range(len(sent) + 1)]
    pi[0] = {('*', '*'): [1, '']}

    q_probs = calc_q_dict(sent)
    for i in range(len(sent)):
        for v in get_tags(i):
            for u in get_tags(i - 1):
                pi[i + 1][(u, v)] = [-np.inf, 'O']
                for t in get_tags(i - 2):
                    cur_prob = pi[i][(t, u)][0] * q_probs[(i,t, u)][tag_to_idx_dict[v]]
                    if cur_prob > pi[i + 1][(u, v)][0]:
                        pi[i + 1][(u, v)][0] = cur_prob
                        pi[i + 1][(u, v)][1] = t

    u, v = max(pi[-1].items(), key=lambda x: x[1])[0]

    if len(sent) == 1:
        predicted_tags[-1] = v
    else:
        predicted_tags[-1] = v
        predicted_tags[-2] = u
        for i in reversed(range(len(predicted_tags) - 2)):
            u = predicted_tags[i + 1]
            v = predicted_tags[i + 2]
            predicted_tags[i] = pi[i + 3][(u, v)][1]
    ### END YOUR CODE
    return predicted_tags

def memm_eval(test_data, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
    Receives: test data set and the parameters learned by memm
    Returns an evaluation of the accuracy of Viterbi & greedy memm
    """
    acc_viterbi, acc_greedy = 0.0, 0.0
    eval_start_timer = time.time()

    correct_greedy_preds = 0
    correct_viterbi_preds = 0
    total_words_count = 0

    gold_tag_seqs = []
    greedy_pred_tag_seqs = []
    viterbi_pred_tag_seqs = []
    for sent in test_data:
        words, true_tags = zip(*sent)
        gold_tag_seqs.append(true_tags)

        ### YOUR CODE HERE
        greedy_pred_tag_seqs.append(memm_greedy(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments))
        viterbi_pred_tag_seqs.append(memm_viterbi(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments))
        ### END YOUR CODE

    greedy_evaluation = evaluate_ner(gold_tag_seqs, greedy_pred_tag_seqs)
    viterbi_evaluation = evaluate_ner(gold_tag_seqs, viterbi_pred_tag_seqs)

    return greedy_evaluation, viterbi_evaluation

def build_tag_to_idx_dict(train_sentences):
    curr_tag_index = 0
    tag_to_idx_dict = {}
    for train_sent in train_sentences:
        for token in train_sent:
            tag = token[1]
            if tag not in tag_to_idx_dict:
                tag_to_idx_dict[tag] = curr_tag_index
                curr_tag_index += 1

    tag_to_idx_dict['*'] = curr_tag_index
    return tag_to_idx_dict


if __name__ == "__main__":
    full_flow_start = time.time()
    train_sents = read_conll_ner_file("data/train.conll")
    dev_sents = read_conll_ner_file("data/dev.conll")

    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    extra_decoding_arguments = build_extra_decoding_arguments(train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)
    tag_to_idx_dict = build_tag_to_idx_dict(train_sents)
    index_to_tag_dict = invert_dict(tag_to_idx_dict)

    vec = DictVectorizer()
    print("Create train examples")
    train_examples, train_labels = create_examples(train_sents, tag_to_idx_dict)


    num_train_examples = len(train_examples)
    print("#example: " + str(num_train_examples))
    print("Done")

    print("Create dev examples")
    dev_examples, dev_labels = create_examples(dev_sents, tag_to_idx_dict)
    num_dev_examples = len(dev_examples)
    print("#example: " + str(num_dev_examples))
    print("Done")

    all_examples = train_examples
    all_examples.extend(dev_examples)

    print("Vectorize examples")
    all_examples_vectorized = vec.fit_transform(all_examples)
    train_examples_vectorized = all_examples_vectorized[:num_train_examples]
    dev_examples_vectorized = all_examples_vectorized[num_train_examples:]
    print("Done")

    logreg = linear_model.LogisticRegression(
        multi_class='multinomial', max_iter=128, solver='lbfgs', C=100000, verbose=1)
    print("Fitting...")
    start = time.time()
    logreg.fit(train_examples_vectorized, train_labels)
    end = time.time()
    print("End training, elapsed " + str(end - start) + " seconds")
    # End of log linear model training

    # Evaluation code - do not make any changes
    start = time.time()
    print("Start evaluation on dev set")

    memm_eval(dev_sents, logreg, vec, index_to_tag_dict, extra_decoding_arguments)
    end = time.time()

    print("Evaluation on dev set elapsed: " + str(end - start) + " seconds")
