import numpy as np
import lda
from sklearn.feature_extraction.text import CountVectorizer

def bagOfWords(texts, documents):
   context = 1
   textList = []
   for text in texts:
       textList.append(' '.join(documents[text]))
   vectorizer = CountVectorizer(analyzer = "word",
                                tokenizer = None,
                                preprocessor = None,
                                stop_words = None,
                                ngram_range= (context, context),
                                max_features = None)
   tdf = vectorizer.fit_transform(textList)
   train_data_features = tdf.toarray()
   featureNames = vectorizer.get_feature_names()
   return train_data_features, featureNames


def ldaModel(texts,topics,iters, nWords, documents):

    dtm, vocab = bagOfWords(texts, documents)

    model = lda.LDA(n_topics=topics, n_iter=iters, random_state=1)
    model.fit(dtm)
    topic_word = model.topic_word_

    n_top_words = nWords
    topic_words = {}
    for i, topic_dist in enumerate(topic_word):
        topic_words[i] = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words[i])))
    doc_topic = model.doc_topic_

    for i in range(len(texts)):
        print("{} (top topic: {})".format(texts[i], doc_topic[i].argmax()))

    probs = np.array(doc_topic)
    meanProbs = np.mean(probs, axis=0)

    return topic_words, meanProbs, probs, vocab, dtm
