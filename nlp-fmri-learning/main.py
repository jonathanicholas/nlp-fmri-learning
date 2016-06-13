import os
from processing import getDocuments
from unsupervised import ldaModel
from scipy.stats import ttest_rel

def loadTexts():
    path = '../texts/math_learning_2letters_full/'
    texts = sorted([os.path.join(path, fn) for fn in os.listdir(path)])
    for text in texts:
        if '.DS_Store' in text:
            texts.remove(text)
    delimiter = ' '
    return texts, getDocuments(texts, delimiter)

def runLDA(texts, documents, nTopics, nWords, nIters):
    ldaModel(texts,nTopics,nIters,nWords,documents)

def main():
    textNames, documents = loadTexts()

    nTopics = 20
    nWords = 10
    nIters = 1000
    topics, topicProbs, gammas, featureNames, dtm  = runLDA(textNames, documents, nTopics, nWords, nIters)

    y1_gammas = gammas[0:35]
    print y1_gammas
    y2_gammas = gammas[35:70]
    print y2_gammas

    t, p = ttest_rel(y1_gammas,y2_gammas)
    print p


if __name__ == '__main__':
    main()
