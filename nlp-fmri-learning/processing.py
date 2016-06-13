'''
Returns document as dict: documents[document] = list of words
'''

def getDocuments(texts, delimiter):
    documents = {}
    for text in texts:
        index = 0
        words = []
        doc = open(text, 'r')
        for word in doc.read().split(delimiter):
            index += 1
            word = word
            if word:
                words.append(word)

        documents[text] = words

    return documents
