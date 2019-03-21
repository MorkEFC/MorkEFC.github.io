import nltk, collections
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('gutenberg')

# # Task 1 (1 mark)
import collections
def get_top_tokens(text_collection, n, stopwords):
    """Return a list of the n most frequent non-stop tokens, sorted by frequency.
    Make sure that the list of tokens returned is lowercased, and that all
    comparison with the list of stop words are not case sensitive.
    >>> get_top_tokens(gutenberg_collection, 10, nltk_stopwords)
    ['shall', 'said', "'s", 'unto', 'lord', 'thou', 'one', 'thy', 'man', 'god']
    >>> my_collection = ["This is sentence 1", "This is sentence 2", "And sentence 3"]
    >>> get_top_tokens(my_collection, 3, [])
    ['sentence', 'this', 'is']

    """
    wordcounter = collections.Counter([w.lower() for a in text_collection for s in nltk.sent_tokenize(a) for w in nltk.word_tokenize(s)])                                         
    list = []
    for word in wordcounter.most_common():
        if word[0] not in stopwords:
            list.append(word[0])  
        if len(list) == n:
            return list   
    return list

#Task 2 (1 mark)
def get_tf(text, template):
    """Return the frequency of each of the tokens listed in the template.
    Make sure that the comparison with the words in the template is not case
    sensitive.

    >>> get_tf("This is sentence 1. This is sentence 2. And sentence 3.", ['this', 'sentence'])
    [2, 3]
    >>> get_tf(gutenberg_collection[0], ['emma', 'my', 'the'])
    [855, 728, 5201]
    """
    wordcounter = collections.Counter([w.lower() for s in nltk.sent_tokenize(text)
                                                 for w in nltk.word_tokenize(s)])
    list = []
    for word in template:
        word = word.lower()
        for item in wordcounter.most_common():
            if item[0] == word:
                list.append(item[1])
    return list

#Task 3 (1 mark)
from math import log
def get_idf(text_collection, template):
    """Return a list of inverse document frequencies for every token listed in the
    template, where each element in text_collection represents one document.
    Again, make sure that the comparisons are not case sensitive. The inverse
    document frequency is computed by the formula indicated in the lectures,
    where the base of log is 10:

                        number of documents
    idf(t) = log(-----------------------------------)
                 number of documents that contain t

    >>> get_idf(gutenberg_collection, ['emma', 'my', 'sam'])
    [0.9542425094393249, 0.0, 1.255272505103306]
    >>> get_idf(gutenberg_collection, ['unto', 'lord', 'thou'])
    [0.47712125471966244, 0.07918124604762482, 0.1413291527964693]
    """
    list = []
    for word in template:
        word = word.lower()
        count = 0
        for text in text_collection:
            words = [words.lower() for words in word_tokenize(text)]
            if word in words:
                count+=1
        if count != 0:       
            list.append(log(len(text_collection)/count, 10))
        else:
            list.append(0)
    return list

# Task 4 (1 mark)
def get_tfidf(text_collection, list_documents, template):
    """Return the tf.idf of each document of the list of documents, where the idf
    is computed relative to the text collection. The tf.idf values should be
    computed based on the words of the template. Again, make sure that all
    comparisons are not case sensitive.
    >>> get_tfidf(gutenberg_collection, gutenberg_collection[:2], ['unto', 'lord', 'thou'])
    [[0.0, 0.4750874762857489, 0.1413291527964693], [0.0, 0.7126312144286233, 0.0]]
    """
    #calculate tf
    tf = {}
    for text in list_documents:
        words = [words.lower() for words in word_tokenize(text)]
        wordCounter = collections.Counter(words)
        temp = {}
        for word in template:
            flag = False
            for item in wordCounter:
                if item == word:
                    temp[word]= wordCounter[item]
                    flag = True
            if flag == False:
                temp[word] = 0
        tf[text] = temp
    idfList = {}
    for word in template:
        word = word.lower()
        count = 0
        for text in text_collection:
            words = [words.lower() for words in word_tokenize(text)]
            if word in words:
                count+=1
        if count != 0:       
            idfList[word] = log(len(text_collection)/count, 10)
        else:
            idfList[word] = 0
    tfidf = []
    for text in list_documents:
        tempList = []
        for word in template:
            tempList.append(tf.get(text).get(word)*idfList.get(word))
        tfidf.append(tempList)
    return tfidf

# Task 5 (1 mark)
from math import sqrt
def cosine_similarity(text_collection, text1, text2, template):
    """Return the cosine similarity between the tfidf of text1 and that of text2
    where the cosine similarity is defined with the formula given in the
    lectures:

                                sum_i(text1_i*text2_i)
    cos(text1, text2) = ----------------------------------------------
                        sqrt(sum_i(text1_i^2)) sqrt(sum_i(text2_i^2))

    You can implement the cosine similarity directly, or you can use a library such as sklearn:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
    >>> cosine_similarity(gutenberg_collection, gutenberg_collection[0], gutenberg_collection[0], ['unto', 'lord', 'thou'])
    1.0
    >>> cosine_similarity(gutenberg_collection, gutenberg_collection[0], gutenberg_collection[1], ['unto', 'lord', 'thou'])
    0.9584884365371023
    """
    return 2.0

# DO NOT MODIFY THE CODE BELOW
if __name__ == "__main__":
    import doctest
    gutenberg_collection = [nltk.corpus.gutenberg.raw(d) for d in nltk.corpus.gutenberg.fileids()]
    nltk_stopwords = nltk.corpus.stopwords.words('english') + [',', '.', ';', "''", ':', '``', '?', '--', '!']
    doctest.testmod()
