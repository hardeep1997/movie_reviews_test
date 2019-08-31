# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 10:13:11 2019

@author: lion
"""
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

def word_feats(words):
    return dict([ (word, True) for word in words])

print(word_feats(['this', 'moview','is', 'Awesome']))

posids = movie_reviews.fileids('pos')
negids = movie_reviews.fileids('neg')


negfeats = [ (word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [ (word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

print(negfeats[5], posfeats[5])

negcutoff = int(len(negfeats)*3/4)
poscutoff = int(len(posfeats)*3/4)
print(negcutoff, poscutoff)

trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
print('Train on %d instances, test on %d instances ' % (len(trainfeats) , len(testfeats)))

classifier = NaiveBayesClassifier.train(trainfeats)
print('Accuracy: ', nltk.classify.util.accuracy(classifier,testfeats)*100)
classifier.show_most_informative_features(50)


'''  Testing our own movie reviews '''

test_data_features = word_feats(movie_reviews.words(fileids = '../test/kabir_neg_reviews.txt'))
print('****Reviews:')
print(classifier.classify(test_data_features))

test_data_features2 = word_feats(movie_reviews.words(fileids = '../test/kabir_pos_reviews.txt'))
print(classifier.classify(test_data_features2))
