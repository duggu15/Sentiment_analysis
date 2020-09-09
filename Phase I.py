import nltk
import re
from collections import Counter
import math
from nltk.corpus import stopwords
import random
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

try:
    pos_tweets = open("short_reviews/positiveless.txt","r", encoding="utf8").read()
except UnicodeDecodeError:
    pos_tweets = open("short_reviews/positiveless.txt","r", encoding="latin-1").read()
           
try:
    neg_tweets = open("short_reviews/negativeless.txt","r", encoding="utf8").read()
except UnicodeDecodeError:
    neg_tweets = open("short_reviews/negativeless.txt","r", encoding="latin-1").read()
           
test_tweet = [("this is horrible day"),
              ("I love the book"),
              ("I feel happy this morning"),
              ("Larry is my friend. "),
              ("I do not like that man."),
              ("My house is not great."),
              ("Your song is annoying. ")]

#start process_tweet
def processTweet(tweet):
    # process the tweets
    
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)    
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end 


 
tweets = []
pos_words=[]
neg_words=[]
all_words = []
accuricies_classifier=[]


## remove stop words
stopset = set(stopwords.words('english'))
def stopword_filtered_word_feats(words):
    return dict([(word, True) for word in words if word not in stopset])

for (words) in neg_tweets.split('\n'):
    words=processTweet(words)
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
    words_filtered=stopword_filtered_word_feats(words_filtered)
    tweets.append((words_filtered, "negative"))
    
for (words) in pos_tweets.split('\n'):
    words=processTweet(words)
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
    words_filtered=stopword_filtered_word_feats(words_filtered)
    tweets.append((words_filtered, "positive"))


###########  NAIVE BAYES   ##########

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0


def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

    
splitRatio = 0.67
training_set1, testing_set1 = splitDataset(tweets, splitRatio)

for words in training_set1:
    if(words[-1]=="positive"):
        pos_words.extend(words[0])
    else:
        neg_words.extend(words[0])
    all_words.extend(words[0])

pos_prob=len(pos_tweets)/(len(pos_tweets)+len(neg_tweets))
neg_prob=len(neg_tweets)/(len(pos_tweets)+len(neg_tweets))

positive_wordlist = Counter(pos_words)
wordlist = Counter(all_words)
negative_wordlist = Counter(neg_words)

v=len(set(all_words))

predictions = []
for test in testing_set1:
    pos=math.log(pos_prob)
    neg=math.log(neg_prob)        

    for word in test[0].keys():
            pos=pos+math.log((positive_wordlist[word]+1)/(len(pos_words)+v))
            neg=neg+math.log((negative_wordlist[word]+1)/(len(neg_words)+v))
                                              
    #print("pos : ",pos)
    #print("neg : ",neg)
    
    if(pos>neg):
            #print(test, " : Positive")
            predictions.append("positive")
    else:
            #print(test," : Negative")
            predictions.append("negative")

accuracy = getAccuracy(testing_set1, predictions)
print('Accuracy: ',accuracy)
accuricies_classifier.append(accuracy)
#######################################
def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

word_features = get_word_features(get_words_in_tweets(tweets))

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

dataset = nltk.classify.apply_features(extract_features, tweets)
def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

    
splitRatio = 0.67
training_set, testing_set = splitDataset(dataset, splitRatio)
#print(training_set)
	
NaiveBayes_classifier = nltk.NaiveBayesClassifier.train(training_set)
accuracy=(nltk.classify.accuracy(NaiveBayes_classifier, testing_set))*100
print("Original Naive Bayes Algo accuracy percent:", accuracy)
accuricies_classifier.append(accuracy)

#classifier.show_most_informative_features(15)
tweet = 'The Movie was fantastic'
print (tweet," : ",NaiveBayes_classifier.classify(extract_features(tweet.split())))                                              



MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
accuracy=(nltk.classify.accuracy(MNB_classifier, testing_set))*100
print("MNB_classifier accuracy percent:",accuracy)
accuricies_classifier.append(accuracy)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
accuracy=(nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100
print("BernoulliNB_classifier accuracy percent:",  accuracy)
accuricies_classifier.append(accuracy)


DecisionTree_classifier = SklearnClassifier(DecisionTreeClassifier())
DecisionTree_classifier.train(training_set)
accuracy=(nltk.classify.accuracy(DecisionTree_classifier, testing_set))*100
print("DecisionTree_classifier accuracy percent:", accuracy)
accuricies_classifier.append(accuracy)


objects = ('OWN', 'NB','MNB', 'BNB', 'DTC')
y_pos = np.arange(len(objects))
 
plt.bar(y_pos, accuricies_classifier, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Classifiers')
 
plt.show()


