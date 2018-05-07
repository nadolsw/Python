#First 'pip install tweepy' into windows command prompt (NOT PYTHON IDE)#
#First 'pip install textblob' into windows command prompt (NOT PYTHON IDE)#

import tweepy
from textblob import TextBlob #Case-Sensitive!#

test = TextBlob("Siraj is angry that he never gewts good matches on Tinder")

print test.tags #See tags from corpus#
print test.words #Tokenize words
print test.sentiment #Quantify sentiment


consumer_key = 'msApQLMvebNMaMa9iQ6gO5Vmb'
consumer_secret = 'VdW7skYwbi5m7QFuda0REWxERemqP3fUvaNrfK7HTn4xpaL8dl'

acces_token = '87503603-vYwdY0fiyJEeLYr4CdvvJ9UZj9yKCOMzOIu7qbTbI'
access_token_secret = '4Y0MqLbKceAcyr0AZWfTDOgoiMFPMNqnbqnWOmddliTxl'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(acces_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Trump')

for tweet in public_tweets:
	print(tweet.text)
	sentiment_analysis = TextBlob(tweet.text)
	print(sentiment_analysis.sentiment)