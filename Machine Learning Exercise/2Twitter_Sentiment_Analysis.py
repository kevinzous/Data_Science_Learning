#Twitter Sentiment Analysis
#Relevee de message; tokenization (separer les msg en mots et ); Lexicon-based approach: sentiment lexicon of each word)
# 1. using a vocabulary: looking at important keywords (usually verbs and adjectives) along with modifiers like negation words

# 2. using rules: look at presence of vocabulary words in sentences, use rules to categorise them by sentiment

# 2. applying ML techniques: treat this as a classification problem, amass a dataset and set up features (which could be the keywords in the vocabulary), and train to identify sentiment

import tweepy
from textblob import TextBlob

consumer_key = 'WkZ01Vd6CPhAQ3tZZaGIhsFYS'
consumer_secrets = 'fPemdV0YQq0Eqj0H6tjuCWpNeqM02Wg4s4ZMkhmOO5z7gtp7k4'

access_token = '220640244-VaX0bN7bsI6BJ2dLDusdYOv6tJsUfE4gmDcolS0x'
access_token_secret = 'QIKAeYAgLYo2gbJjUymIHeYelGktlW4IKlEXoot1hZK5n'

auth = tweepy.OAuthHandler(consumer_key,consumer_secrets)
auth.set_access_token(access_token,access_token_secret)
api= tweepy.API(auth)
public_tweets = api.search('Trump')
Tweets_Neg=[]
Tweets_Pos=[]

for tweet in public_tweets:
	#print(tweet.text)

	analysis=TextBlob(tweet.text)
	polarity=analysis.sentiment.polarity
	print(polarity)
	if polarity>0:
		Tweets_Pos=Tweets_Pos+[tweet.text,polarity]
	else: 
		Tweets_Neg=Tweets_Neg+[tweet.text,polarity]

print(Tweets_Neg)
print(Tweets_Pos)
