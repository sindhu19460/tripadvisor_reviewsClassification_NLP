#  Tripadvisor_Reviews_Classification_using_NLP

Data Science intern.

# Business Object

Natural Language processing on reviews of products in Amazon: Analyze the reviews posted by the customers and build an algorithm to find the emotions (Positive, Negative etc.,)

# Data 
Reviews to be extracted from Tripadvisor for specific Hotel.

I used the lib to extract the Reviews From TripAdviosre it is main challenge i faced before.
sys, csv ,selenium import webdriver.

# Exploratory Data Analysis:
Apply various text mining methods and generate Insights from the text data.
I used technicise as Below.
1 nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
2 FreqDist
3 stopwords
4 Wordcloud
5 wordnet
6 n-gram,bi-gram,tri-gram
7 stemming
8 Tokenization
9 Sentiment Analysis
10 Vader
11 Textblob

# Model Building Part
I used several alogrithm (around 11) But i got a Good accuracy in Random Forest so i used random Forest Classifier for deployment Part.
https://github.com/sindhu19460/tripadvisor_reviewsClassification_NLP/blob/main/Model_Building.ipynb

First i checked data is balanced or not ,when i got knew data is imbalance , first i balance and i moved on to the furture.

# Deployment Part
deployed using Flask

# Thank You.
