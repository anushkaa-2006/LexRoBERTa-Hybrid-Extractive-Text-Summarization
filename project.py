import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import LancasterStemmer,RegexpStemmer,PorterStemmer,SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder,TrigramCollocationFinder,ngrams
from sklearn.feature_extraction.text import CountVectorizer
from nltk.wsd import lesk

#data1="NLP stands for Natural Language Processing. And now I'm going to use . To learn something new here"
#data2="I am Anushkaa"," I am studying in GPS"
data3="The mouse nibbled on a piece of cheese in the corner of the room"
data4="She clicked the mouse to select the file on her computer"

# w=nltk.word_tokenize(data1)
# print(w)
# print("\n")
# p=nltk.pos_tag(w)
# print (p)


# sent=nltk.sent_tokenize(data)
# for i in sent:
#     print(i)
# print()

# word=nltk.word_tokenize(data)
# for i in word:
#     print(i)
# print()

# a=list()
# stop_words=stopwords.words("english")+list(punctuation)
# for i in w:
#     if i not in stop_words:
#         a.append(i)
# # print()


# l=LancasterStemmer()
# r=RegexpStemmer('ed')
# p=PorterStemmer()
# s=SnowballStemmer('english')
# print (l.stem("changed"))
# print (r.stem("changed"))
# print (p.stem("changed"))
# print (s.stem("changed"))
# print()


# wl=WordNetLemmatizer()
# b=list()
# for i in a:
#     b.append(wl.lemmatize(i))



# words=nltk.word_tokenize(data2)
# b=BigramCollocationFinder.from_words(words)
# t=TrigramCollocationFinder.from_words(words)
# n=ngrams(words,5)
# print(b.ngram_fd.keys())
# print(list(t.ngram_fd))
# for i in n:
#     print(i)

# vect=CountVectorizer()
# count=vect.fit_transform(b)    #convert text into numerical forms 
# print(count.toarray())


l=lesk(nltk.word_tokenize(data3),"mouse")  #word sense disambigution to get the meaning of word
print (l.definition())
