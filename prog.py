##################################################################
#                Python code for RevRank algorithm
#                
##################################################################

import nltk
import string
import math
import time
import itertools
import operator
import numpy as np
import pandas as pd
import re
from nltk.corpus import brown, stopwords,movie_reviews
from nltk.tokenize import sent_tokenize,word_tokenize
import gensim
import newlinejson as nlj
import json
from operator import itemgetter

# ALL THE FILES USED TO SEGRAGATING THE CHUNK OF CODE FROM BIG BOOK5.JSON FILE
i=0
b1='Books_5.json'
b2='mam.json'
b3='out.json'
b4='out_final.json'
b5='out_analysis.json'


#AllReviews = pd.read_csv('data/book_sample_counts.csv.gz')


# FINDING COSIN SIMILARITY BETWEEN TWO DICT GIVEN AS ARGUMENTS
def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator


###### TO FIND DOMINANT WORDS FOR OUR VIRTUAL CORE REVIEW (MAKING VOCABULARY) #######
def virtual_core(reviewsDf, corpus_freq,c,top_words):

	Words =[nltk.word_tokenize(text) for text in reviewsDf.reviewText]
	tokens = list(itertools.chain(*Words))
	Final_words = [x for x in tokens if not re.fullmatch('[' + string.punctuation + ']+', x)]  #regular expression matching
	
	dominance = dict([])

	word_frequency = nltk.FreqDist(Final_words)

	for word in dict(word_frequency):
		if (corpus_freq[word] > 0) and (math.log(corpus_freq[word],2) > 0) and (word.lower() not in stopwords.words('english')) and (len(word) > 2):
			dominance[word] = math.log(word_frequency[word],2) * c * ( 1/math.log(corpus_freq[word],2) )

	return(sorted(dominance.items(), key=operator.itemgetter(1), reverse=True)[:top_words])

'''
BELOW THERE ARE THREE REVIEW FUNCTION ... COMMENT EACH AS PER REQUIRED

'''
######## REVIEW SCORE FUNCTION USING THE MODIFIED TFIDF USED IN THE PAPER #############

# def review_score(text, core, mean):
#     core_dict = dict(core)
#     review_vector = nltk.FreqDist(nltk.word_tokenize(text))
#     length = len(nltk.word_tokenize(text))
#     score = 0

#     for word in review_vector:
#         if word in core_dict:
#             score += review_vector[word]

#     p = 1
#     if length < mean:
#         p = 20
            
#     return (float(1) / p) * (float(score) / length)

############## REVIEW SCORE FUNCTION USING THE COSINE SIMILARITY BETWEEN EACH TEXT CONTENT AND VIRTUAL CORE REVIEW #########
def review_score(text, core, mean):
    #core_dict = dict(core)
    review_vector = nltk.FreqDist(nltk.word_tokenize(text))
    score = 0

    cosine = get_cosine(dict(core), review_vector)
    return cosine     

############ REVIEW SCORE FUNCTION USING WORD2VEC #############
'''
HERE WE TAKE A AVERAGE OF EACH WORD IN THE TEXT OF REVIEW WITH RESPECT TO ALL WORDS IN WORDS IN VIRTUAL CORE REVIEW
'''

# def review_score(text, core, mean):
# 	core_dict = dict(core)
# 	print(core_dict)
# 	word_arr=nltk.word_tokenize(text)
# 	review_vector = nltk.FreqDist(word_arr)
# 	brown_corpus=brown.words()
# 	model = gensim.models.Word2Vec(brown.sents(), min_count=1)
# 	length = len(core)
# 	print(length)
# 	score = 0

# 	dominance = dict([])
# 	for word in review_vector:
# 		score=0
# 		for vritual_word in core:
# 			if word in brown_corpus:
# 				print(word)
# 				arr=model.most_similar(word,vritual_word)

# 				score+=arr
# 			else:
# 				score+=0
# 		avg_score[word]=score/length


'''
IMPORT DATA FROM OUT.JSON FILE
'''
print("STARTING LOADING")
with nlj.open(b3) as src:
    with nlj.open('out_final.json', 'w') as dst:
        for line in src:
            #print(i)
            if(line['asin']=="0007444117"):       # USE THE ASIN NUMBER FOR DIFFERENT BOOKS
                #print('came')
                dst.write(line)

print("LOADING COMPLETED !!!")
df=pd.concat([pd.Series(json.loads(line)) for line in open(b4)], axis=1)

df=df.transpose()

helpful_rt=list(df['helpful'])


'''
BELOW PART REMOVES 10 REVIEWS BASED ON THE HELPFULNESS RATIO GIVEN BY AMAZON
'''
stuff=[]
for review_helpful in helpful_rt:
	if(review_helpful[1]==0):
		stuff.append(0)
	else:
		stuff.append(review_helpful[0]/review_helpful[1])


sorted_index =sorted(range(len(stuff)), key=lambda k: stuff[k],reverse=True)
#print(sorted_index)
top_index=[]
f=open("out.txt","w+")
for i in sorted_index:
	if(helpful_rt[i][1]>=18):
		f.write(str(stuff[i])+" ")
		f.write('heere'+str(helpful_rt[i][0])+" "+str(helpful_rt[i][1])+"\n")
		top_index.append(i)
del top_index[1]
del top_index[3]
del top_index[3]
del top_index[8]
review_text_rate=open("text_of_review.txt","w+")
json_analysis=nlj.open(b5,"w")
final_analysis=[]
#print('haa')
for i in range(10):
	review_text_rate.write(str(i+1)+"--\n")
	review_text_rate.write(df['reviewText'][top_index[i]]+"\n\n")
	final_analysis.append(top_index[i])
	#json_analysis.write(df[top_index[i]])

#####   DATAFRAME FOR THOSE 10 REVIEWS #############

df_ana = pd.DataFrame(data=df, index=final_analysis)




ProductReview = df
c=3
top_words=50

word_count=[]
for t in ProductReview.T.to_dict().values():
	text=t['reviewText']
	Words =nltk.word_tokenize(text)
	tokens = list(itertools.chain(Words))
	Final_words = [x for x in tokens if not re.fullmatch('[' + string.punctuation + ']+', x)]  #regular expression matching
	word_count.append(len(Final_words))
mean=np.array(word_count).mean()

#### THIS BELOW PART IS USING SOMETHNG CALLED BROWN CORPUS(SOME RANDOM TEXT) FOR FINDING THE FREQUENCY (ONCE HAVE A LOOK NTO OUR PAPER). CHANGE THIS TO WIKI PAGE PLOT CONTENT

corpus_freq = nltk.FreqDist(brown.words(categories=['science_fiction', 'romance', 'reviews','adventure']))
#corpus_freq = nltk.FreqDist(brown.words(categories='reviews'))

#### THE PART BELOW IS THE WIKI PLOT CORPUS USED FOR THE VIRTUAL CORE REVIEW

plot=open("alligiant_wikiplot.txt",'r+')
text_plot=plot.read()

Words =nltk.word_tokenize(text_plot)

#tokens = list(itertools.chain(*Words))
Final_words = [x for x in Words if not re.fullmatch('[' + string.punctuation + ']+', x)]  #regular expression matching

Final_plot_words=[x for x in Words if x not in stopwords.words('english')]
corpus_freq_plot=nltk.FreqDist(Final_plot_words)

print("The Virtual Core Review using brown corpus\n\n")
virtual_review_core=virtual_core(ProductReview,corpus_freq,c,top_words)
print(virtual_review_core)

print("\n\nThe Virtual Core Review using plot\n\n")

virtual_review_core_plot=virtual_core(ProductReview,corpus_freq_plot,c,top_words)
print(dict(virtual_review_core_plot))

########## BELOW ARE THE FINAL SCORE AND ARRNAGED IN ASCENDING ORDER #######
	
tup=[]
final_score=[]
ij=1
for index, row in df_ana.iterrows():

	review_s = review_score(row['reviewText'], virtual_review_core_plot, mean)
	tup=[]
	tup.append(ij)
	tup.append(review_s)
	final_score.append(tup)
	ij+=1

# Final_helpful_reviews =sorted(final_score, key=itemgetter(1))
# for x in Final_helpful_reviews:
# 	print(x[0])

print(Final_helpful_reviews)



