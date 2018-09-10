# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 17:22:35 2017

@author: Alison
"""
from __future__ import division
import numpy as np
import pandas as pd
from sklearn import svm
import json
from nltk.corpus import stopwords
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression 
from sklearn.neural_network import MLPClassifier 
from sklearn.preprocessing import Normalizer
from sklearn import dummy
#from __future__ import print_function
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from gensim.models import Doc2Vec
import gensim.models.doc2vec
import multiprocessing
#%matplotlib inline
import matplotlib.pyplot as plt

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"


tokenizer = RegexpTokenizer(r'\w+')

from gensim.models import doc2vec
from collections import namedtuple



    
def enumeratemod(sequence, start=0):
    n = start
    for elem in sequence:
        yield n, elem
        n += 2
    
def encode_target(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """


    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumeratemod(targets,start=-1)}
    df_mod["new_column"] = df_mod[target_column].replace(map_to_int)
    #if df_mod["new_column"] == 0: 
       #df_mod["new_column"] = df_mod["new_column"].replace(1)
    return (df_mod, targets)
    

new_labels3,pf = encode_target(labels3, "Outcome")
new_labels6,pf2 = encode_target(labels6, "Outcome")
new_labels8,pf3 = encode_target(labels8, "Outcome")  
from sklearn.model_selection import cross_val_score

#CLASSIFIERS 
def perform_svm (X,labels):
    clf = svm.SVC(kernel='linear')
    scores = cross_val_score(clf,X,labels, cv=10)
    return scores
    
def perform_lin_svm(X,labels):
    clf=svm.LinearSVC()
    scores = cross_val_score(clf,X,labels, cv=10)
    return scores
    

def perform_naive_b(X,labels):
    clf=MultinomialNB()
    scores = cross_val_score(clf,X,labels, cv=10)
    return scores
    
    
def perform_log(X,labels):
    clf=LogisticRegression ()
    scores = cross_val_score(clf,X,labels, cv=10)
    return scores
    

def perform_mlp(X,labels):
    clf=MLPClassifier (solver = 'lbfgs', random_state=1, alpha= 1e-5, hidden_layer_sizes=(5,2))
    scores = cross_val_score(clf,X,labels, cv=10)
    return scores
    
    
 def perform_randomforest(X,labels):
    clf=ensemble.RandomForestClassifier()
    scores = cross_val_score(clf,X,labels, cv=10)
    return scores
    
#----------------------------------------------------------
#Importing files and computing scores for the reference study 
       
a3_full=pd.read_csv("/Users/Alison/Downloads/echr_dataset/ngrams_a3_full.csv",header=None)
a3_featuresNames=pd.read_csv("/Users/Alison/Downloads/echr_dataset/ngrams_a3_featureNames.txt")
a3_law=pd.read_csv("/Users/Alison/Downloads/echr_dataset/ngrams_a3_law.csv",header=None)
a3_procedure=pd.read_csv("/Users/Alison/Downloads/echr_dataset/ngrams_a3_procedure.csv",header=None)
a3_relevantLaw=pd.read_csv("/Users/Alison/Downloads/echr_dataset/ngrams_a3_relevantLaw.csv",header=None)
labels3=pd.read_csv("/Users/Alison/Downloads/echr_dataset/cases_a3.csv")
a3_topics=pd.read_csv("/Users/Alison/Downloads/echr_dataset/topics3.csv",header=None,sep='\t')
a3_circumstances=pd.read_csv("/Users/Alison/Downloads/echr_dataset/ngrams_a3_circumstances.csv",header=None)


a6_full=pd.read_csv("/Users/Alison/Downloads/echr_dataset/Article6/ngrams_a6_full.csv",header=None)
a6_featureNames=pd.read_csv("/Users/Alison/Downloads/echr_dataset/Article6/ngrams_a6_featureNames.txt")
a6_law=pd.read_csv("/Users/Alison/Downloads/echr_dataset/Article6/ngrams_a6_law.csv",header=None)
a6_procedure=pd.read_csv("/Users/Alison/Downloads/echr_dataset/Article6/ngrams_a6_procedure.csv",header=None)
a6_relevantLaw=pd.read_csv("/Users/Alison/Downloads/echr_dataset/Article6/ngrams_a6_relevantLaw.csv",header=None)
labels6=pd.read_csv("/Users/Alison/Downloads/echr_dataset/Article6/cases_a6.csv")
a6_topics=pd.read_csv("/Users/Alison/Downloads/echr_dataset/Article6/topics6.csv",header=None,sep='\t')
a6_circumstances=pd.read_csv("/Users/Alison/Downloads/echr_dataset/Article6/ngrams_a6_circumstances.csv",header=None)


a8_full=pd.read_csv("/Users/Alison/Downloads/echr_dataset/Article8/ngrams_a8_full.csv",header=None)
a8_featureNames=pd.read_csv("/Users/Alison/Downloads/echr_dataset/Article8/ngrams_a8_featureNames.txt")
a8_law=pd.read_csv("/Users/Alison/Downloads/echr_dataset/Article8/ngrams_a8_laws.csv",header=None)
a8_procedure=pd.read_csv("/Users/Alison/Downloads/echr_dataset/Article8/ngrams_a8_procedure.csv",header=None)
a8_relevantLaw=pd.read_csv("/Users/Alison/Downloads/echr_dataset/Article8/ngrams_a8_relevantLaw.csv",header=None)
labels8=pd.read_csv("/Users/Alison/Downloads/echr_dataset/Article8/cases_a8.csv")
a8_topics=pd.read_csv("/Users/Alison/Downloads/echr_dataset/Article8/topics8.csv",header=None,sep='\t')
a8_circumstances=pd.read_csv("/Users/Alison/Downloads/echr_dataset/Article8/ngrams_a8_circumstanes.csv",header=None)

scoresa3_full=perform_svm(a3_full,new_labels3["new_column"])
scoresa3_law=perform_svm(a3_law,new_labels3["new_column"])
scoresa3_procedure=perform_svm(a3_procedure,new_labels3["new_column"])
scoresa3_relevantLaw=perform_svm(a3_relevantLaw,new_labels3["new_column"])
scoresa3_topics=perform_svm(a3_topics,new_labels3["new_column"])
scoresa3_circumstances=perform_svm(a3_circumstances,new_labels3["new_column"])

print("Accuracy for full Article3 : %0.2f (+/- %0.2f)" % (scoresa3_full.mean(), scoresa3_full.std() ))
print("Accuracy for law section Article3: %0.2f (+/- %0.2f)" % (scoresa3_law.mean(), scoresa3_law.std() ))
print("Accuracy for procedure Article3: %0.2f (+/- %0.2f)" % (scoresa3_procedure.mean(), scoresa3_procedure.std() ))
print("Accuracy for releventLaw Article3: %0.2f (+/- %0.2f)" % (scoresa3_relevantLaw.mean(), scoresa3_relevantLaw.std() ))
print("Accuracy for topics Article3: %0.2f (+/- %0.2f)" % (scoresa3_topics.mean(), scoresa3_topics.std() ))
print("Accuracy for circumstances Article3: %0.2f (+/- %0.2f)" % (scoresa3_circumstances.mean(), scoresa3_circumstances.std() ))

scoresa6_full=perform_svm(a6_full,new_labels6["new_column"])
scoresa6_law=perform_svm(a6_law,new_labels6["new_column"])
scoresa6_procedure=perform_svm(a6_procedure,new_labels6["new_column"])
scoresa6_relevantLaw=perform_svm(a6_relevantLaw,new_labels6["new_column"])
scoresa6_topics=perform_svm(a6_topics,new_labels6["new_column"])
scoresa6_circumstances=perform_svm(a6_circumstances,new_labels6["new_column"])

print("Accuracy for full Article6: %0.2f (+/- %0.2f)" % (scoresa6_full.mean(), scoresa6_full.std() ))
print("Accuracy for law section Article6: %0.2f (+/- %0.2f)" % (scoresa6_law.mean(), scoresa6_law.std() ))
print("Accuracy for procedure Article6: %0.2f (+/- %0.2f)" % (scoresa6_procedure.mean(), scoresa6_procedure.std() ))
print("Accuracy for relevantLaw Article6: %0.2f (+/- %0.2f)" % (scoresa6_relevantLaw.mean(), scoresa6_relevantLaw.std() ))
print("Accuracy for topics Article6: %0.2f (+/- %0.2f)" % (scoresa6_topics.mean(), scoresa6_topics.std() ))
print("Accuracy for circumstances Article6: %0.2f (+/- %0.2f)" % (scoresa6_circumstances.mean(), scoresa6_circumstances.std() ))

scoresa8_full=perform_svm(a8_full,new_labels8["new_column"])
scoresa8_law=perform_svm(a8_law,new_labels8["new_column"])
scoresa8_procedure=perform_svm(a8_procedure,new_labels8["new_column"])
scoresa8_relevantLaw=perform_svm(a8_relevantLaw,new_labels8["new_column"])
scoresa8_topics=perform_svm(a8_topics,new_labels8["new_column"])
scoresa8_circumstances=perform_svm(a8_circumstances,new_labels8["new_column"])

print("Accuracy for full Article8: %0.2f (+/- %0.2f)" % (scoresa8_full.mean(), scoresa8_full.std() ))
print("Accuracy for law section Article8: %0.2f (+/- %0.2f)" % (scoresa8_law.mean(), scoresa8_law.std()))
print("Accuracy for procedure Article8: %0.2f (+/- %0.2f)" % (scoresa8_procedure.mean(), scoresa8_procedure.std() ))
print("Accuracy for relevantLaw Article8 : %0.2f (+/- %0.2f)" % (scoresa8_relevantLaw.mean(), scoresa8_relevantLaw.std() ))
print("Accuracy for topics Article8: %0.2f (+/- %0.2f)" % (scoresa8_topics.mean(), scoresa8_topics.std() ))
print("Accuracy for circumstances Article8: %0.2f (+/- %0.2f)" % (scoresa8_circumstances.mean(), scoresa8_circumstances.std() ))

#_____________________________________________________

#Opening my datafile and storing all data in "example" 
example=[]

with open('/Users/Alison/Downloads/echr_dataset/predictions.json') as data_file:    
    data = json.load(data_file)
    for k, v  in data.items():
        example.append(v)
    
articles_dataset=[]
for x in range(len(example)):
    articles_dataset.append(example[x]["articles"])


violations=[]  
for x in range(len(example)):
    violations.append(example[x]["decision_any_violation"])
    
country=[]
for x in range(len(example)):
    country.append(example[x]["country"])
        

#__________________________________________________

#Functions needed for CountVectorizer
def tokenize(string):
    #Convert string to lowercase and split into words (ignoring
    #punctuation), returning list of words.
    
    return re.findall(r'\w+', string.lower())

normalizer=Normalizer()
    
 #__________________________________________________
    
#Testing the model for all articles non balanced 
from sklearn.feature_extraction.text import CountVectorizer
kikoo=[]
for n in range (len(example)):
    thomaslove = example[n]["comm_text"][0:-1]
    kikoo.append(thomaslove)
vectorizer=CountVectorizer(tokenizer=tokenize, lowercase=True,ngram_range=(1, 4), max_features=2000, stop_words=stopwords.words('english'))
sauve=vectorizer.fit_transform(kikoo)
#len(vectorizer.get_feature_names())
normalizer=Normalizer()
sauve=normalizer.transform(sauve)
sauve.toarray()
nouvel_essai=perform_svm(sauve,violations)
print("Accuracy SVM Comm' all and no validation test: %0.2f (+/- %0.2f)" % (nouvel_essai.mean(), nouvel_essai.std() ))

dumdum= dummy.DummyClassifier(strategy='most_frequent')
scores_dummy = cross_val_score(dumdum,sauve, violations, cv=10)
print("Accuracy Dummy Comm' all and no validation test: : %0.2f (+/- %0.2f)" % (scores_dummy.mean(), scores_dummy.std() * 2))



#_________________________________________________________________________________

#Procedure to  make the dataset balanced and testing for Task 2with svm and dummy
violation_yes=[]
violation_no=[]
for n in range(len(example)):
   if violations[n]==0:
      violation_no.append(example[n])
   if violations[n]==1:
       violation_yes.append(example[n])
balanced_violation_yes=violation_yes[0:166]
balanced_dataset=balanced_violation_yes + violation_no
violations_balanced=["1"] * 166 + ["-1"] * 166

balanced_context=[]
for n in range (len(balanced_dataset)):
    thomaslove = balanced_dataset[n]["comm_text"][0:-1]
    balanced_context.append(thomaslove)
vectorizer=CountVectorizer(tokenizer=tokenize, lowercase=True,ngram_range=(1, 4), max_features=2000, stop_words=stopwords.words('english'))
balanced=vectorizer.fit_transform(balanced_context)
#len(vectorizer.get_feature_names())
normalizer=Normalizer()
balanced=normalizer.transform(balanced)
balanced.toarray()
score_balanced=perform_svm(balanced,violations_balanced)
print("Accuracy BoW SVM Comm' balanced and no validation test: %0.2f (+/- %0.2f)" %(score_balanced.mean(), score_balanced.std() ))
scores_bdummy = cross_val_score(dumdum,balanced, violations_balanced, cv=10)
print("Accuracy Dummy Comm' balanced and no validation test: : %0.2f (+/- %0.2f)" % (scores_bdummy.mean(), scores_bdummy.std() ))


#__________________________________________________
    
#Creating balanced Country Vectors for Task 2 and testing with SVM. To get the results with specific article, replace balanced_dataset
#by balanced_dataset3,balanced_dataset6 or balanced_dataset8 and the labels accordingly. 
countries_balanced=[]
for n in range(len(balanced_dataset)):
    countries_balanced.append(balanced_dataset[n]["country"])
    
    
vector_country_balanced=list(enumerate(countries_balanced)) #if doesn't work try list(vector_country)
frame_countryb=pd.DataFrame(vector_country_balanced)
targetsb = frame_countryb[1].unique()
map_to_int = {name: n for n, name in enumerate(targetsb)}
frame_countryb["subjectid"] = frame_countryb[1].replace(map_to_int)
againb=frame_countryb['subjectid'].astype(int)
againb=againb.values.reshape(-1, 1)



from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit( againb)
vector_countryb= enc.transform(againb).toarray()
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores_country.mean(), scores_country.std() * 2))
scores_countryb=perform_svm(vector_countryb,violations_balanced)
print("Accuracy SVM Comm balanced with Country_label: %0.2f (+/- %0.2f)" % (scores_countryb.mean(), scores_countryb.std() ))

#__________________________________________________
    
article3=[]
article6=[]
article8=[]
for n in range(len(example)):
    output=example[n]["articles"].encode()
    output=tokenize(output)
    for i in range (len(output)):
       if output[i]=='3':
          article3.append(example[n])
       if output[i]=='6':
           article6.append(example[n])
       if output[i]=='8':
           article8.append(example[n])

#Performing Bow SVM with balanced dataset for Article 3 
violation_3=[]
noviolation_3=[]
for n in range(len(article3)):
    output=article3[n]["decision_violations"].encode()
    output=tokenize(output)
    output=set(output)
    if "3" in output:
        violation_3.append(article3[n])
    else:
        noviolation_3.append(article3[n])

balanced_dataset_3=violation_3[0:38] + noviolation_3
dataset_3=violation_3 + noviolation_3
violations3_balanced=["1"] * 38 + ["-1"] * 38
violations3=["1"] * 218 + ["-1"] * 38

balanced_context=[]
for n in range (len(balanced_dataset_3)):
    thomaslove = balanced_dataset_3[n]["comm_text"][0:-1]
    balanced_context.append(thomaslove)
vectorizer=CountVectorizer(tokenizer=tokenize, lowercase=True,ngram_range=(1, 4), max_features=2000, stop_words=stopwords.words('english'))
balanced=vectorizer.fit_transform(balanced_context)
normalizer=Normalizer()
balanced=normalizer.transform(balanced)
balanced.toarray()

nouvel_essai=perform_svm(balanced,violations3_balanced)
print("Accuracy BoW SVM Comm balanced Article 3: %0.2f (+/- %0.2f)" % (nouvel_essai.mean(), nouvel_essai.std() ))


#Performing Bow SVM with balanced dataset for Article 6
violation_6=[]
noviolation_6=[]
for n in range(len(article6)):
    output=article6[n]["decision_violations"].encode()
    output=tokenize(output)
    output=set(output)
    if "6" in output:
        violation_6.append(article6[n])
    else:
        noviolation_6.append(article6[n])


balanced_dataset_6=violation_6[0:66] + noviolation_6
violations6_balanced=["1"] * 66 + ["-1"] * 66
dataset6=violation_6 + noviolation_6
violations6=["1"] * 173 + ["-1"] * 66
balanced_context=[]


for n in range (len(balanced_dataset_6)):
    thomaslove = balanced_dataset_6[n]["comm_text"][0:-1]
    balanced_context.append(thomaslove)
    #violations6.append(balanced_dataset_6[n]["decision_any_violation"])
vectorizer=CountVectorizer(tokenizer=tokenize, lowercase=True,ngram_range=(1, 4), max_features=2000, stop_words=stopwords.words('english'))
balanced=vectorizer.fit_transform(balanced_context)
#len(vectorizer.get_feature_names())
normalizer=Normalizer()
balanced=normalizer.transform(balanced)
balanced.toarray()

nouvel_essai=perform_svm(balanced,violations6_balanced)
print("Accuracy BoW SVM Comm balanced Article 6: %0.2f (+/- %0.2f)" % (nouvel_essai.mean(), nouvel_essai.std() ))



#Performing Bow SVM with balanced dataset for Article 8
violation_8=[]
noviolation_8=[]
for n in range(len(article8)):
    output=article8[n]["decision_violations"].encode()
    output=tokenize(output)
    output=set(output)
    if "8" in output:
        #print(output)
        violation_8.append(article8[n])
    else:
        noviolation_8.append(article8[n])
        
        
        balanced_dataset_8=violation_8[0:44] + noviolation_8

balanced_context=[]
violations8=["1"] * 44 + ["-1"] * 44

for n in range (len(balanced_dataset_8)):
    thomaslove = balanced_dataset_8[n]["comm_text"][0:-1]
    balanced_context.append(thomaslove)
    #violations6.append(balanced_dataset_8[n]["decision_any_violation"])
vectorizer=CountVectorizer(tokenizer=tokenize, lowercase=True,ngram_range=(1, 4), max_features=2000, stop_words=stopwords.words('english'))
balanced=vectorizer.fit_transform(balanced_context)
#len(vectorizer.get_feature_names())
normalizer=Normalizer()
balanced=normalizer.transform(balanced)
balanced.toarray()
nouvel_essai=perform_svm(balanced,violations8)
print("Accuracy BoW SVM Comm balanced Article 8: %0.2f (+/- %0.2f)" % (nouvel_essai.mean(), nouvel_essai.std() ))


#__________________________________________________
#Task 2 Bow and SVM Adding countries in stop_words 
    
countries_stopwords=pd.read_csv("/Users/Alison/Downloads/echr_dataset/countries.csv")
nationalities_countries=(list(countries_stopwords["nationality"]))+(list(countries_stopwords["en_short_name"]))
nationalities_countries= [w.lower() for w in nationalities_countries]
plainstring1 = [unicode(nationalities_countries[i], "utf-8")for i in range(len(nationalities_countries))]

stop_words=stopwords.words('english')
stop_words_bis= stop_words+plainstring1

balanced_context=[]

for n in range (len(balanced_dataset)):
    thomaslove = balanced_dataset[n]["comm_text"][0:-1]
    balanced_context.append(thomaslove)
vectorizer=CountVectorizer(tokenizer=tokenize, lowercase=True,ngram_range=(1, 4), max_features=2000, stop_words=stop_words_bis)
balanced=vectorizer.fit_transform(balanced_context)
#len(vectorizer.get_feature_names())
normalizer=Normalizer()
balanced=normalizer.transform(balanced)
balanced.toarray()

nouvel_essai=perform_svm(balanced,violations_balanced)
print("Accuracy BoW SVM Comm' balanced, country in stop,  and no validation test: %0.2f (+/- %0.2f)" % (nouvel_essai.mean(), nouvel_essai.std() * 2))
#__________________________________________________

#Task 2 Bow and SVM Adding countries in stop_words 
aditional_stop = ['00', '000', '01', '02', '03', '04', '05', '06', '07', '08', '09', '1', '1 2', '1 3',  '10', '10 2010', '10 29', '10 29 10', '10 29 10 2010', '100', '11', '12', '13', '14','15', '16', '17', '18', '19', '1980', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2', '2 3','20', '2000', '2001', '2002', '2003', '2004', '2005','2006', '2006 applicant', '2007', '2008', '2009', '2010','2014', '21', '22', '23', '24', '25', '26','27', '28', '29', '29 10', '29 10 2010', '3', '30', '31', '32', '33', '34', '35', '36', '39', '4', '4 convention', '40', '45', '5', '5 1','5 3', '5 4' , '50', '6', '6 1', '6 3',  '6 may', '6 may 2010','60', '7', '8', '9', '95', '98', '99', 'january', 'january 2006', 'january 2007', 'january 2008','january 2009', 'january 2010', 'january 2011', 'january 2012', 'january 2013','february', 'february 2007', 'february 2008', 'february 2009', 'february 2010', 'february 2011', 'february 2012', 'march', 'march 2006', 'march 2007', 'march 2008', 'march 2009', 'march 2010', 'march 2011', 'march 2012','april', 'april 2006', 'april 2007', 'april 2008', 'april 2009', 'april 2010', 'april 2011', 'april 2012', 'may', 'may 2006', 'may 2007', 'may 2008', 'may 2009', 'may 2010', 'may 2010 detained', 'may 2011', 'may 2012','july', 'july 2005', 'july 2007', 'july 2008', 'july 2009', 'july 2010', 'july 2011', 'july 2012', 'june', 'june 2004', 'june 2006', 'june 2007', 'june 2008', 'june 2009', 'june 2010', 'june 2011', 'june 2012','august', 'august 2005', 'august 2006', 'august 2009', 'august 2010', 'august 2011','september', 'september 2006', 'september 2007', 'september 2008', 'september 2009', 'september 2010', 'september 2011', 'september 2012','october', 'october 2003', 'october 2005', 'october 2006', 'october 2007', 'october 2008', 'october 2009', 'october 2010', 'october 2011', 'october 2012','november', 'november 2005', 'november 2006', 'november 2007', 'november 2008', 'november 2009', 'november 2010', 'november 2011', 'november 2012','december', 'december 2005', 'december 2006', 'december 2007', 'december 2008', 'december 2009', 'december 2010', 'december 2011', 'december 2012']
plainstring2=[unicode(aditional_stop[i], "utf-8")for i in range(len(aditional_stop))]
stop_words2= stop_words+plainstring2

balanced_context=[]
for n in range (len(balanced_dataset)):
    thomaslove = balanced_dataset[n]["comm_text"][0:-1]
    balanced_context.append(thomaslove)
vectorizer=CountVectorizer(tokenizer=tokenize, lowercase=True,ngram_range=(1, 4), max_features=2000, stop_words=stop_words2)
balanced=vectorizer.fit_transform(balanced_context)
#len(vectorizer.get_feature_names())
normalizer=Normalizer()
balanced=normalizer.transform(balanced)
balanced.toarray()
score_balanced=perform_svm(balanced,violations_balanced)
print("Accuracy SVM Comm' balanced, no numbers, and no validation test: %0.2f (+/- %0.2f)" %(score_balanced.mean(), score_balanced.std() * 2))

#Combination of both lists 
stop_words_comb=stop_words_bis+stop_words2


#__________________________________________________
#Code to combine feature vectors example here for countries and BoW 

balanced_context=[]
for n in range (len(balanced_dataset)):
    thomaslove = balanced_dataset[n]["comm_text"][0:-1]
    balanced_context.append(thomaslove)
vectorizer=CountVectorizer(tokenizer=tokenize, lowercase=True,ngram_range=(1, 4), max_features=2000, stop_words=stopwords.words('english'))
balanced=vectorizer.fit_transform(balanced_context)
#len(vectorizer.get_feature_names())
normalizer=Normalizer()
balanced=normalizer.transform(balanced)
balanced.toarray()

balanced=balanced.toarray()
unejolieliste=[]
for i in range(len(violations_balanced)):
   a=np.hstack((balanced[i],vector_countryb[i]))
   unejolieliste.append(a)

joliematrice=np.matrix(unejolieliste)

#Or simply a=np.hstack((balanced,vector_countryb) depending on the format 

score_balanced=perform_svm(joliematrice,violations_balanced)
print("Accuracy SVM Comm' balanced and no validation test: %0.2f (+/- %0.2f)" 
%(score_balanced.mean(), score_balanced.std()))

#__________________________________________________
# Tfidf for Task 2 SVM 

from sklearn.feature_extraction.text import TfidfVectorizer

balanced_context=[]
for n in range (len(balanced_dataset)):
    thomaslove = balanced_dataset[n]["comm_text"][0:-1]
    balanced_context.append(thomaslove)
vectorizer=TfidfVectorizer(tokenizer=tokenize, lowercase=True,ngram_range=(1, 4), max_features=2000, stop_words=stopwords.words('english'))
balanced=vectorizer.fit_transform(balanced_context)
#len(vectorizer.get_feature_names())
normalizer=Normalizer()
balanced=normalizer.transform(balanced)
balanced.toarray()
score_balanced=perform_svm(balanced,violations_balanced)
print("Accuracy SVM Comm' Tfidf balanced and no validation test: %0.2f (+/- %0.2f)" 
%(score_balanced.mean(), score_balanced.std()))


#__________________________________________________
# Clustering with K-means for Task 2 11 clusters, commented part for Spectral Clustering 


balanced_context=[]
for n in range (len(balanced_dataset)):
    thomaslove = balanced_dataset[n]["comm_text"][0:-1]
    balanced_context.append(thomaslove)
vectorizer=CountVectorizer(tokenizer=tokenize, lowercase=True,ngram_range=(1, 4), max_features=2000, stop_words=stopwords.words('english'))
balanced=vectorizer.fit_transform(balanced_context).A
normalizedb=normalizer.transform(balanced)
dist = cosine_similarity(balanced.T)
dist1=np.matrix(dist)

terms = vectorizer.get_feature_names()
n_cluster=11
#sp=SpectralClustering(n_cluster,affinity='precomputed')
km= KMeans(n_cluster)
#sp.fit(dist1)
km.fit(normalizedb.T)
#clusters=sp.labels_.tolist()
clusters=km.labels_.tolist()
topics = { 'cluster': clusters,'term':terms }
frame = pd.DataFrame(topics, index = [clusters] , columns = ['cluster', 'term'])
frame['cluster'].value_counts()
a=frame['cluster'].value_counts()
print("Top terms per cluster:")
print()

#Only for K-mean 
dicts = {}
for i in range(n_cluster):
    une_liste=[]
    #print("Cluster %d words:" % i, end='')
    if a[i]==1:
        une_liste.append(frame.loc[i]['term'])
    else:
        for title in frame.loc[i]['term'].values.tolist():
            une_liste.append(title)
    dicts[i] =une_liste
        #print(' %s,' % title, end='')
    
    
#Only for Spectral Clustering 
"""dicts = {}
for i in range(n_cluster):
    une_liste=[]
    #print("Cluster %d words:" % i, end='')
    for title in frame.loc[i]['term'].values.tolist():
        une_liste.append(title) 
    dicts[i] =une_liste
        #print(' %s,' % title, end='')"""
myframe=pd.DataFrame( columns=dicts.keys())

myframen=pd.DataFrame(balanced, columns=[terms[i]for i in range(len(terms))])


#Common part to K-means and Spectral Clustering    
for i in range(n_cluster):
    ma_liste=[]    
    for j in dicts[(i)]:
        ma_liste.append(j)
        myframen['maliste %d'% (i)]=myframen[ma_liste].sum(axis=1)

#Here 2011 because 11 clusters, to be adapted when different number of clusters          
myframe=myframen.iloc[:,2000:2011]

myframe = normalizer.transform(myframe)
myframe= np.matrix(myframe)
score_balanced=perform_svm(myframe,violations_balanced)
print("Accuracy SVM Comm' balanced and no validation test: %0.2f (+/- %0.2f)" %(score_balanced.mean(), score_balanced.std() ))

#To stack BoW and topics
balanced_context=[]
for n in range (len(balanced_dataset)):
    thomaslove = balanced_dataset[n]["comm_text"][0:-1]
    balanced_context.append(thomaslove)
vectorizer=CountVectorizer(tokenizer=tokenize, lowercase=True,ngram_range=(1, 4), max_features=2000, stop_words=stop_words_bis)
balanced=vectorizer.fit_transform(balanced_context)
#len(vectorizer.get_feature_names())
normalizer=Normalizer()
balanced=normalizer.transform(balanced)
balanced=balanced.toarray()
joliematrice=np.hstack((myframe,balanced))
joliematrice=np.matrix(joliematrice)
score_balanced=perform_svm(joliematrice,violations_balanced)
print("Accuracy BoW and Topics SVM Comm' balanced and no validation test: %0.2f (+/- %0.2f)" 
%(score_balanced.mean(), score_balanced.std()))


#__________________________________________________

#To perform PCA for best number of clusters 


Sigma = np.cov(balanced.T)
evals, evecs = np.linalg.eig(Sigma)
sorted_evals=np.sort_complex(evals)
plt.plot(sorted_evals)
plt.xlabel('PCs in descending order')
plt.ylabel('Projected variance')

#For cumulative Variance 
#c_var = np.cumsum(evals/np.sum(sorted_evals)) #cumulative variance. 
#plt.plot(c_var) 


#__________________________________________________

#For Doc2vec : example with Task 2 


#Updated
balanced_context=[]
for n in range (len(balanced_dataset)):
    communication_texts = balanced_dataset[n]["comm_text"][0:-1]
    balanced_context.append( communication_texts )


docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i, text in enumerate(balanced_context):
    words = text.lower().split()
    words=[x for x in words if x not in stopwords.words('english')]
    tags = [i]
    docs.append(analyzedDocument(words, tags))

import random
model = doc2vec.Doc2Vec(size = 100, window = 4, min_count = 1, workers = 4)
model.build_vocab(docs)
alpha_val = 0.025        # Initial learning rate
min_alpha_val = 1e-4     # Minimum for linear learning rate decay
passes = 15              # Number of passes of one document during training

alpha_delta = (alpha_val - min_alpha_val) / (passes - 1)
for epoch in range(20):

    # Shuffling gets better results

    random.shuffle(docs)

    # Train

    model.alpha, model.min_alpha = alpha_val, alpha_val

    model.train(docs, total_examples=model.corpus_count, epochs=model.iter)

    # Logs

    print('Completed pass %i at alpha %f' % (epoch + 1, alpha_val))

    # Next run alpha

    alpha_val -= alpha_delta
model.save('/Users/Alison/Downloads/echr_dataset/my_model.doc2vec')
model = Doc2Vec.load('/Users/Alison/Downloads/echr_dataset/my_model.doc2vec')
score_balanced=perform_svm(model.docvecs,violations_balanced)
print("Accuracy Doc2vec SVM Comm' balanced and no validation test: %0.2f (+/- %0.2f)" 
%(score_balanced.mean(), score_balanced.std()))

