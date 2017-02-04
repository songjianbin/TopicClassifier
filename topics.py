'''
part 2 : topic modelling

Description of the problem :

Goal : Find the topic for each document P(T | D) where D is a list of words. 

The problem involves 3 various modes : supervised , semi supervised and unsupervised learning

The supervised learning method involves using all the training data.Naive Bayes modelis used to predict the topics

P(Topic | Document) = argmax(i) [  (P(Document | Topic_i)*P(Topic_i))  ]

For semisupervised learning , Expectation Maximization Algorithm is used.

To select the documents in training data for which labels can be seen , a biased coin with bias== fraction is flipped.

Once the documents are selected , the rest of the train data is considered as validation. 

Using the priors of the selected train data, the topics of the validation are predicted .This is called the Expectation step(E-step)

Once the validation topics are predicted, we assume they are right and update the priors and likelihood. This step is called the 

Maximization step(M-Step)

The E-Step and M-Step are done back and forth for either 5 times or till convergence. 

Once, we have the labels for the whole training data , we use these labels and predict topics for test data like the same way 

as done in the supervised approach(i.e Naive Bayes)

For Unsupervised learning, Expectation Maximization Algorithm is used again. However, since we do not have labels for the train data, 
the labels are randomly asigned and the EM approach is used to converge the clusters.

With the given labels, we predict the topics for test data. 

A brief decription of functions :

get_document : opens a documents, takes all words into a lost after removong stop words and punctuations .

get_files: takes a folder and creates a dict with subfolder as key and list of files in it as value

likelihood_dict :  calculates likelihood for each (word, topic) tuple 

top_words: gets top 10 words for each topic based on P(t_i | w_j) = P(w_j|t_i)*P(t_i)

coin_flip :flips a biased coin with probability of "yes" as given fraction

fraction_split : takes given fraction of files from train data

prior : calculates prior for each topic and returns a dict

updated_priors : calculates updated priors for semi supervised and unsupervised learning

max_pos : gives argmax for all posteriors P( T | W)

Problems, assumptions, simplifications: 

a)The first step is reading data: All punctuations are replaced, and the top 20 stopwords are removed. 

b)The data converges typically in 2-3 iterations.So, the max number of iterations is limited to 5. 

c)log of the posterior probabilty is taken since the values are too small.

Results: 

a) For supervised learning , the accuracy is 72.32% . The confusion matrix can be seen in the file confusion_matrix_1

b)For semisupervised learning with fraction == 0.8, the accuracy is 70.89%. The confusion matrix can be seen in the file confusion_matrix_0.8

c)For semisupervised learning with fraction == 0.5, the accuracy is 67.95%. The confusion matrix can be seen in the file confusion_matrix_0.5

d)For semisupervised learning with fraction == 0.1, the accuracy is 53.47%. The confusion matrix can be seen in the file confusion_matrix_0.1

e)For unsupervised learning i.e fraction == 0, the accuracy is 5.96%.The confusion matrix can be seen in the file confusion_matrix_0 

Note: The results may vary by 2-3% everytime because of randomness in generating labels and selecting train data with a coinflip

'''
import numpy as n
import os 
import sys
import random
import math
import re
import pickle
import json
from copy import deepcopy
from collections import Counter
#imports the construct_confusion_matrix from confusion_matrix.py
from confusion_matrix import *
#read all the args

mode = sys.argv[1]
folder = sys.argv[2]
model_file = sys.argv[3] 
if mode == "train":
	fraction = float(sys.argv[4])


# opens a documents, takes all words into a lost after removong stop words and punctuations .
#input document name (and path)
#output : list of words 
def get_document(document):
	f=open(document, 'r')
	words = re.split(" |\.",f.read().replace('-','').replace('\n',' ').replace('>','').replace('*','').replace('(','').replace(')','').replace('<','').replace('/','').replace('\\','').strip().lower())
	f.close()
	#stop words take from http://www.audiencedialogue.net/stopwords.html
	stop_words = set(['the','of', 'and', 'to', 'a', 'in' , 'that', 'is', 'was','he', 'for','it','with','as','his','on','be','at','by','i'])
	return list(set(words)-stop_words)

#takes a folder and creates a dict with subfolder as key and list of files in it as value
#input :folder name
#output: dictionary : {topic : [files]}
def get_files(folder):
	classes = os.listdir(folder)
	doc_list={}
	for cls in classes:
		if cls != ".DS_Store":
			doc_list[cls]=[(get_document(folder+"/"+cls+"/"+i),folder+"/"+cls+"/"+i) for i in os.listdir(folder+"/"+cls)]
	return doc_list

#input: [files]
#output : dictionary {(word,topic):P(word|topic)}
def likelihood_dict(files):
	print "computing likelihoods"
	word_dict={}
	total_dict={}
	for key in files:
		word_count = 0 
		#total_dict[key]  = len(files[key])  #no of documents in a topic
		for words,doc_name in files[key]:
			word_count+=len(words)
			for word in words:
				if (word,key) in word_dict:
					word_dict[(word,key)]+1.0
				else:
					word_dict[(word,key)]=1.0
		total_dict[key] = word_count		
	for tup in word_dict:
		word_dict[tup]= float(word_dict[tup])/total_dict[tup[1]]
	return word_dict

#top 10 words in each docs 
def top_words(likelihood):
	d= {}
	tup_list=[]
	for i in files.keys():
		temp = filter(lambda tup:tup[0][1]==i, likelihood.items())
		temp2 = sorted(temp, key= lambda tup:tup[1], reverse= True)[0:9]
		for j in temp2:
			if j[0][1] in d:
				d[j[0][1]]+=[j[0][0]]
			else:
				d[j[0][1]]=[j[0][0]]
		
	#tup_list = [sorted(filter(lambda tup:tup[0][1]==i,likelihood.items()), key= lambda tup:tup[1], reverse=True)[0:9] for i in files.keys()]
	#print [(tup[0][0][1], tup[0][0][0]) for tup in tup_list]
	#q=  {d.setdefault(k,[]).append(v) for k,v in tuple([(tup[0][0][1],tup[0][0][0]) for tup in tup_list])}	
	f = open("distinctive_words.txt", "w")
	#print d
	json.dump(d, f)
	f.close()

#output : "yes" or "no" woth P(yes)= fraction
def coin_flip(fraction):
	return "yes" if float(random.randrange(0,101))/100 <= fraction else "no"

#for semi supervised learning
#input : fraction,files
#output: train dict, validation dict
def fraction_split(fraction , files ):
	train_dict={}
	validation_dict={}
	for key in files:
		train,validation=[],[]
		for fle in files[key]:
			if coin_flip(fraction)=="yes":
				train.append(fle) 
			else :
				validation.append(fle)
		train_dict[key] = train
		validation_dict[key]= validation
	return train_dict, validation_dict

#input : files
#output : dictionary {topic : P(topic)}
def prior(files):
	return {key: float(len(value))/sum(map(len, files.values())) for key,value in files.items()}

#input : result dictionary generated in semi supervised learning after every iteration
#output : dictionary {topic : P(topic)}
def updated_prior(dict1):
	pred=  [tup[0] for tup in dict1.values()]
	return {k: float(pred.count(k))/len(pred) for k in pred}

#output : P(topic | Words)
def posterior(fle, topic, prior):
	words = get_document(fle)
	pos = math.log(prior[topic])
	for word in words:
		pos += math.log(likelihood[(word,topic)]) if (word,topic) in likelihood else 10**-8
	return pos

#list_of_topics
def topics(files):
	return files.keys()

#argmax(P(T_i | words)) 
def max_pos(fle, list_of_topics, prior):
	return min([(posterior(fle, topic, prior),topic) for topic in list_of_topics])		
		
#merges two dictionaries
def merge_dicts(dict1, dict2):
	return {k:list(set(dict1[k])|set(dict2[k])) for k in set(dict1.keys()+dict2.keys())}

#list of docs in a folder
def doc_list(folder):
	f= []
	for fle in os.listdir(folder):
		for item in  os.listdir(folder+"/"+fle):
			f.append((folder+"/"+fle+"/"+item,fle))
	return f

#returns topics given result in semisupervised leanrning to update priors
def get_topics(dict1, list_of_topics):
	file_dict = {}
	files = dict1.keys()
	for fle in files:
		pred = dict1[fle][0]
		if pred in file_dict:
			file_dict[pred] += [(get_document(fle),fle)]
		else:
			file_dict[pred] = [(get_document(fle), fle)]
			
	return file_dict

#merges all values in dict where value is a list into a single list
def get_values(dict1):
	return reduce(lambda x,y:x+y , dict1.values())


files = get_files(folder)
#print sum(prior(files).values())

#train mode begins
if mode=="train":
	#training for supervised leaarning
	if fraction == 1.0:
		print "reading files"
		print "computing likelihood, training data"
		likelihood = likelihood_dict(files)
		top_words(likelihood)
		#print len(likelihood)
		f = open(model_file, "w")
		pickle.dump(likelihood, f)
		f.close()
	#training for semi supervised learning
	if 0 < fraction < 1.0:
		list_of_topics= topics(files)
		result_dict={}
		#split data using given fraction
		train , validation = fraction_split(fraction, files)
		for fle in train:
			result_dict[fle[0]]= (fle[1], fle[1])
		priors = prior(train)
		#expectaton step begins here
		for i in range(5):
			print i , "iteration"
			#likelihood is updated after every iteration
			likelihood = likelihood_dict(train)
			validation_docs= get_values(validation)
			total = len(validation_docs)
			count = 0
			for fle in validation_docs:
				#print count
				count+=1
				topic= max_pos(fle[1] , list_of_topics, priors)
				#print topic[1] , fle[1].split('/')[2] , fle[1]
				result_dict[fle[1]]= (topic[1], fle[1].split('/')[-2])
				if count%(total/10)==0:
					sys.stdout.write("\r")
					sys.stdout.write(str(['#']*(1+(100*count/total)/10)+[" "]*(10-(1+(100*count/total)/10)))+"---"+str((count/(total/10))*10) + " percent completed in iteration number "+ str(i))
					sys.stdout.flush()
			#maximization step
			priors = updated_prior(result_dict)
			print float(len(filter(lambda tup:tup[0]==tup[1], result_dict.values())))/len(result_dict.values())
		#write likelihood to file
		top_words(likelihood)
		f = open(model_file, "w")
		pickle.dump(likelihood, f)
		f.close()
	#training for unsupervised learning
	if fraction ==0 :
		list_of_topics= topics(files)
		result_dict={}
		train = get_values(get_files(folder))
		#topics are assgined randomly
		for fle in train:
		
			#print fle[1] ,  fle[1].split('/')[2]
			result_dict[fle[1]]= (random.choice(list_of_topics), fle[1].split('/')[-2])
		files = get_topics(result_dict, list_of_topics)
		y= 0 
		priors = prior(files)
		for i in range(5):
			print i , "iteration"
			likelihood = likelihood_dict(files)
			validation_docs= get_values(files)
			
			total = len(validation_docs)
			count = 0
			for fle in validation_docs:
				#print count
				count+=1
				topic= max_pos(fle[1] , list_of_topics, priors)
				#print topic[1] , fle[1].split('/')[2], fle[1]
				result_dict[fle[1]]= (topic[1], fle[1].split('/')[-2])
				if count%(total/10)==0:
					sys.stdout.write("\r")
					sys.stdout.write(str(['#']*(1+(100*count/total)/10)+[" "]*(10-(1+(100*count/total)/10)))+"---"+str((count/(total/10))*10) + " percent completed in iteration number "+ str(i))
					sys.stdout.flush()
			priors = updated_prior(result_dict)
			
			x = float(len(filter(lambda tup:tup[0]==tup[1], result_dict.values())))/len(result_dict.values())
			if x==y:
				break
			y = deepcopy(x)
		top_words(likelihood)
	
		f = open(model_file, "w")
		pickle.dump(likelihood, f)
		f.close()
			
#testmode 
if mode == "test":
	result_dict={}
	f = open(model_file, "r")
	print "reading data from training file"
	likelihood = pickle.load(f)
	f.close()
	#print likelihood[str(("the", "religion"))]
	#print likelihood[str(("the", "guns"))]
	priors = prior(files)
	test_documents = doc_list(folder)
	list_of_topics= topics(files)
	correct = 0
	total = len(test_documents)
	count = 0
	print "hangon there!!, labelling documents " 
	for fle in test_documents:
		count+=1
		topic = max_pos(fle[0], list_of_topics , priors)
		#print fle[1], topic[1]
		#print fle[0] , fle[0].split('/')[3] 
		result_dict[fle[0].split('/')[-1]]= (topic[1], fle[1])
		if topic[1] == fle[1]:
			correct+=1.0
		if count%(total/10)==0:
			sys.stdout.write("\r")
			sys.stdout.write(str(['#']*(1+(100*count/total)/10)+[" "]*(10-(1+(100*count/total)/10)))+"---"+str((count/(total/10))*10) + " percent completed")
			sys.stdout.flush()
	#print len(result_dict )
	cm=  construct_confusion_matrix(list_of_topics , result_dict)	
	f  = open("confusion_matrix",'w')
	f.write(cm)
	f.close()
	print "\n accuracy", correct / total
