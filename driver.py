import pandas as pd
import os
import glob;
train_path = "../resource/asnlib/public/aclImdb/train/" # use terminal to ls files under this directory
test_path = "../resource/asnlib/public/imdb_te.csv" # test data for grade evaluation
from sets import Set
import csv
import numpy as np;
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

def predict_int(predict):
	new=[]
	for i in predict:
		n=int(i);
		new.append(n)
	return new;	

def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
 	'''Implement this module to extract
	and combine text files under train_path directory into 
    imdb_tr.csv. Each text file in train_path should be stored 
    as a row in imdb_tr.csv. And imdb_tr.csv should have two 
    columns, "text" and label'''
	 # number of training examples to see pos,neg
	print "data preprocessing";
	pos_path = inpath+"pos"
	neg_path = inpath+"neg";
	pos_files = os.listdir(pos_path);
	neg_files = os.listdir(neg_path);
	del neg_files[0];
	del pos_files[0];
	#print len(pos_files),len(neg_files)
	pos_read = 0;
	neg_read = 0;
	index = 0;
	examples = 12000#len(pos_files);
	finput = open("stopwords.en.txt","r");
	with finput as f:
		data = f.readlines();
	data = [x.strip('\n') for x in data]
	sw = set() #stopwords
	for i in data:
		sw.add(i);
		
	#print data;
	
	file_name = "imdb_tr.csv";
	myfile = open(file_name, 'wb')
	wr = csv.writer(myfile)
	row= []
	row.append("");
	row.append("text");
	row.append("polarity");
	wr.writerow(row);
       	flag=1;
	w=0;
	while( (pos_read<examples) and  (neg_read<examples)):
		
		if(flag==1):
			loc = pos_path+"/"+pos_files[pos_read];
			flag=0;
			polarity =int( 1)
			pos_read = pos_read+1;
		else:
			loc = neg_path+"/"+neg_files[neg_read];
			neg_read = neg_read+1;
			flag=1;
			polarity = int(0);
		finput  = open(loc,"r");
		with finput as f:
			data=f.readlines();
			
		sentence =  data[0]
		
			#print sentence;
		line = sentence.split(" ");
		line2 = [];
		#print line;
		#print line[3];
		for x in range( len(line)):

			if(x >= len(line)):
				break;
			word = line[x];
			word=word.lower();
			#print word;
			if(word in sw):
				w=w+1
				#print x,word
				#del line[x];
			else:
				line2.append(word);
		
		#print line2;
		outline = " ".join(line2);
		out = [];
		out.append(outline);
		out.append(polarity);
		wr.writerow([index]+out)
		index=index+1;
def bigram(x,y,test):
	bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                    token_pattern=r'\b\w+\b', min_df=1)
	x_train  = bigram_vectorizer.fit_transform(x);
	clf = SGDClassifier(loss='hinge', penalty='l1').fit(x_train,y )
	x_test = bigram_vectorizer.transform(test);
	predict = clf.predict(x_test);
	new=predict_int(predict);
	return new;

def tfidf_bigram(x,y,test):
	'''
	bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                    token_pattern=r'\b\w+\b', min_df=1)
        x_train  = bigram_vectorizer.fit_transform(x);
        #TFIDF USING BIGRAM                                                                              
	tfidf_vector = TfidfVectorizer(use_idf=True,min_df=1);
        x_train_tfidf = tfidf_vector.fit_transform(x_train);
        clf = SGDClassifier(loss='hinge', penalty='l1').fit(x_train_tfidf,y )
        x_test = tfidf_vector.transform(test);
        p2 = clf.predict(x_test)
	'''
	bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                    token_pattern=r'\b\w+\b', min_df=1)
        x_train  = bigram_vectorizer.fit_transform(x);
	tfidf_vector = TfidfTransformer();
	x_train_tfidf = tfidf_vector.fit_transform(x_train);
        clf = SGDClassifier(loss='hinge', penalty='l1').fit(x_train_tfidf,y )
        bi_test = bigram_vectorizer.transform(test);
	x_test = tfidf_vector.transform(bi_test);
        predict = clf.predict(x_test);
        new=predict_int(predict);

        return new;


def tfidf_unigram(x,y,test):
	vectorizer = CountVectorizer(min_df=1);
        x_train_uni = vectorizer.fit_transform(x);
	
	#tf-idf                                                                       
        tfidf_vector = TfidfTransformer();                       
        x_train_tfidf = tfidf_vector.fit_transform(x_train_uni);                      
        clf = SGDClassifier(loss='hinge', penalty='l1').fit(x_train_tfidf,y )         
        uni_test = vectorizer.transform(test);
	x_test = tfidf_vector.transform(uni_test);                                        
        p2 = clf.predict(x_test)
	new=predict_int(p2);
	return new;
def unigram(x,y,test):
	corpus=[]
	train_labels=[];
	#print x[1],type(y[1]);
	for i in x:
		corpus.append(i);
	
	vectorizer = CountVectorizer(min_df=1);
	x_train_uni = vectorizer.fit_transform(x);
	clf = SGDClassifier(loss='hinge', penalty='l1').fit(x_train_uni,y )
	x_test = vectorizer.transform(test)
	predict = clf.predict(x_test);
	new=predict_int(predict);
	return new;

if __name__ == "__main__":
	unigram_file = "unigram.output.txt"
        bigram_file = "bigram.output.txt"
        uni_tf = "unigramtfidf.output.txt"
        bi_tf = "bigramtfidf.output.txt"
	save=[1,0,1,0,1,1,1,0,0,0,0];
	np.savetxt("testing.csv",save,fmt='%i',delimiter=",");
	imdb_data_preprocess(train_path);
	finput = open("imdb_tr.csv","r");
	#test=pd.read_csv(test_path, sep=",", header=None,encoding = 'ISO-8859-1')
	
	data = pd.read_csv("imdb_tr.csv",index_col=0);
	test = pd.read_csv(test_path,index_col=0,encoding = 'ISO-8859-1');
	
	unigram=unigram(data.text,data.polarity,test.text);
	
	np.savetxt(unigram_file,unigram,fmt='%i',delimiter=",");
	
	bigram= bigram(data.text,data.polarity,test.text);
	np.savetxt(bigram_file,bigram,fmt='%i',delimiter=",");
	tf_uni= tfidf_unigram(data.text,data.polarity,test.text);
	np.savetxt(uni_tf,tf_uni,fmt='%i',delimiter=",");
	tf_bi= tfidf_bigram(data.text,data.polarity,test.text);
	np.savetxt(bi_tf,tf_bi,fmt='%i',delimiter=",");
	
	mismatch=0;
	
	
	

