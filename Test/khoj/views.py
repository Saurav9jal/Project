from django.shortcuts import render_to_response, get_object_or_404
from django.http import Http404,HttpResponse, HttpResponseRedirect
from django.template import RequestContext
from pygoogle import pygoogle
from django import forms
from khoj.forms import PostForm
from collections import OrderedDict

import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction

from bs4 import BeautifulSoup
import urllib
#from mpl_toolkits.gtktools import error_message





global infereddic 
infereddic = {}
infereddic2 = {}
global s_term

def khoj(request):
    
    #form = PostForm(request.POST)
    #s_term = request.POST.get('s_term',request.GET.get('s_term',None))
    s_term =PostForm(request.POST)
    #s_term = forms.CharField(error_messages = my_default_errors) 
    top_10 = {}
    dummy = OrderedDict()
    dummy2 = OrderedDict()
    if request.POST:
        s_term =PostForm(request.POST)
        dummy = OrderedDict()
        #dummy2.clear()
        if s_term.is_valid():
            c_dummy2 = ""
            print s_term.cleaned_data
            #s_term.save()
            if s_term.cleaned_data.values()[0] in infereddic2.keys():
                c_dummy = infereddic2[s_term.cleaned_data.values()[0]]
                print type(c_dummy)
                c_dummy2 = " ".join(c_dummy)
            else :
                infereddic2.setdefault(s_term.cleaned_data.values()[0],[])
                
                
            #if s_term.cleaned_data.values()[0] in infereddic.keys():
             #   dummy = infereddic[s_term.cleaned_data.values()[0]]
            #else :
             #   infereddic.setdefault(s_term.cleaned_data.values()[0], {})
            #print infereddic
            #print type(c_dummy)
            print ";;;;;;;;;;;;;;;;;;;;"
            print s_term.cleaned_data.values()[0]+" "+c_dummy2
            result = pygoogle(s_term.cleaned_data.values()[0]+" "+c_dummy2)
            result.pages = 2
            top_10 = {}
            n=0
            print "dummmmmmmyy" 
            print dummy
            for k,v in result.search().iteritems():
                if n<10:
                    if k in dummy.keys():
                        n+=1
                    else:
                        top_10[k]=v
                        n += 1
                else :
                    break
            print "top_100->"
            print top_10
            #if dummy:
             #   
              #  dummy2 = OrderedDict(dummy.items()+top_10.items())
               # print "dummyyy"
                #print dummy
                #print "2"
                #print dummy2
            #else:
             #   dummy2 = top_10
            
            if 'select-id' in request.POST:
                selected_ids = request.POST.getlist('select-id',[])
                text1 = []
                for i in selected_ids:
                    r_open = urllib.urlopen(i[:-1]).read()
                    soup = BeautifulSoup(r_open)
                    text1.append(soup.title.string)
                
                stopwords = nltk.corpus.stopwords.words('english')
                print stopwords[:10]
                from nltk.stem.snowball import SnowballStemmer
                stemmer = SnowballStemmer('english')
                
                def tokenize_and_stem(text):
                    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
                    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
                    #print tokens    
                    filtered_tokens = []
                    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
                    for token in tokens:
                        if re.search('[a-zA-Z]', token):
                            filtered_tokens.append(token)
                    stems = [stemmer.stem(t) for t in filtered_tokens]
                    #print stems[:20]    
                    return stems

                def tokenize_only(text):
                    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
                    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
                    filtered_tokens = []
                    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
                    for token in tokens:
                        if re.search('[a-zA-Z]', token):
                            filtered_tokens.append(token)
                    #print filtered_tokens[:20]
                    return filtered_tokens
                
                totalvocab_stemmed = []
                totalvocab_tokenized = []
                
                for i in text1:
                    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
                    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
                    allwords_tokenized = tokenize_only(i)
                    totalvocab_tokenized.extend(allwords_tokenized)

                vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
                print 'there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame'
 
                from sklearn.feature_extraction.text import TfidfVectorizer

                #define vectorizer parameters
                tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                                   min_df=0.2, stop_words='english',
                                                   use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

                tfidf_matrix = tfidf_vectorizer.fit_transform(text1) #fit the vectorizer to synopses

                print(tfidf_matrix.shape)

                terms = tfidf_vectorizer.get_feature_names()

                from sklearn.metrics.pairwise import cosine_similarity
                dist = 1 - cosine_similarity(tfidf_matrix)

                from sklearn.cluster import KMeans

                num_clusters = 1

                km = KMeans(n_clusters=num_clusters)
                km.fit(tfidf_matrix)

                clusters = km.labels_.tolist()
                print clusters

                #from __future__ import print_function
                order_centroids = km.cluster_centers_.argsort()[:, ::-1]
                print "-------start--------------------------------------------"
                for i in range(num_clusters):
                    print "Cluster %d words:" % i
    
                    for ind in order_centroids[i, :3]: #replace 6 with n words per cluster
                        print "---------------------this----------------------------------"
                        print ' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore')
                        infereddic2[s_term.cleaned_data.values()[0]].append(vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'))
  
                
                
                #infereddic2[s_term.cleaned_data.values()[0]].append(vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore')[0])]) 
                #----------------------------------------------------------------------------    
                #for i in selected_ids:
                    #print "This is id"+i
                    
                    
                    
               #     for k,v in dummy2.iteritems():
                        #print k+"-------"+v
              #          if v==i[:-1]:
                            #print v
             #               infereddic[s_term.cleaned_data.values()[0]][k]=v
                            #print k
            print "-------"
            #print infereddic
            print "=========="
                #print selected_ids
                #print s_term.cleaned_data
                
    return render_to_response('khoj.html',{'form':s_term,'result': top_10}, context_instance=RequestContext(request))
    
    
    
    
    '''if request.POST:
        if 'term' in request.POST:
            s_term = request.POST['term']
            if s_term in infereddic.keys():
                dummy = infereddic[s_term]
                return render_to_response('khoj.html',{'result': dummy}, context_instance=RequestContext(request))
                
            else :
                n=0
                infereddic.setdefault(s_term, {})
                res = pygoogle(s_term)
                res.pages = 2
                top_10 = {}
                for k,v in res.search().iteritems():
                    if n<10:
                        top_10[k]=v
                        n += 1
                    else :
                        break
                dummy = top_10
                render_to_response('khoj.html',{'result': dummy}, context_instance=RequestContext(request))
                if 'clicked' in request.POST:
                    urls = request.POST['url']
                    tle = request.POST['title']
                    infereddic[s_term][urls]=tle
            return render_to_response('khoj.html',{'result': dummy}, context_instance=RequestContext(request))
            '''























