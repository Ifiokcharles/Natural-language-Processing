import matplotlib.pyplot as plt
import random
import nltk
import re
import pandas as pd
import pylab as pl
import sklearn.datasets
import nltk
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('gutenberg')
from nltk.corpus import gutenberg 

files_en = gutenberg.fileids()
selected_titles = ['3623-8.txt','19528-8.txt','24681-8.txt']
label_titles = ['a','b','c']
#Downloading and opening 5 books
#upload the text 
text_1 = gutenberg.open('3623-8.txt').read()
text_2 = gutenberg.open('19528-8.txt').read()
text_3 = gutenberg.open('24681-8.txt').read()


#remove numbers from the text
removeNum1 = re.sub('[^a-zA-Z]',' ', text_1 )
removeNum2 = re.sub('[^a-zA-Z]',' ', text_2 )
removeNum3 = re.sub('[^a-zA-Z]',' ', text_3 )


#Tokenizing data
from nltk import regexp_tokenize
pattern = r'''(?x) (?:[A-Z]\.)+ | \w+(?:[-]\w+)* | \$?\d+(?:\.\d+)?%?| \.\.\. | [][.,;"'?():-_`]'''

tokens_1 = regexp_tokenize(text_1, pattern)
tokens_2 = regexp_tokenize(text_2, pattern)
tokens_3 = regexp_tokenize(text_3, pattern)


#Removing stopwords
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
tokenstopwords_1 = [w for w in tokens_1 if not w in stop_words]
tokenstopwords_2 = [w for w in tokens_2 if not w in stop_words]
tokenstopwords_3 = [w for w in tokens_3 if not w in stop_words]


#we provide several stemmers
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
stems_1 = []
for t in tokenstopwords_1:    
    stems_1.append(porter.stem(t))
print(stems_1)
stems_2 = []
for t in tokenstopwords_2:    
    stems_2.append(porter.stem(t))
print(stems_2)
stems_3 = []
for t in tokenstopwords_3:    
    stems_3.append(porter.stem(t))
print(stems_3)


lemmatizer = WordNetLemmatizer()
lemma_1 = []
for t in tokenstopwords_1:    
    lemma_1.append(lemmatizer.lemmatize(t))
lemma_2 = []
for t in tokenstopwords_2:    
    lemma_2.append(lemmatizer.lemmatize(t))
lemma_3 = []
for t in tokenstopwords_3:    
    lemma_3.append(lemmatizer.lemmatize(t))


#Creating 200 documents from each book. 
#Each document contains 150 words    
doc_1 = []
doc_2 = []
doc_3 = []


for i in range(0,200):
    for j in range(0,151):
        data_1 = (random.sample(lemma_1, j))
    for j in range(0,151):
        data_2 = (random.sample(lemma_2, j))
    for j in range(0,151):
        data_3 = (random.sample(lemma_3, j))
   
    doc_1.append(' '.join(str(''.join(str(x) for x in v)) for v in data_1))
    doc_2.append(' '.join(str(''.join(str(x) for x in v)) for v in data_2))
    doc_3.append(' '.join(str(''.join(str(x) for x in v)) for v in data_3))


   

# Adding labels to each text of 200 document
df = pd.DataFrame()
df['text']  = doc_1


df1 = pd.DataFrame()
df1['text']  = doc_2


df2 = pd.DataFrame()
df2['text']  = doc_3


df5 = pd.concat([df,df1,df2],ignore_index=True) # Combining all dataframe into one.
df8=df5.rename({'text':'booktexts'}, axis='columns') # Renaming of the colomns



# Creating a corpus of all 5 books
corpus = []
for i in range(0, 600):
    review = re.sub('[^a-zA-Z]',' ', df8['booktexts'][i]) #this will only keep letters from A-z and will remove any numbers and puntuation
    review = review.lower()# this will convert all letters to lower cases
    reviewer = " ".join(review.split())# remove white spaces
    corpus.append(reviewer)
# Creating the Bags of Words Model
from sklearn.feature_extraction.text import CountVectorizer  
vectorizer = CountVectorizer()  
Xvec = vectorizer.fit_transform(corpus).toarray()
X_bow = pd.DataFrame(Xvec)

#Td-if
X_tf = TfidfVectorizer()
X_tfd =  X_tf.fit_transform(corpus).toarray()
X_tfd = pd.DataFrame(X_tfd)


# Applying LDA
from sklearn.decomposition import  LatentDirichletAllocation as LDA
lda = LDA(n_components = 5)
X_lda = lda.fit_transform(X_bow)
X_LDa = pd.DataFrame(X_lda)


Xbow_cluster = X_bow.iloc[:, 0:X_bow.size].values
Xtf_cluster = X_tfd.iloc[:, 0:X_tfd.size].values




# kmeans
from sklearn.cluster import KMeans
modelkmeansbow = KMeans(n_clusters=3, init='k-means++', random_state = 0)
modelkmeansbow.fit(Xbow_cluster)
y_kmeansbow = modelkmeansbow.fit_predict(Xbow_cluster)

modelkmeanstf = KMeans(n_clusters=3, init='k-means++',  random_state = 0)
modelkmeanstf.fit(Xbow_cluster)
y_kmeanstf = modelkmeanstf.fit_predict(Xtf_cluster)


order_centroidsbow = modelkmeansbow.cluster_centers_.argsort()[:, ::-1]
termsbow = vectorizer.get_feature_names()

order_centroidstf = modelkmeanstf.cluster_centers_.argsort()[:, ::-1]
termstf = vectorizer.get_feature_names()


# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogrambow = sch.dendrogram(sch.linkage(Xbow_cluster, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('authors')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hcbow = AgglomerativeClustering(n_clusters=3, affinity = 'euclidean', linkage = 'ward')
y_hcbow = hcbow.fit_predict(Xbow_cluster)

#tf-idf
dendrogramtf = sch.dendrogram(sch.linkage(Xtf_cluster, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('authors')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hctf = AgglomerativeClustering( n_clusters=3, affinity = 'euclidean', linkage = 'ward')
y_hctf = hctf.fit_predict(Xtf_cluster)

from sklearn.metrics import cohen_kappa_score
cohen_kappa_score(y_kmeansbow, y_hcbow)
