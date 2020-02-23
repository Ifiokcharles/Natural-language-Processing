import matplotlib.pyplot as plt
import random
import nltk
import re
import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim
from sklearn import mixture
from copy import deepcopy
from sklearn.metrics.cluster import adjusted_rand_score
from gensim import corpora, models
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('gutenberg')
from nltk.corpus import gutenberg 

files_en = gutenberg.fileids()
selected_titles = ['3623-8.txt','19528-8.txt','24681-8.txt','29444-8.txt','milton-paradise.txt']
#Downloading and opening 5 books
#upload the text 
text_1 = gutenberg.open('3623-8.txt').read()
text_2 = gutenberg.open('19528-8.txt').read()
text_3 = gutenberg.open('24681-8.txt').read()
text_4 = gutenberg.open('29444-8.txt').read()
text_5 = gutenberg.open('milton-paradise.txt').read()

#remove numbers from the text
removeNum1 = re.sub('[^a-zA-Z]',' ', text_1 )
removeNum2 = re.sub('[^a-zA-Z]',' ', text_2 )
removeNum3 = re.sub('[^a-zA-Z]',' ', text_3 )
removeNum4 = re.sub('[^a-zA-Z]',' ', text_4 )
removeNum5 = re.sub('[^a-zA-Z]',' ', text_5 )

#Tokenizing data
from nltk import regexp_tokenize
pattern = r'''(?x) (?:[A-Z]\.)+ | \w+(?:[-]\w+)* | \$?\d+(?:\.\d+)?%?| \.\.\. | [][.,;"'?():-_`]'''

tokens_1 = regexp_tokenize(text_1, pattern)
tokens_2 = regexp_tokenize(text_2, pattern)
tokens_3 = regexp_tokenize(text_3, pattern)
tokens_4 = regexp_tokenize(text_4, pattern)
tokens_5 = regexp_tokenize(text_5, pattern)

#Removing stopwords
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
tokenstopwords_1 = [w for w in tokens_1 if not w in stop_words]
tokenstopwords_2 = [w for w in tokens_2 if not w in stop_words]
tokenstopwords_3 = [w for w in tokens_3 if not w in stop_words]
tokenstopwords_4 = [w for w in tokens_4 if not w in stop_words]
tokenstopwords_5 = [w for w in tokens_5 if not w in stop_words]

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
stems_4 = []
for t in tokenstopwords_4:    
    stems_4.append(porter.stem(t))
print(stems_4)
stems_5 = []
for t in tokenstopwords_5:    
    stems_5.append(porter.stem(t))
print(stems_5)

#Creating 200 documents from each book. 
#Each document contains 150 words    
doc_1 = []
doc_2 = []
doc_3 = []
doc_4 = []
doc_5 = []

for i in range(0,200):
    for j in range(0,151):
        data_1 = (random.sample(stems_1, j))
    for j in range(0,151):
        data_2 = (random.sample(stems_2, j))
    for j in range(0,151):
        data_3 = (random.sample(stems_3, j))
    for j in range(0,151):
        data_4 = (random.sample(stems_4, j))
    for j in range(0,151):
        data_5 = (random.sample(stems_5, j))
    
    doc_1.append(' '.join(str(''.join(str(x) for x in v)) for v in data_1))
    doc_2.append(' '.join(str(''.join(str(x) for x in v)) for v in data_2))
    doc_3.append(' '.join(str(''.join(str(x) for x in v)) for v in data_3))
    doc_4.append(' '.join(str(''.join(str(x) for x in v)) for v in data_4))
    doc_5.append(' '.join(str(''.join(str(x) for x in v)) for v in data_5))

# Adding labels to each text of 200 document
df = pd.DataFrame()
df['text']  = doc_1
df['authors'] = 'a'

df1 = pd.DataFrame()
df1['text']  = doc_2
df1['authors'] = 'b'

df2 = pd.DataFrame()
df2['text']  = doc_3
df2['authors'] = 'c'

df3 = pd.DataFrame()
df3['text']  = doc_4
df3['authors'] = 'd'

df4 = pd.DataFrame()
df4['text']  = doc_5
df4['authors'] = 'e'

df5 = pd.concat([df,df1,df2,df3,df4],ignore_index=True) # Combining all dataframe into one.
df8=df5.rename({'text':'booktexts'}, axis='columns') # Renaming of the colomns

# Creating a corpus of all 5 books
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]',' ', df8['booktexts'][i]) #this will only keep letters from A-z and will remove any numbers and puntuation
    review = review.lower()# this will convert all letters to lower cases
    reviewer = " ".join(review.split())# remove white spaces
    corpus.append(reviewer)
# Creating the Bags of Words Model
from sklearn.feature_extraction.text import CountVectorizer  
vectorizer = CountVectorizer()  
Xvec = vectorizer.fit_transform(corpus).toarray()
X_bow = pd.DataFrame(Xvec)
Y = vectorizer.get_feature_names()

yvec = df8.iloc[:, 1]
yvecs = pd.DataFrame(yvec)
#Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
yvec=labelencoder_y.fit_transform(yvec)

#Td-if
X_tf = TfidfVectorizer()
X_tfd =  X_tf.fit_transform(corpus).toarray()
X_tfd = pd.DataFrame(X_tfd)


# Applying LDA
from sklearn.decomposition import  LatentDirichletAllocation as LDA
lda = LDA(n_components = 10, max_iter=10, n_jobs = -1, batch_size=128, learning_decay=0.5)
lda.fit_transform(X_bow)
X_lda = lda.fit_transform(X_bow)
X_LDa = pd.DataFrame(X_lda)


gaussianbow = mixture.GaussianMixture(n_components = 5, covariance_type = 'full')
gaussianbow.fit(X_bow)
gaussianpredbow = gaussianbow.fit_predict(X_bow)

gaussian = mixture.GaussianMixture(n_components = 5, covariance_type = 'full')
gaussian.fit(X_tfd)
gaussianpred = gaussian.fit_predict(X_tfd)

gaussianlda = mixture.GaussianMixture(n_components = 5, covariance_type = 'full')
gaussianlda.fit(X_lda)
gaussianpredlda = gaussian.fit_predict(X_lda)


Xbow_cluster = X_bow.iloc[:, 0:X_bow.size].values
Xtf_cluster = X_tfd.iloc[:, 0:X_tfd.size].values
Xlda_cluster = X_LDa.iloc[:, 0:X_lda.size].values

'''kmeans'''
from sklearn.cluster import KMeans

#kmeans with BOW
modelkmeansbow = KMeans(n_clusters=5)
modelkmeansbow.fit(Xbow_cluster)
y_kmeansbow = modelkmeansbow.fit_predict(Xbow_cluster)
y_kmeansbows = pd.DataFrame(y_kmeansbow)

#kmeans with TF
modelkmeanstf = KMeans(n_clusters=5)
modelkmeanstf.fit(Xtf_cluster)
y_kmeanstf = modelkmeanstf.fit_predict(Xtf_cluster)
y_kmeanstfs = pd.DataFrame(y_kmeanstf)


#kmeans with LDA
modelkmeanslda = KMeans(n_clusters=5)
modelkmeanslda.fit(X_LDa)
y_kmeanslda = modelkmeanslda.fit_predict(X_lda)
y_kmeanstfs = pd.DataFrame(y_kmeanstf)

order_centroidsbow = modelkmeansbow.cluster_centers_.argsort()[:, ::-1]
termsbow = vectorizer.get_feature_names()

order_centroidstf = modelkmeanstf.cluster_centers_.argsort()[:, ::-1]
termstf = vectorizer.get_feature_names()

order_centroidslda = modelkmeanslda.cluster_centers_.argsort()[:, ::-1]
termslda = vectorizer.get_feature_names()

# Fitting Hierarchical Clustering to the dataset for BOW
from sklearn.cluster import AgglomerativeClustering
hcbow = AgglomerativeClustering(n_clusters=5, affinity = 'euclidean', linkage = 'ward')
y_hcbow = hcbow.fit_predict(Xbow_cluster)

# Fitting Hierarchical Clustering to the dataset for TF-IDF
from sklearn.cluster import AgglomerativeClustering
hctf = AgglomerativeClustering( n_clusters=5, affinity = 'euclidean', linkage = 'ward')
y_hctf = hctf.fit_predict(Xtf_cluster)

# Fitting Hierarchical Clustering to the dataset for LDA
from sklearn.cluster import AgglomerativeClustering
hclda = AgglomerativeClustering( n_clusters=5, affinity = 'euclidean', linkage = 'ward')
y_hclda = hclda.fit_predict(Xlda_cluster)



'''pyLDAvis'''
texts = [[word for word in document.lower().split() if word not in stop_words]

          for document in corpus]
all_tokens = sum(texts, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts = [[word for word in text if word not in tokens_once]
          for text in texts]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
lda = models.LdaModel(corpus, id2word=dictionary, num_topics=10,passes=10)
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda,corpus,dictionary)
vis


'''PCA plot'''

#kmeansbowPCA
from sklearn.decomposition import PCA
bow_pca = PCA(n_components=2)
BowComponents = bow_pca.fit_transform(X_bow)
BowDf = pd.DataFrame(data = BowComponents, columns = ['bow component 1', 'bow component 2'])

bow_centers = bow_pca.transform(modelkmeansbow.cluster_centers_)

n = BowComponents.shape[0]
centers_old = np.zeros(bow_centers.shape) # to store old centers
centers_new = deepcopy(bow_centers)

clusters = np.zeros(n)
distances = np.zeros((n,5))

error = np.linalg.norm(centers_new - centers_old)

while error != 0:
    # Measure the distance to every center
    for i in range(5):
        distances[:,i] = np.linalg.norm(BowComponents - bow_centers[i], axis=1)
    # Assign all training data to closest center
    clusters = np.argmin(distances, axis = 1)
    centers_old = deepcopy(centers_new)
    # Calculate mean for every cluster and update the center
    for i in range(5):
        centers_new[i] = np.mean(BowComponents[clusters == i], axis=0)
    error = np.linalg.norm(centers_new - centers_old)
centers_new 
centers_new = centers_new

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Bow Component 1', fontsize = 10)
ax.set_ylabel('Bow Component 2', fontsize = 10)
ax.set_title('2Bow component PCA', fontsize = 15)

colors = ['y', 'b', 'orange', 'c', 'r']  
for color, i, target_name in zip(colors, [0, 1, 2, 3, 4], selected_titles):
    ax.scatter(BowComponents[yvec == i, 0], BowComponents[yvec == i, 1], alpha=.8, color=color,
                label=target_name, s = 10)
ax.legend(loc='best', shadow=False, scatterpoints=1)
ax.scatter(centers_new[:,0], centers_new[:,1], marker='*', c='k', label = 'centroid', s=150)
ax.grid()

#kmeanstfPCA
tf_pca = PCA(n_components=2)
tfComponents = tf_pca.fit_transform(X_tfd)

tf_centers = tf_pca.transform(modelkmeanstf.cluster_centers_)

n = tfComponents.shape[0]
centers_old = np.zeros(tf_centers.shape) # to store old centers
centers_new = deepcopy(tf_centers)

clusters = np.zeros(n)
distances = np.zeros((n,5))

error = np.linalg.norm(centers_new - centers_old)

while error != 0:
    # Measure the distance to every center
    for i in range(5):
        distances[:,i] = np.linalg.norm(tfComponents - tf_centers[i], axis=1)
    # Assign all training data to closest center
    clusters = np.argmin(distances, axis = 1)
    centers_old = deepcopy(centers_new)
    # Calculate mean for every cluster and update the center
    for i in range(5):
        centers_new[i] = np.mean(tfComponents[clusters == i], axis=0)
    error = np.linalg.norm(centers_new - centers_old)
centers_new 
centers_new = centers_new

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('tf Component 1', fontsize = 10)
ax.set_ylabel('tf Component 2', fontsize = 10)
ax.set_title('2tf component PCA', fontsize = 15)

colors = ['y', 'b', 'orange', 'c', 'r']  
for color, i, target_name in zip(colors, [0, 1, 2, 3, 4], selected_titles):
    ax.scatter(tfComponents[yvec == i, 0], tfComponents[yvec == i, 1], alpha=.8, color=color,
                label=target_name, s = 10)
ax.legend(loc='best', shadow=False, scatterpoints=1)
ax.scatter(centers_new[:,0], centers_new[:,1], marker='*', c='k', label = 'centroid', s=150)
ax.grid()

print("Top terms per cluster:")


'''WordCloud'''

#bow
clusterz = []
order_centroids = modelkmeansbow.cluster_centers_.argsort()[:, ::-1]
for i in range(5):
    #print ("Cluster %d:" % i)
    clust = []
    for ind in order_centroids[i, :30]:
        clust.append(Y[ind])
    
    clusterz.append(' '.join(str(''.join(str(x) for x in v)) for v in clust))

for i in range(len(clusterz)):
    clusterwords = "".join(str(x) for x in clusterz[i])
    wordcloud = WordCloud(max_font_size = 60).generate(clusterwords)
    plt.figure(figsize=(16,12))
    plt.imshow(wordcloud, interpolation = "bilinear")
    plt.axis("off")
    plt.show()

#tf
clusterztf = []
order_centroids = modelkmeanstf.cluster_centers_.argsort()[:, ::-1]
for i in range(5):
    #print ("Cluster %d:" % i)
    clusttf = []
    for ind in order_centroids[i, :30]:
        clusttf.append(Y[ind])
    
    clusterztf.append(' '.join(str(''.join(str(x) for x in v)) for v in clusttf))

for i in range(len(clusterztf)):
    clusterwordstf = "".join(str(x) for x in clusterztf[i])
    wordcloudtf = WordCloud(max_font_size = 60).generate(clusterwordstf)
    plt.figure(figsize=(16,12))
    plt.imshow(wordcloudtf, interpolation = "bilinear")
    plt.axis("off")
    plt.show()



'''evaluation'''

#silhouette
from sklearn.metrics import silhouette_score
silhouette_bow_kmeans = (silhouette_score(Xbow_cluster, y_kmeansbow, metric='euclidean'))
silhouette_tf_kmeans = (silhouette_score(Xtf_cluster, y_kmeanstf, metric='euclidean'))
silhouette_lda_kmeans = (silhouette_score(Xlda_cluster, y_kmeanslda, metric='euclidean'))
silhouette_agg_bow = (silhouette_score(Xbow_cluster, y_hcbow, metric='euclidean'))
silhouette_agg_tf = (silhouette_score(Xtf_cluster, y_hctf, metric='euclidean'))
silhouette_agg_lda = (silhouette_score(Xlda_cluster, y_hclda, metric='euclidean'))
silhouette_gaubow = (silhouette_score(Xbow_cluster, gaussianpredbow, metric='euclidean'))
silhouette_gautf = (silhouette_score(Xtf_cluster, gaussianpred, metric='euclidean'))
silhouette_gaulda = (silhouette_score(Xlda_cluster, gaussianpredlda, metric='euclidean'))

#Bar graph
from sklearn.metrics import cohen_kappa_score
kappa_bow_kmeans=cohen_kappa_score(y_kmeansbow, yvec)
kappa_tf_kmeans=cohen_kappa_score(y_kmeanstf, yvec)
kappa_lda_kmeans=cohen_kappa_score(y_kmeanslda, yvec)
kappa_agg_bow=cohen_kappa_score(y_hcbow, yvec)
kappa_agg_tf=cohen_kappa_score(y_hctf, yvec)
kappa_agg_lda=cohen_kappa_score(y_hclda, yvec)
kappa_gau_bow=cohen_kappa_score(gaussianpredbow, yvec)
kappa_gau_tf=cohen_kappa_score(gaussianpred, yvec)
kappa_gau_lda=cohen_kappa_score(gaussianpredlda, yvec)

sil_df = [silhouette_bow_kmeans,silhouette_tf_kmeans,silhouette_lda_kmeans,silhouette_agg_bow,silhouette_agg_tf, silhouette_agg_lda, silhouette_gaubow, silhouette_gautf, silhouette_gaulda]
kappa_df = [kappa_bow_kmeans,kappa_tf_kmeans,kappa_lda_kmeans,kappa_agg_bow,kappa_agg_tf,kappa_agg_lda,kappa_gau_bow,kappa_gau_tf,kappa_gau_lda]
 
N = 9
indi = np.arange(N)
 
plt.bar(indi, sil_df, width = 0.3, color='r', align = 'center', label = 'silhoutte')
plt.bar(indi+0.3, kappa_df, width = 0.3, color='g', align = 'center', label = 'kappa')
 
plt.xticks(indi+0.3)
plt.xticks(np.arange(N),('Bow_means', 'Tf_means', 'Lda_means', 'Aggbow', 'Aggtf', 'Agglda', 'Gau_bow', 'Gau_tf', 'Gau_lda'))
plt.legend()

#Kappa score
print('kappa for bow and kmeans is %s ' %cohen_kappa_score(y_kmeansbow, yvec))
print('kappa for tf and kmeans is %s '%cohen_kappa_score(y_kmeanstf, yvec))
print('kappa for lda and kmeans is %s '%cohen_kappa_score(y_kmeanslda, yvec))
print('kappa for bow and agg is %s '%cohen_kappa_score(y_hcbow, yvec))
print('kappa for tf and agg is %s '%cohen_kappa_score(y_hctf, yvec))
print('kappa for lda and agg is %s '%cohen_kappa_score(y_hclda, yvec))
print('kappa for bow and gauss is %s '%cohen_kappa_score(gaussianpredbow, yvec))
print('kappa for tf and gauss is %s '%cohen_kappa_score(gaussianpred, yvec))
print('kappa for lda and gauss is %s '%cohen_kappa_score(gaussianpredlda, yvec))

#ARI
adjusted_rand_score(yvec, y_kmeanstf)
adjusted_rand_score(yvec, y_kmeansbow)
adjusted_rand_score(yvec, y_kmeanslda)

adjusted_rand_score(yvec, gaussianpred)
adjusted_rand_score(yvec, gaussianpredbow)
adjusted_rand_score(yvec, gaussianpredlda)

adjusted_rand_score(yvec, y_hctf)
adjusted_rand_score(yvec, y_hcbow)
adjusted_rand_score(yvec, y_hclda)
