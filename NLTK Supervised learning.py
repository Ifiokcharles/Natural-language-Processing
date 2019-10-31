import matplotlib.pyplot as plt
import random
import nltk
import re
import pandas as pd
import sklearn.datasets
from wordcloud import WordCloud
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('gutenberg')
from nltk.corpus import gutenberg 

files_en = gutenberg.fileids()
selected_titles = ['3623-8.txt','19528-8.txt','24681-8.txt',
                   '29444-8.txt', 'milton-paradise.txt']
label_titles = ['a','b','c','d','e']
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
yvec = df8.iloc[:, 1]
X_bow = pd.DataFrame(Xvec)
Y = vectorizer.get_feature_names()

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
yvec=labelencoder_y.fit_transform(yvec)

#Td-if
X_tf = TfidfVectorizer()
X_tfd =  X_tf.fit_transform(corpus).toarray()
X_tfd = pd.DataFrame(X_tfd)

dataset_train = sklearn.datasets.base.Bunch(data=corpus, 
target_names=selected_titles,target=yvec, subset='train', 
shuffle=True,random_state=42)

doc_vec = pd.DataFrame(dataset_train.target)

#Machine learning tecnques
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
Xbow_train, Xbow_test, ybow_train, ybow_test = train_test_split(X_bow, doc_vec, test_size = 0.30, random_state = 100)#bow
ybow_train = pd.Series(ybow_train.iloc[:, 0])
ybow_test = pd.Series(ybow_test.iloc[:, 0])

Xtf_train, Xtf_test, ytf_train, ytf_test = train_test_split(X_tfd, doc_vec, test_size = 0.30, random_state = 100)#tf
ytf_train = pd.Series(ytf_train.iloc[:, 0])
ytf_test = pd.Series(ytf_test.iloc[:, 0])

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifierbow = SVC(kernel = 'rbf', random_state = 0, C=1, gamma = 0.01)
classifiertf = SVC(kernel = 'linear', random_state = 0, C=1)
#fit to traning set
classifierbow.fit(Xbow_train, ybow_train)
y_predbow = classifierbow.predict(Xbow_test)

classifiertf.fit(Xtf_train, ytf_train)
y_predtf = classifiertf.predict(Xtf_test)

#applying Ten fold cross validation
from sklearn.model_selection import cross_val_score
accuraciesbow = cross_val_score(estimator = classifierbow, X = Xbow_train, y = ybow_train, cv = 10, n_jobs = -1)
opbow = accuraciesbow.mean()

accuraciestf = cross_val_score(estimator = classifiertf, X = Xtf_train, y = ytf_train, cv = 10, n_jobs = -1)
optf = accuraciestf.mean()

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifierNBbow = GaussianNB()
classifierNBtf = GaussianNB()

classifierNBbow.fit(Xbow_train, ybow_train)
classifierNBtf.fit(Xtf_train, ytf_train)
# Predicting the Test set results
y_predNBbow = classifierNBbow.predict(Xbow_test)
y_predNBtf = classifierNBtf.predict(Xtf_test)

accuraciesNBbow = cross_val_score(estimator = classifierNBbow, X = Xbow_train, y = ybow_train, cv = 10, n_jobs = -1)
opNBbow = accuraciesNBbow.mean()

accuraciesNBtf = cross_val_score(estimator = classifierNBtf, X = Xtf_train, y = ytf_train, cv = 10, n_jobs = -1)
opNBtf = accuraciesNBtf.mean()

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifierCTbow = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifierCTtf = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifierCTbow.fit(Xbow_train, ybow_train)
classifierCTtf.fit(Xtf_train, ytf_train)

# Predicting the Test set results
y_predCTbow = classifierCTbow.predict(Xbow_test)
y_predCTtf = classifierCTtf.predict(Xtf_test)

accuraciesCTbow = cross_val_score(estimator = classifierCTbow, X = Xbow_train, y = ybow_train, cv = 10, n_jobs = -1)
opCTbow = accuraciesCTbow.mean()

accuraciesCTtf = cross_val_score(estimator = classifierCTtf, X = Xtf_train, y = ytf_train, cv = 10, n_jobs = -1)
opCTtf = accuraciesCTtf.mean()

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifierKNNbow = KNeighborsClassifier(n_neighbors = 5)
classifierKNNtf = KNeighborsClassifier(n_neighbors = 5)
classifierKNNbow.fit(Xbow_train, ybow_train)
classifierKNNtf.fit(Xtf_train, ytf_train)

# Predicting the Test set results
y_predKNNbow = classifierKNNbow.predict(Xbow_test)
y_predKNNtf = classifierKNNtf.predict(Xtf_test)

accuraciesKNNbow = cross_val_score(estimator = classifierKNNbow, X = Xbow_train, y = ybow_train, cv = 10, n_jobs = -1)
opKNNbow = accuraciesKNNbow.mean()

accuraciesKNNtf = cross_val_score(estimator = classifierKNNtf, X = Xtf_train, y = ytf_train, cv = 10, n_jobs = -1)
opKNNtf = accuraciesKNNtf.mean()

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmknn = confusion_matrix(ybow_test, y_predKNNbow)
cmknn1 = confusion_matrix(ytf_test, y_predKNNtf)
Matrix_KNN_bow = (metrics.classification_report(ybow_test, y_predKNNbow))
Matrix_KNN_tf = (metrics.classification_report(ytf_test, y_predKNNtf))


cmdt = confusion_matrix(ybow_test, y_predCTbow)
cmdt1 = confusion_matrix(ytf_test, y_predCTtf)
Matrix_dt_bow = (metrics.classification_report(ybow_test, y_predCTbow))
Matrix_dt_tf = (metrics.classification_report(ytf_test, y_predCTtf))

cmNB = confusion_matrix(ybow_test, y_predNBbow)
cmNB1 = confusion_matrix(ytf_test, y_predNBtf)
Matrix_NB_bow = (metrics.classification_report(ybow_test, y_predNBbow))
Matrix_NB_tf = (metrics.classification_report(ytf_test, y_predNBtf))

cmSM = confusion_matrix(ybow_test, y_predbow)
cmSVM1 = confusion_matrix(ytf_test, y_predtf)
Matrix_SM_bow = (metrics.classification_report(ybow_test, y_predbow))
Matrix_SVM_tf = (metrics.classification_report(ytf_test, y_predtf))

misclasses=[] #finding the misclassified author
mislocs=[]
i = 0
for index in ybow_test:
    if index != y_predCTtf[i] and cmdt1[index, y_predCTtf[i]]>=1:
        misclasses.append("'{}' predicted wrongly as '{}'".format(index, y_predCTtf[i]))
        mislocs.append(ybow_test.index[i])
    i+=1

Error_locations = pd.DataFrame({'Document': mislocs, 'Misclassification': misclasses})

   
rowX = int(Xbow_test.size/len(Xbow_test))


#looking for similar words between misclassified document and other documents
save1=[]
save2=[]
for index, row in Xbow_test.iterrows():
    if index == mislocs[0]:
        for k in range(0,rowX):
            save1.append(row[k])

for index, row in Xbow_test.iterrows():
    if index in range (800,1000):
        for k in range(0,rowX):
            save2.append(row[k])

author0=[]
author1=[]

author0 = save1
    
while save2 != []:
    author1.append(save2[:rowX])
    save2 = save2[rowX:]
    
    
k=0
find_error=[]
for index in author1:
    for i in range (0,rowX):
        if author0[i] > 0 and author1[k][i]>0:
            find_error.append(i)
    k+=1
       
save_occuringword=[]
for index in find_error:
    for j in range (0, rowX):
        if index == j:
            save_occuringword.append(Y[j])
    
occuring=[]
occuring.append(' '.join(str(''.join(str(x) for x in v)) for v 
                                      in save_occuringword))

from sklearn.feature_extraction.text import CountVectorizer
wordcount = CountVectorizer()
occurence = wordcount.fit_transform(occuring).toarray()
similarwords = pd.Series(wordcount.get_feature_names())
occurence = occurence.T
occurence = pd.Series(occurence[:, 0])
Frequencyofwords = pd.DataFrame({'Words':similarwords, 'Frequency':occurence})

#Visualizations
reoccuring = " ".join(str(x) for x in occuring)#word cloud plot for particular document where miss classification occurs
cloud = WordCloud(max_font_size=60).generate(reoccuring)
plt.figure(figsize=(16,12))
plt.imshow(cloud, interpolation="bilinear")
plt.axis("off")
plt.show()

freqdist = nltk.FreqDist(save_occuringword)#Find the freq distribution
plt.figure(figsize=(20,5))
freqdist.plot(100)

docsforauthor = corpus[800:1000]#Generating a specific author 
combinedocs=[]
combinedocs.append(' '.join(str(''.join(str(x) for x in v)) for v in docsforauthor))
stringcombinedocs = " ".join(str(x) for x in combinedocs)
separatedocs = stringcombinedocs.split(' ')

count=0
savecount=[]
occuringforspecificauthors=[]
for index in similarwords:
    for indexes in separatedocs:
       if index == indexes:
            count+=1
            occuringforspecificauthors.append(indexes)
    savecount.append(count)
    count=0

random.shuffle(occuringforspecificauthors)

#Visualizations for specific author
reoccuringforspecificauthor = " ".join(str(x) for x in occuringforspecificauthors)#word cloud plot for specific author
cloudauthor = WordCloud(max_font_size=60).generate(reoccuringforspecificauthor)
plt.figure(figsize=(16,12))
plt.imshow(cloudauthor, interpolation="bilinear")
plt.axis("off")
plt.show()

freqdist = nltk.FreqDist(occuringforspecificauthors)#find the freq distribution
plt.figure(figsize=(20,5))
freqdist.plot(100)



#ten fold cross validation plot
n = [0,1,2,3,4,5,6,7,8,9]        
plt.figure()
plt.plot(n, accuraciesKNNbow, 'r-', marker = 'o', label = 'knn_bow')
plt.plot(n, accuraciesKNNtf, 'm-', marker = '*', label = 'knn_tf')
plt.plot(n, accuraciesbow, 'k-', marker = '+', label = 'svm_bow')
plt.plot(n, accuraciestf, 'y-', marker = 'x', label = 'svm_grid_tf')
plt.plot(n, accuraciesCTbow, 'g-', marker = '^', label = 'tree_bow')
plt.plot(n, accuraciesCTtf, 'b-', marker = 'd', label = 'tree_tf')
plt.xlabel('ten fold')
plt.ylabel('accuracy')
plt.legend(loc='upper left', bbox_to_anchor=(1, 0.8), shadow=True, ncol=1)
plt.title('Ten fold cross validation')



plt.figure()#The actual vs predictions plot
plt.scatter(ytf_test, y_predCTtf, edgecolors=(0,0,0))
plt.plot([doc_vec.min(), doc_vec.max()], [doc_vec.min(), doc_vec.max()], 'b--', lw=4)
plt.xlabel('Actual')
plt.ylabel('Predictions')







