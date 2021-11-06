pip install bnlp_toolkit

!pip install bangla-stemmer

from bangla_stemmer.stemmer import stemmer

__stopwords = ["টা","অবশ্য","অনেক","অনেকে","অনেকেই","অন্তত","অথবা","অথচ","অর্থাত","অন্য","আজ","আছে","আপনার","আপনি","আবার","আমরা","আমাকে","আমাদের"
             ,"আমার","আমি","আরও","আর","আগে","আগেই","আই","অতএব","আগামী","অবধি","অনুযায়ী","আদ্যভাগে","এই","একই","একে","একটি","এখন","এখনও"
             ,"এখানে","এখানেই","এটি","এটা","এটাই","এতটাই","এবং","একবার","এবার","এদের","এঁদের","এমন","এমনকী","এল","এর","এরা","এঁরা","এস","এত"
             ,"এতে","এসে","একে","এ","ঐ"," ই","ইহা","ইত্যাদি","উনি","উপর","উপরে","উচিত","ও","ওই","ওর","ওরা","ওঁর","ওঁরা","ওকে","ওদের","ওঁদের",
             "ওখানে","কত","কবে","করতে","কয়েক","কয়েকটি","করবে","করলেন","করার","কারও","করা","করি","করিয়ে","করার","করাই","করলে","করলেন",
             "করিতে","করিয়া","করেছিলেন","করছে","করছেন","করেছেন","করেছে","করেন","করবেন","করায়","করে","করেই","কাছ","কাছে","কাজে","কারণ","কিছু",
             "কিছুই","কিন্তু","কিংবা","কি","কী","কেউ","কেউই","কাউকে","কেন","কে","কোনও","কোনো","কোন","কখনও","ক্ষেত্রে","খুব	গুলি","গিয়ে","গিয়েছে",
             "গেছে","গেল","গেলে","গোটা","চলে","ছাড়া","ছাড়াও","ছিলেন","ছিল",'ছিলো',"জন্য","জানা","ঠিক","তিনি","তিনঐ","তিনিও","তখন","তবে","তবু","তাঁদের",
             "তাঁাহারা","তাঁরা","তাঁর","তাঁকে","তাই","তেমন","তাকে","তাহা","তাহাতে","তাহার","তাদের","তারপর","তারা","তারৈ","তার","তাহলে","তিনি","তা",
             "তাও","তাতে","তো","তত","তুমি","তোমার","তথা","থাকে","থাকা","থাকায়","থেকে","থেকেও","থাকবে","থাকেন","থাকবেন","থেকেই","দিকে","দিতে",
             "দিয়ে","দিয়েছে","দিয়েছেন","দিলেন","দু","দুটি","দুটো","দেয়","দেওয়া","দেওয়ার","দেখা","দেখে","দেখতে","দ্বারা","ধরে","ধরা","নয়","নানা","না",
             "নাকি","নাগাদ","নিতে","নিজে","নিজেই","নিজের","নিজেদের","নিয়ে","নেওয়া","নেওয়ার","নেই","নাই","পক্ষে","পর্যন্ত","পাওয়া","পারেন","পারি","পারে",
             "পরে","পরেই","পরেও","পর","পেয়ে","প্রতি","প্রভৃতি","প্রায়","ফের","ফলে","ফিরে","ব্যবহার","বলতে","বললেন","বলেছেন","বলল","বলা","বলেন","বলে",
             "বহু","বসে","বার","বা","বিনা","বরং","বদলে","বাদে","বার","বিশেষ","বিভিন্ন	বিষয়টি","ব্যবহার","ব্যাপারে","ভাবে","ভাবেই","মধ্যে","মধ্যেই","মধ্যেও",
             "মধ্যভাগে","মাধ্যমে","মাত্র","মতো","মতোই","মোটেই","যখন","যদি","যদিও","যাবে","যায়","যাকে","যাওয়া","যাওয়ার","যত","যতটা","যা","যার","যারা",
             "যাঁর","যাঁরা","যাদের","যান","যাচ্ছে","যেতে","যাতে","যেন","যেমন","যেখানে","যিনি","যে","রেখে","রাখা","রয়েছে","রকম","শুধু","সঙ্গে","সঙ্গেও",
             "সমস্ত","সব","সবার","সহ","সুতরাং","সহিত","সেই","সেটা","সেটি","সেটাই","সেটাও","সম্প্রতি","সেখান","সেখানে","সে","স্পষ্ট","স্বয়ং","হইতে","হইবে",
             "হৈলে","হইয়া","হচ্ছে","হত","হতে","হতেই","হবে","হবেন","হয়েছিল","হয়েছে","হয়েছেন","হয়ে","হয়নি","হয়","হয়েই","হয়তো","হল","হলে","হলেই","হলেও",
             "হলো","হিসাবে","হওয়া","হওয়ার","হওয়ায়","হন","হোক","জন","জনকে","জনের","জানতে","জানায়","জানিয়ে","জানানো","জানিয়েছে","জন্য","জন্যওজে",
             "জে","বেশ","দেন","তুলে","ছিলেন","চান","চায়","চেয়ে","মোট","যথেষ্ট","টি","দু","একটি","নিজের","তারৈ","আমি","ঐ","আপনি","করিয়ে","তত","জন্য","যখন",
             "হত","সেটাও","করার","ওঁদের","শুধু","তাহার","ওদের","দেওয়ার","নিজেই","হল","হলে","হলেই","হলেও","হলো","হাজার","হিসাবে","হৈলে","হোক","হয়",
              "আমার","দিলেন","ফিরে","গেলে","জানা","আপনার","তাঁর","উপর","তাকে","রয়েছে","যাকে","এঁরা","তাদের","সেই","হবেন","কোনও","অনুযায়ী","যান","তাও",
              "পরেও","গেছে","অবধি","কয়েকটি","কাছে","এটি","আগেই","এতটাই","হইয়া","যা","হৈলে","আবার","তারা","সে","হয়েছে","সহিত","যাবে","তখন","গিয়েছে","দিয়ে",
              "কিছুই","তবে","নিতে","রেখে"," ই","সহ","যাঁরা","নানা","হলো","যাঁর","তোমার","পর","ছাড়াও","করলে","যত","তবু","তিনিও","না","দেখতে","দেওয়া","থেকেও","কাজে",
              "ক্ষেত্রে","কয়েক","হচ্ছে","হয়েছিল","থেকেই","অথবা","সঙ্গেও","বদলে","দ্বারা","পক্ষে","গেল","বলতে","পাওয়া","কত","মধ্যে","বলা","জে","নেই","তাই","কি","সেটা","একে",
              "যেখানে","এত","হলেও","টি","করেই","করছে","হন","প্রায়","মধ্যভাগে","কারণ","এবার","করেছে","করেন","আর","যেন","নিজেদের","হয়েই","নিজে","একবার","নাই",
              "বাদে","যাতে","এর","ঠিক","তার","ও","পেয়ে","করলেন","মোট","ব্যাপারে","কাছ","করা","চেয়ে","কেউ","নাগাদ","করি","বলেছেন","নেওয়ার","কাউকে","ভাবে","দিকে",
              "তারপর","যেমন","ওখানে","খুব","গুলি","অর্থাত","তো","ছিলেন","কোন","পারেন","হয়তো","বরং","কেউই","জনকে","প্রভৃতি","দুটো","তাঁকে","এখন","অন্য","ওর","ছিল",
              "ওকে","তুলে","দিয়েছে","জানানো","ওঁরা","এটাই","তুমি","করিতে","তাহলে","দেন","বলে","যে","হলেই","এমনকী","হল","বহু","বলল","মধ্যেই","ধরে","তাঁদের","তেমন","আই",
              "হইবে","তাহাতে","নেওয়া","যিনি","এঁদের","অনেকে","হতে","কে","ধরা","হইতে","করায়","ব্যবহার","থাকে","বসে","থাকেন","থাকবে","স্বয়ং","এরা","দেয়","নিয়ে","কবে",
              "সবার","দেখে","চলে","যেতে","ইত্যাদি","সেখান","চান","অন্তত","হবে","সেটাই","পর্যন্ত","মাধ্যমে","এমন","ভাবেই","দিয়েছেন","ওরা","করে","তাতে","এবং","এতে","ইহা",
              "জন্যওজে","সুতরাং","আমাকে","বিশেষ","এসে","করতে","এখানেই","আমরা","কিন্তু","তিনি","বিনা","আজ","কারও","করিয়া","তা","ছাড়া","থেকে","যারা","হয়","হওয়া","এল",
              "মাত্র","ফের","জানতে","জানিয়ে","বললেন","মতোই","সাথে","কর","করেছেন","করবেন","হলে","নাকি","সঙ্গে","আগামী","এখনও","তাঁাহারা","দিতে","তাঁরা","আগে","আমাদের",
              "সেটি","বলেন","স্পষ্ট","কোনো","হোক","থাকবেন","জন","করছেন","অবশ্য","গিয়ে","হয়নি","এখানে","করবে","কিছু","হওয়ায়","কখনও","যাদের","বার","হয়ে","পারি","জানিয়েছে",
              "আদ্যভাগে","আরও","মতো","যায়","যাওয়ার","কিংবা","যদি","পরেই","জনের","হিসাবে","এস","দুটি","জানায়","গোটা","যাওয়া","তথা","সমস্ত","যদিও","করাই","হতেই","হয়েছেন",
              "নয়","বিভিন্ন","বিষয়টি","রকম","অনেক","করেছিলেন","উপরে","এ","এদের","উনি","হয়","সব","পরে","প্রতি","যার","মধ্যেও","মোটেই","এই","বা","বেশ","পারে","যতটা","অনেকেই",
              "যাচ্ছে","অথচ","অতএব","একই","দেখা","চায়","আছে","থাকায়","যথেষ্ট","কী","তাহা","রাখা","ওঁর","সেখানে","সম্প্রতি","তিনঐ","উচিত","হওয়ার","ফলে","ওই","কেন","থাকা","এটা",
              "অতএব","অথচ","অথবা","অনুযায়ী","অনেক","অনেকে","অনেকেই","অন্তত","অন্য","অবধি","অবশ্য","অর্থাত","আই","আগামী","আগে","আগেই","আছে","আজ","আদ্যভাগে","আপনার",
              "আপনি","আবার","আমরা","আমাকে","আমাদের","আমার","আমি","আর","আরও","ই","ইত্যাদি","ইহা","উচিত","উত্তর","উনি","উপর","উপরে","এ","এঁদের","এঁরা","এই","একই",
              "একটি","একবার","একে","এক্","এখন","এখনও","এখানে","এখানেই","এটা","এটাই","এটি","এত","এতটাই","এতে","এদের","এব","এবং","এবার","এমন","এমনকী","এমনি","এর",
              "এরা","এল","এস","এসে","ঐ","ও","ওঁদের","ওঁর","ওঁরা","ওই","ওকে","ওখানে","ওদের","ওর","ওরা","কখনও","কত","কবে","কমনে","কয়েক","কয়েকটি","করছে","করছেন","করতে",
              "করবে","করবেন","করলে","করলেন","করা","করাই","করায়","করার","করি","করিতে","করিয়া","করিয়ে","করে","করেই","করেছিলেন","করেছে","করেছেন","করেন","কাউকে","কাছ",
              "কাছে","কাজ","কাজে","কারও","কারণ","কি","কিংবা","কিছু","কিছুই","কিন্তু","কী","কে","কেউ","কেউই","কেখা","কেন","কোটি","কোন","কোনও","কোনো","ক্ষেত্রে","কয়েক","খুব",
              "গিয়ে","গিয়েছে","গিয়ে","গুলি","গেছে","গেল","গেলে","গোটা","চলে","চান","চায়","চার","চালু","চেয়ে","চেষ্টা","ছাড়া","ছাড়াও","ছিল","ছিলেন","জন","জনকে","জনের","জন্য",
              "জন্যওজে","জানতে","জানা","জানানো","জানায়","জানিয়ে","জানিয়েছে","জে","জ্নজন","টি","ঠিক","তখন","তত","তথা","তবু","তবে","তা","তাঁকে","তাঁদের","তাঁর","তাঁরা","তাঁাহারা",
              "তাই","তাও","তাকে","তাতে","তাদের","তার","তারপর","তারা","তারৈ","তাহলে","তাহা","তাহাতে","তাহার","তিনঐ","তিনি","তিনিও","তুমি","তুলে","তেমন","তো","তোমার","থাকবে",
              "থাকবেন","থাকা","থাকায়","থাকে","থাকেন","থেকে","থেকেই","থেকেও","দিকে","দিতে","দিন","দিয়ে","দিয়েছে","দিয়েছেন","দিলেন","দু","দুই","দুটি","দুটো","দেওয়া","দেওয়ার","দেওয়া",
              "দেখতে","দেখা","দেখে","দেন","দেয়","দ্বারা","ধরা","ধরে","ধামার","নতুন","নয়","না","নাই","নাকি","নাগাদ","নানা","নিজে","নিজেই","নিজেদের","নিজের","নিতে","নিয়ে","নিয়ে","নেই",
              "নেওয়া","নেওয়ার","নেওয়া","নয়","পক্ষে","পর","পরে","পরেই","পরেও","পর্যন্ত","পাওয়া","পাচ","পারি","পারে","পারেন","পি","পেয়ে","পেয়্র্","প্রতি","প্রথম","প্রভৃতি","প্রযন্ত","প্রাথমিক",
              "প্রায়","প্রায়","ফলে","ফিরে","ফের","বক্তব্য","বদলে","বন","বরং","বলতে","বলল","বললেন","বলা","বলে","বলেছেন","বলেন","বসে","বহু","বা","বাদে","বার","বি","বিনা","বিভিন্ন","বিশেষ",
              "বিষয়টি","বেশ","বেশি","ব্যবহার","ব্যাপারে","ভাবে","ভাবেই","মতো","মতোই","মধ্যভাগে","মধ্যে","মধ্যেই","মধ্যেও","মনে","মাত্র","মাধ্যমে","মোট","মোটেই","যখন","যত","যতটা","যথেষ্ট",
              "যদি","যদিও","যা","যাঁর","যাঁরা","যাওয়া","যাওয়ার","যাওয়া","যাকে","যাচ্ছে","যাতে","যাদের","যান","যাবে","যায়","যার","যারা","যিনি","যে","যেখানে","যেতে","যেন","যেমন","র","রকম",
              "রয়েছে","রাখা","রেখে","লক্ষ","শুধু","শুরু","সঙ্গে","সঙ্গেও","সব","সবার","সমস্ত","সম্প্রতি","সহ","সহিত","সাধারণ","সামনে","সি","সুতরাং","সে","সেই","সেখান","সেখানে","সেটা","সেটাই",
              "সেটাও","সেটি","স্পষ্ট","স্বয়ং","হইতে","হইবে","হইয়া","হওয়া","হওয়ায়","হওয়ার","হচ্ছে","হত","হতে","হতেই","হন","হবে","হবেন","হয়","হয়তো","হয়নি","হয়ে","হয়েই","হয়েছিল","হয়েছে",
              "হয়েছেন",'•', '!', '"', '#', '”', '“', '$', '%', '&', "'", '–', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_',
               '`', '{', '|', '}', '~', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '…', '‘','’','টা','া', 'ো', 'ে', 'ি', 'ী']

x=single_inflections_character = 'া,ো,ে,ি,ী'.split(",")

print(x)

import re
from bnlp.corpus import stopwords, punctuations

# !wget https://www.omicronlab.com/download/fonts/kalpurush.ttf

# !wget https://www.omicronlab.com/download/fonts/Siyamrupali.ttf

"""#Importing Libraries"""

# Commented out IPython magic to ensure Python compatibility.
#Libraries
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
import itertools
import pickle

  
#Feature extraction and splitting
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


#Model
from sklearn.tree import DecisionTreeClassifier
from sklearn import naive_bayes
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import AdaBoostClassifier

#Evaluation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score
from scipy.sparse import csr_matrix
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn import linear_model, tree, ensemble
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



"""**Confusion Matrix Library**"""

def plot_confusion_matrix(cm, classes, extention,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title +" "+ extention)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

"""#Data Spliting Machine"""

class DataSplit():
  
  def __init__(self,df,y):
    self.df=df # Tokenized column
    self.y=y
  
  def split_df(self):
    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.df, self.y, random_state=4,test_size=0.2)
    print(self.x_train.shape)  # 80% Train set
    print(self.x_test.shape)   # 20% Test set
    return self


  def split_val_df(self,mat,y_train):
    self.x_val_train, self.x_val_test, self.y_val_train, self.y_val_test = train_test_split(mat, y_train , random_state=4,test_size=0.12)
    print(self.x_val_train.shape)  # 70% Train_val set
    print(self.x_val_test.shape)     # 10% Test_val set
    return self.x_val_train, self.x_val_test, self.y_val_train, self.y_val_test

class trainable():
  def __init__(self,x_train,x_test,y_train,y_test,x_val_train,x_val_test,y_val_train,y_val_test):
    self.x_train = x_train
    self.x_test = x_test
    self.y_train = y_train
    self.y_test = y_test

    self.x_val_train = x_val_train
    self.x_val_test = x_val_test
    self.y_val_train = y_val_train
    self.y_val_test = y_val_test

"""#Data Preprocessing Mechine

**Data frem preprocess**
"""

class DfPreProcess():

  def __init__(self, df ):
    self.df = df
    



  '''
  Remove null and duplicate rows -------------------------
  '''

  def remove_garbage(self,column_name):
    print("Operation on column: ",column_name )
    print("Number of null values")
    print(self.df.isnull().sum())
    self.df = self.df.dropna()
    dup = self.df[self.df.duplicated([column_name])]
    print("Duplicated rows: ",dup.shape)
    self.df = self.df.drop_duplicates([column_name])
    print("Garbage freedata-set shape: ",self.df.shape)
    print('')

    return self.df

  # ------------------------------------------------------------
  '''
  split fake and real dataframe -------------------------
  '''

  def racism(self):
    self.df_real = self.df[self.df['Label']==1]
    self.df_fake = self.df[self.df['Label']==0]

    return self.df_real,self.df_fake

  # ------------------------------------------------------------

"""**Text Pre-process**"""

class TextPreProcess():
  num_clusters = 2
  num_seeds = 10
  max_iterations = 300
  labels_color_map_predict = {
      0: '#20b2aa', 1: '#ff7373', 
      # 2: '#ffe4e1', 3: '#005073', 4: '#4d0404',
      # 5: '#ccc0ba', 6: '#4700f9', 7: '#f6f900', 8: '#00f91d', 9: '#da8c49'
  }
  labels_color_map_false = {
      0: '#ff0000', 1: '#0000ff', 
      # 2: '#ffe4e1', 3: '#005073', 4: '#4d0404',
      # 5: '#ccc0ba', 6: '#4700f9', 7: '#f6f900', 8: '#00f91d', 9: '#da8c49'
  }
  pca_num_components = 2
  tsne_num_components = 2
  # self.stemmed = self.cleaned_text = none
  
  # cleaned_text=None
  # tf_idf_matrix=None

  # train_cleaned =None
  def __init__(self, train,test, y ):
    self.train = train
    self.test = test
    self.y = y
    self.Y_true= y.to_numpy(dtype="int32")
    


  # stop_words = set(stopwords.words('english'))
  to_remove = ['•', '!', '"', '#', '”', '“', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '…','‘','’']
  # stop_words.update(to_remove)
  # print('Number of stopwords:', len(stop_words))


  '''
  Regular expression -------------------------
  '''

  def clean(self,text):
    text = re.sub('[%s]' % re.escape(punctuations), '', text)
    # text = re.sub('[%s]' % re.escape(to_remove), '', text)
    text = re.sub('[\t\n\r]', '', text)
    text = re.sub('[\-\=\+\*\/\–]', ' ', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text) 
    text = re.sub('\xa0', '', text)
    text = re.sub('[0-9০-৯]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub('\[[^]]*\]', '', text)
    # text = [x for x in text if x not in self.single_inflections_character]
    # text = (" ").join([word for word in text.split() if not word in stop_words])
    text = "".join([char for char in text if not char in self.to_remove])
    return text


  def cleaned_texts(self,data):
    cleaned_text = data.apply(lambda a: self.clean(str(a)))
    return cleaned_text
  # ------------------------------------------------------------

  # Stemmer-----------------------------
  def stemmer_document(self,text):
      
    x=str(text)
    l=x.split()

    stmr = stemmer.BanglaStemmer()
    stm = stmr.stem(l)

    out=' '.join(stm)
    
    return str(out)

  def stemming(self,cleaned_text):
    stemmed = cleaned_text.apply(lambda x: self.stemmer_document(str(x)))
    return stemmed

  
  '''
  Vectorize-------------------------
  '''

  # def vectorizer2(self):
  #   tf_idf_vectorizer = TfidfVectorizer(analyzer="word", use_idf=True, smooth_idf=True, ngram_range=(2, 3))
  #   self.tf_idf_matrix = tf_idf_vectorizer.fit_transform(self.cleaned_text)
  #   return self.tf_idf_matrix
  # --------------------------------------------------------------
  '''
  Vectorize-------------------------
  '''

  def vectorizer_fit_transform(self,stopwords,stemmed):
    # text= self.cleaned_text if self.stemmed.empty else self.stemmed
    self.vector=TfidfVectorizer(use_idf=True,stop_words=stopwords, ngram_range=(1, 1))
    self.vector.fit(stemmed)
    tf_idf_matrix = self.vector.fit_transform(stemmed)
    return tf_idf_matrix
  # --------------------------------------------------------------
  def vectorizer_transform(self,stopwords,stemmed):
    tf_idf_matrix = self.vector.transform(stemmed)
    return tf_idf_matrix
  # --------------------------------------------------------------

  '''
  Clastarize-------------------------
  '''

  def clusterizer(self,tf_idf_matrix):
    clustering_model = KMeans(
        n_clusters=self.num_clusters,
        max_iter=self.max_iterations,
        precompute_distances="auto",
        n_jobs=-1
      )

    self.labels = clustering_model.fit_predict(tf_idf_matrix)
    return self.labels

  # ---------------------------------------

  '''
  Plot-------------------------
  '''

  def plot(self,tf_idf_matrix):
      X = tf_idf_matrix.todense()
      reduced_data = PCA(n_components=self.pca_num_components).fit_transform(X)
      # print reduced_data

      fig, ax = plt.subplots()
      for index, instance in enumerate(reduced_data):
          # print instance, index, labels[index]
          pca_comp_1, pca_comp_2 = reduced_data[index]

          if(self.labels[index]==self.Y_true[index]):
            color = self.labels_color_map_predict[self.labels[index]]
          else:
            color = self.labels_color_map_false[self.Y_true[index]]

          ax.scatter(pca_comp_1, pca_comp_2, c=color)
      plt.show()



      # t-SNE plot
      embeddings = TSNE(n_components=tsne_num_components)
      Y = embeddings.fit_transform(X)
      plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
      return plt.show()
  # -----------------------------------------------------------

  def fit(self,__stopwords):
    self.train_cleaned = self.cleaned_texts(self.train)
    self.train_stemmed=self.stemming(self.train_cleaned)
    self.train_vector=self.vectorizer_fit_transform(__stopwords,self.train_stemmed)

    self.test_cleaned = self.cleaned_texts(self.test)
    self.test_stemmed=self.stemming(self.test_cleaned)
    self.test_vector=self.vectorizer_transform(__stopwords,self.test_stemmed)

"""#Dataset Loading"""

data_path = "/content/loyal_marged_data_4_11_21_sigma_2.xlsx"

data_raw = pd.read_excel(data_path)
data_raw.head(3)

data_raw.shape

# data_raw = data_raw.drop(['Source','Date'], axis=1)
# data_raw.head(3)

# data_raw['Label'] = data_raw['Class'].astype('category').cat.codes
# data_raw['Label'].value_counts()

"""#Pre Process data-set"""

df_preprocess = DfPreProcess(data_raw)
clean_df= df_preprocess.remove_garbage('Statement')
clean_df= df_preprocess.remove_garbage('Title')
clean_df.shape

lc=clean_df['Label'].value_counts()
print(lc)

sns.countplot(clean_df['Label'])

plot = lc.plot.pie(y='Label', figsize=(5, 5),autopct="%.1f%%")

data =  clean_df.iloc[:,:-1]
y = clean_df['Label']

"""#Data Spliting"""

splited_df = DataSplit(data,y).split_df()
# marged = DataSplit(marged_vector,y).split_df()

splited_df.x_train.shape,splited_df.y_train.shape

"""#Pre process text

***Statement***
"""

text_preprocess_statement = TextPreProcess(splited_df.x_train['Statement'],splited_df.x_test['Statement'],splited_df.y_train)
text_preprocess_statement.fit(__stopwords)
# text_preprocess_statement.cleaned_texts()
# text_preprocess_statement.stemming()
# text_preprocess_statement.vectorizer_fit_transform(__stopwords)
# text_preprocess_statement.clusterizer()

text_preprocess_statement.train_vector.shape
hgdcy

text_preprocess_statement.test_vector.shape

train_statement_vector = text_preprocess_statement.train_vector
test_statement_vector = text_preprocess_statement.test_vector
train_statement_vector.shape,test_statement_vector.shape

# text_preprocess_statement.train_vector.shape,y_train.shape

x_val_train_statement, x_val_test_statement, y_val_train, y_val_test = splited_df.split_val_df(text_preprocess_statement.train_vector,splited_df.y_train)

statement = trainable(train_statement_vector, test_statement_vector, splited_df.y_train, splited_df.y_test, x_val_train_statement, x_val_test_statement, y_val_train, y_val_test  )

statement.x_train.shape,statement.x_test.shape,statement.y_train.shape,statement.y_test.shape,statement.x_val_train.shape,statement.x_val_test.shape,statement.y_val_train.shape,statement.y_val_test.shape,

"""***Title***"""

text_preprocess_title = TextPreProcess(splited_df.x_train['Title'],splited_df.x_test['Title'],splited_df.y_train)
text_preprocess_title.fit(__stopwords)
# text_preprocess_title.cleaned_texts()
# text_preprocess_title.stemming()
# text_preprocess_title.vectorizer(__stopwords)
# # text_preprocess_statement.clusterizer()

train_title_vector = text_preprocess_title.train_vector
test_title_vector = text_preprocess_title.test_vector
train_title_vector.shape,test_title_vector.shape

trein_marged_vector= csr_matrix(np.hstack((train_title_vector.todense(),statement.x_train.todense())))
test_marged_vector= csr_matrix(np.hstack((test_title_vector.todense(),statement.x_test.todense())))
trein_marged_vector.shape,test_marged_vector.shape

x_val_train_marded, x_val_test_marded, y_val_train, y_val_test = splited_df.split_val_df(trein_marged_vector,splited_df.y_train)

marged = trainable(trein_marged_vector, test_marged_vector, splited_df.y_train, splited_df.y_test, x_val_train_marded, x_val_test_marded, y_val_train, y_val_test  )

marged.x_train.shape,marged.x_test.shape,marged.y_train.shape,marged.y_test.shape,marged.x_val_train.shape,marged.x_val_test.shape,marged.y_val_train.shape,marged.y_val_test.shape,

"""# Not Sure"""

# text_preprocess_statement.clusterizer()

# text_preprocess_statement.y.to_numpy(dtype="int32")

# text_preprocess_statement.plot()

# data_raw.Statement[2]

# text_preprocess_statement.cleaned_text[2]

# text_preprocess_statement.cleaned_text.head(3)

# text_preprocess_title = TextPreProcess(clean_df['Title'],y)
# text_preprocess_title.cleaned_texts()
# text_preprocess_title.stemming()
# text_preprocess_title.vectorizer(__stopwords)
# # text_preprocess_title.clusterizer()
# title_vector = text_preprocess_title.tf_idf_matrix
# title_vector.shape

# title_vector.shape

# text_preprocess_title.clusterizer()

# marged_vector =np.hstack((title_vector.todense(),statement_vector.todense()))
# marged_vector= csr_matrix(np.hstack((title_vector.todense(),statement_vector.todense())))
# marged_vector.shape

"""#Preprocess text data

# Model Training

***Decision Tree***
"""

def dTree(callback,name):
  # print(callback.x_train.shape)
  # print(callback.y_train.shape)
  dtc = DecisionTreeClassifier(criterion='entropy')
  dtc.fit(callback.x_train, callback.y_train)
  y_pred=dtc.predict(callback.x_test)

  a1 = accuracy_score(callback.y_test, y_pred)
  print("Accuracy :", a1)

  ms_f1 = f1_score(callback.y_test, y_pred)
  print("F1 :", ms_f1)

  ms_pre = precision_score(callback.y_test, y_pred)
  print("Precision :", ms_pre)

  ms_rec = recall_score(callback.y_test, y_pred)
  print("Recall :", ms_rec)


  cm = metrics.confusion_matrix(callback.y_test, y_pred)
  plot_confusion_matrix(cm, classes=['FAKE', 'REAL'],extention='on %s set'%name)
  return dtc

dtc = dTree(statement,'statement')

marged.x_train.shape

dtc_marged=dTree(marged,'marged')

"""***SVM***"""

def svm_machine(callback,name):
  C_list = [0.1, 1, 10, 100]
  gamma_list = [0.001, 0.01, 0.1, 1]

  best_acc = -np.inf

  for C in C_list:
    for gamma in gamma_list:
      svm_test = SVC(C=C, gamma=gamma)
      svm_test.fit(callback.x_val_train, callback.y_val_train)
      predictions = svm_test.predict(callback.x_val_test)
      acc = accuracy_score(callback.y_val_test, predictions)
      if acc > best_acc:
        best_acc = acc
        best_C = C
        best_gamma = gamma

  print("Best Accuricy of val",best_acc)
  print("Best c of val",best_C)
  print("Best gama of val",best_gamma)
  print("")

  svm = SVC(C=best_C, gamma=best_gamma)
  svm.fit(callback.x_train, callback.y_train)

  y_pred = svm.predict(callback.x_test)

  a2 = accuracy_score(callback.y_test, y_pred)
  print("Accuracy :", a2)

  ms_f3 = f1_score(callback.y_test, y_pred)
  print("F1 :", ms_f3)

  ms_pre = precision_score(callback.y_test, y_pred)
  print("Precision :", ms_pre)

  ms_rec = recall_score(callback.y_test, y_pred)
  print("Recall :", ms_rec)

  cm = metrics.confusion_matrix(callback.y_test, y_pred)
  plot_confusion_matrix(cm, classes=['FAKE', 'REAL'],extention='on %s set'%name)
  return svm

svm = svm_machine(statement,'statement')
pickle.dump(svm, open('svmfit.pkl','wb'))

svm_marged = svm_machine(marged,'marged')

"""***Random Forest***"""

def rf_machine(callback,name):
  model_rfm=RandomForestClassifier()

  #fitting the data into model
  model_rfm.fit(callback.x_train,callback.y_train)

  y_pred = model_rfm.predict(callback.x_test)

  a3 = accuracy_score(callback.y_test, y_pred)
  print("Accuracy :", a3)

  ms_f3 = f1_score(callback.y_test, y_pred)
  print("F1 :", ms_f3)

  ms_pre = precision_score(callback.y_test, y_pred)
  print("Precision :", ms_pre)

  ms_rec = recall_score(callback.y_test, y_pred)
  print("Recall :", ms_rec)

  print(f"Fake News Random Forest Model Accuracy : {model_rfm.score(callback.x_test,callback.y_test)*100:.2f}%")

  cm = metrics.confusion_matrix(callback.y_test, y_pred)
  plot_confusion_matrix(cm, classes=['FAKE', 'REAL'],extention='on %s set'%name)
  return model_rfm

model_rfm = rf_machine(statement,'statement')

model_rfm_marged = rf_machine(marged,'marged')

"""***MNB***"""

def mnb_machine(callback,name):
  MNB = naive_bayes.MultinomialNB()

  model1=MNB.fit(callback.x_train, callback.y_train)

  y_pred = model1.predict(callback.x_test)

  a1 = accuracy_score(callback.y_test, y_pred)
  print("Accuracy :", a1)

  ms_f1 = f1_score(callback.y_test, y_pred)
  print("F1 :", ms_f1)

  ms_pre = precision_score(callback.y_test, y_pred)
  print("Precision :", ms_pre)

  ms_rec = recall_score(callback.y_test, y_pred)
  print("Recall :", ms_rec)

  cm = metrics.confusion_matrix(callback.y_test, y_pred)
  plot_confusion_matrix(cm, classes=['FAKE', 'REAL'],extention='on %s set'%name)
  return MNB

MNB = mnb_machine(statement,'statement')

MNB_marged = mnb_machine(marged,'marged')

"""***GaussianNB***"""

from sklearn.naive_bayes import GaussianNB
def gnb_machine(callback):
  model=GaussianNB()
  model.fit(callback.x_train, callback.y_train)

  # model_rfm.fit(callback.x_train,callback.y_train)

  # m4 = model_rfm.predict(callback.x_test)

  # a4 = accuracy_score(callback.y_test, m4)
  # print("Accuracy :", a4)

  # ms_f4 = f1_score(callback.y_test, m4)
  # print("F1 :", ms_f4)


  # cm = metrics.confusion_matrix(callback.y_test, m4)
  # plot_confusion_matrix(cm, classes=['FAKE', 'REAL'],extention='on Training Set')

# gnb_machine(statement)

# gnb_machine(marged)

"""# K-fold"""

X=statement.x_train
y=statement.y_train

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

score = cross_val_score(tree.DecisionTreeClassifier(random_state= 42), X, y, cv= kf, scoring="accuracy")
print(f'Scores for each fold are: {score}')
print(f'Average score: {"{:.2f}".format(score.mean())}')

max_depth = [1,2,3,4,5,6,7,8,9,10]

for val in max_depth:
    score = cross_val_score(tree.DecisionTreeClassifier(max_depth= val, random_state= 42), X, y, cv= kf, scoring="accuracy")
    print(f'Average score({val}): {"{:.3f}".format(score.mean())}')

score = cross_val_score(ensemble.RandomForestClassifier(random_state= 42), X, y, cv= kf, scoring="accuracy")
print(f'Scores for each fold are: {score}')
print(f'Average score: {"{:.2f}".format(score.mean())}')

n_estimators = [50, 100, 150, 200, 250, 300, 350]

for val in n_estimators:
    score = cross_val_score(ensemble.RandomForestClassifier(n_estimators= val, random_state= 42), X, y, cv= kf, scoring="accuracy")
    print(f'Average score({val}): {"{:.8f}".format(score.mean())}')

score = cross_val_score(ensemble.RandomForestClassifier(n_estimators= 300,random_state= 42), X, y, cv= kf, scoring="accuracy")
print(f'Scores for each fold are: {score}')
print(f'Average score: {"{:.2f}".format(score.mean())}')

n_splits= [3,5,7,9,11,13,15]
for val in n_splits:
  kf = StratifiedKFold(n_splits=val, shuffle=True, random_state=42)
  score = cross_val_score(tree.DecisionTreeClassifier(max_depth= 5,random_state= 42), X, y, cv= kf, scoring="accuracy")
  print(f'Scores for each fold are: {score}')
  print(f'Average score for %s fold: {"{:.2f}".format(score.mean())}' %val)

n_splits= [3,5,7,9,11,13,15]
for val in n_splits:
  kf = StratifiedKFold(n_splits=val, shuffle=True, random_state=42)
  score = cross_val_score(ensemble.RandomForestClassifier(n_estimators= 300,random_state= 42), X, y, cv= kf, scoring="accuracy")
  print(f'Scores for each fold are: {score}')
  print(f'Average score for %s: {"{:.2f}".format(score.mean())}'%val)

n_splits= [3,5,7,9,11,13,15]
for val in n_splits:
  kf = StratifiedKFold(n_splits=val, shuffle=True, random_state=42)
  score = cross_val_score(SVC(C=10, gamma=1), X, y, cv= kf, scoring="accuracy")
  print(f'Scores for each fold are: {score}')
  print(f'Average score for %s: {"{:.2f}".format(score.mean())}'%val)

n_splits= [3,5,7,9,11,13,15]
for val in n_splits:
  kf = StratifiedKFold(n_splits=val, shuffle=True, random_state=42)
  score = cross_val_score(naive_bayes.MultinomialNB(), X, y, cv= kf, scoring="accuracy")
  print(f'Scores for each fold are: {score}')
  print(f'Average score for %s: {"{:.2f}".format(score.mean())}'%val)

"""# New Predict"""

def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Real News"

def avg_output_lable(n):
    if n <= 0.5:
        return "Fake News"
    elif n > 0.5:
        return "Real News"

# predict = pd.dataframe

predict_title="""প্রেমিকার স্বামীকে চাঁদে ৩ কাঠা জমি কিনে দিতে চান বাপ্পারাজ"""

predict_statement= """প্রেমিকার স্বামীর জন্য প্রেমিকাসহ নিজের সর্বস্ব বিলিয়ে দেওয়া বাপ্পারাজের মজ্জাগত। প্রতিটি সিনেমায় কোন না কোনভাবে প্রেমিকার স্বামীর পাশে ছিলেন বাপ্পারাজ৷ সম্প্রতি প্রেমিকার স্বামীকে চাঁদে ৩ কাঠা জমি কিনে দিতে চেয়ে মহানূভবতার আরো একটি উজ্জ্বল দৃষ্টান্ত স্থাপন করলেন তিনি৷ নিজের একটি ফেক আইডি থেকে এমনটাই জানান বাপ্পারাজ। ফেক আইডিটি থেকে ফেসবুক লাইভে এসে মুখে দুঃখের একটি হাসি ফুটিয়ে বাপ্পারাজ বলেন, 'আমি আর কার জন্য করবো! কে আছে আমার! প্রেমিকার স্বামীরাই তো আমার সব। চাঁদে জমি কেনার খবরে তাই সবার আগে তাদের কথাই মাথায় আসলো৷ প্রেমিকার স্বামীদের লিস্ট করা হচ্ছে, ক্যারিয়ারের সব সিনেমা মিলিয়ে ২০-২৫ জন তো হবেই৷ সবাইকেই ৩ কাঠা করে জমি কিনে দিতে চাই৷ এরপর আবার শান্তিতে আরো একটা প্রেমে নামার প্ল্যান।' লাইভের এই পর্যায়ে ক্যামেরা অফ না করেই পেছন ফিরে চোখের পানি মুছে আবারো হাসি হাসি মুখ করে ক্যামেরার দিকে তাকান বাপ্পারাজ৷ লাইভে এক ভক্তের টাকা পয়সা সংক্রান্ত প্রশ্নে বাপ্পারাজ বলেন, 'এত চিন্তার কিছু নেই৷ একটা কিডনি বেছে শবনমের স্বামীকে আইফোন তের কিনে দিছিলাম। চাঁদে জমি কেনার জন্য বাকি কিডনিটাও বিক্রি করেছি৷ একটা চোখ বিক্রির ব্যাপারে কথা চলছে৷ রক্তের বাজারও ভালো৷ আশা করি টাকা জোগাড় হয়ে যাবে৷ ওদের জন্য এইটুকু যে আমাকে পারতেই হবে।'"""

data = {'title':[predict_title],
        'statement':[predict_statement]}
  
# Create DataFrame
df = pd.DataFrame(data)

df

# dtc_pred_marged=dtc_marged.predict(marged_vector)

"""# Predict Manual input (Statement)"""

predict_test_cleaned = text_preprocess_statement.cleaned_texts(df['statement'])
predict_test_stemmed = text_preprocess_statement.stemming(predict_test_cleaned)
predict_test_vector = text_preprocess_statement.vectorizer_transform(__stopwords,predict_test_stemmed)

predict_test_vector.shape

dtc_pred=dtc.predict(predict_test_vector)
svm_pred=svm.predict(predict_test_vector)
rfm_pred=model_rfm.predict(predict_test_vector)
mnb_pred=MNB.predict(predict_test_vector)

print("\n\nDT Prediction: {} \nSVM Prediction: {} \nRF Prediction: {} \nMNB Prediction: {}".format(output_lable(dtc_pred[0]), output_lable(svm_pred[0]),  output_lable(rfm_pred[0]), output_lable(mnb_pred[0])))

all_pred = dtc_pred[0] + svm_pred[0] + rfm_pred[0] + mnb_pred[0]
print(avg_output_lable(all_pred/4))
print("\n\nAverage Prediction: {} ".format(avg_output_lable(all_pred/4)))

"""# Predict Manual input (marged)"""

title_predict_test_cleaned = text_preprocess_title.cleaned_texts(df['title'])
title_predict_test_stemmed = text_preprocess_title.stemming(title_predict_test_cleaned)
title_predict_test_vector = text_preprocess_title.vectorizer_transform(__stopwords,title_predict_test_stemmed)

title_predict_test_vector.shape

marged_vector =np.hstack((title_predict_test_vector.todense(),predict_test_vector.todense()))
# marged_vector= csr_matrix(np.hstack((title_vector.todense(),statement_vector.todense())))
marged_vector.shape

dtc_pred_marged=dtc_marged.predict(marged_vector)
svm_pred_marged=svm_marged.predict(marged_vector)
rfm_pred_marged=model_rfm_marged.predict(marged_vector)
mnb_pred_marged=MNB_marged.predict(marged_vector)

print("\n\nDT Prediction: {} \nSVM Prediction: {} \nRF Prediction: {} \nMNB Prediction: {}".format(output_lable(dtc_pred_marged[0]), output_lable(svm_pred_marged[0]),  output_lable(rfm_pred_marged[0]), output_lable(mnb_pred_marged[0])))

all_pred_marged = dtc_pred_marged[0] + svm_pred_marged[0] + rfm_pred_marged[0] + mnb_pred_marged[0]
print(avg_output_lable(all_pred_marged/4))
print("\n\nAverage Prediction: {} ".format(avg_output_lable(all_pred_marged/4)))

print("\n\nAverage Prediction of statement and marged: {} ".format(avg_output_lable((all_pred+all_pred_marged)/8)))