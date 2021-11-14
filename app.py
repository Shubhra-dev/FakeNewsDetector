from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def first():
    return render_template('index.html')

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
import numpy as np
import itertools
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC
from sklearn import naive_bayes
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

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
  vector = pickle.load(open('vectorfit.pkl','rb'))

  def __init__(self):
    pass
  
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


  def cleaned_texts(self, data):
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
  # --------------------------------------------------------------
  def vectorizer_transform(self,stemmed):
    
    tf_idf_matrix = self.vector.transform(stemmed)
    return tf_idf_matrix
  # -------------------------------------------------------------

"""# Predict Manual input (Statement)"""
@app.route("/prediction", methods=["POST"])
def home():
  predict_title = request.form['heading']
  predict_statement = request.form['statement']
  
  data = {'title':[predict_title],
        'statement':[predict_statement]}
  # Create DataFrame
  df = pd.DataFrame(data)
    
  textPreProcess = TextPreProcess()
  predict_test_cleaned = textPreProcess.cleaned_texts(df['statement'])
  predict_test_stemmed = textPreProcess.stemming(predict_test_cleaned)
  predict_test_vector = textPreProcess.vectorizer_transform(predict_test_stemmed)
    
  result = 0
        
  svm = pickle.load(open('svmfit.pkl','rb'))
  svm_pred=svm.predict(predict_test_vector)
  result+=svm_pred
  if svm_pred == 1 :
        svm_pred_value = "True"
  else:
    svm_pred_value = "Fake"
    

  svmlin = pickle.load(open('svmLinearfit','rb'))
  svmlin_pred=svmlin.predict(predict_test_vector)
  result+=svmlin_pred
  if svmlin_pred == 1 :
        svmlin_pred_value = "True"
  else:
    svm_pred_value = "Fake"

  lr = pickle.load(open('lrfit.pkl','rb'))
  lr_pred=svm.predict(predict_test_vector)
  result+=lr_pred
  if lr_pred == 1 :
        lr_pred_value = "True"
  else:
    lr_pred_value = "Fake"
  
  rfm = pickle.load(open('rfmfit','rb'))
  rfm_pred=rfm.predict(predict_test_vector)
  result+=rfm_pred
  if rfm_pred == 1 :
        rfm_pred_value = "True"
  else:
    rfm_pred_value = "Fake"
    
  
  mnb = pickle.load(open('mnbfit.pkl','rb'))
  mnb_pred=svm.predict(predict_test_vector)
  result+=mnb_pred
  if mnb_pred == 1 :
        mnb_pred_value = "True"
  else:
    mnb_pred_value = "Fake"
  
  mlp = pickle.load(open('mlpfit','rb'))
  mlp_pred=svmlin.predict(predict_test_vector)
  result+=mlp_pred
  if mlp_pred == 1 :
        mlp_pred_value = "True"
  else:
    mlp_pred_value = "Fake"


  if (result/6) >= 0.5:
    return render_template('true.html',acc=((result/6)*100),svm=svm_pred_value,svmlin=svmlin_pred_value,rfm=rfm_pred_value,lr=lr_pred_value,mnb=mnb_pred_value,mlp=mlp_pred_value)
  else:
    return render_template('fake.html',acc=((1-(result/6))*100),svm=svm_pred_value,svmlin=svmlin_pred_value,rfm=rfm_pred_value,lr=lr_pred_value,mnb=mnb_pred_value,mlp=mlp_pred_value)  

if __name__ == '__main__':
    app.run(debug=True)
