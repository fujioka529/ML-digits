#S_ML_lightning_classifer_sep18_2023.py.py
#simple model with hyperparamters search
# one time save to VM: vectorizer, clf_lightning
'''
Python 3.10.11
import  sklearn
sklearn.__version__
'1.3.0'

'''
import sys
print('python version is ', sys.version)
print('path to python exe ' ,  sys.executable)
import  sklearn
print('sklearn  version' , sklearn.__version__)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix

import  numpy as  np
from sklearn.metrics import f1_score




if 0: #  use only 0
    q = 0
else: #read data from file 
    q = 0
    text_file = open("y_test.txt", "r",encoding='utf-8', errors="ignore")
    y_test = text_file.readlines()
    y_test =  [int(x) for x in y_test]
    text_file = open("y_train.txt", "r",encoding='utf-8', errors="ignore")
    y_train = text_file.readlines()
    y_train =  [int(x) for x in y_train]
    text_file = open("X_test.txt", "r",encoding='utf-8', errors="ignore")
    X_test = text_file.readlines()    
    text_file = open("X_train.txt", "r",encoding='utf-8', errors="ignore")
    X_train = text_file.readlines()
    q = 0


#
y_train =  np.array( y_train )
#y_train[0:934] =  0
#y_train[ [7,  8 ,  29,  68 ,  909] ] =  0

y_test =  np.array( y_test )
q = 0


def preprocessing_objects(X_train, data_to_vectorize ):
    vectorizer = TfidfVectorizer( sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english")    
    #X_train_original =  copy.deepcopy(X_train)
    from gensim.parsing.preprocessing import preprocess_documents
    from gensim.parsing.porter import PorterStemmer
    use_PorterStemmer = PorterStemmer()
    X_train =  use_PorterStemmer.stem_documents( X_train )
    X_train =  preprocess_documents(X_train)
    X_train = [" ".join(x) for x in X_train]
    vectorizer.fit(X_train)
    data_to_vectorize =  use_PorterStemmer.stem_documents( data_to_vectorize )
    data_to_vectorize =  preprocess_documents(data_to_vectorize)
    data_to_vectorize = [" ".join(x) for x in data_to_vectorize]  
    X_train_vectorised = vectorizer.transform(data_to_vectorize)
    return X_train_vectorised, vectorizer ,  preprocess_documents ,  use_PorterStemmer
X_train_vectorised, TFIDF_vectorizer,   my_preprocess_documents ,  my_PorterStemmer =  preprocessing_objects(X_train, X_train )


def do_preprocessing(use_PorterStemmer, preprocess_documents, vectorizer, data_to_vectorize ):
    data_to_vectorize =  use_PorterStemmer.stem_documents( data_to_vectorize )
    data_to_vectorize =  preprocess_documents(data_to_vectorize)
    data_to_vectorize = [" ".join(x) for x in data_to_vectorize]  
    X_vectorised = vectorizer.transform(data_to_vectorize)
    return X_vectorised

test_data_vectorised =  do_preprocessing(my_PorterStemmer, my_preprocess_documents, TFIDF_vectorizer,  X_test )
q = 0

if 1:
    #https://github.com/scikit-learn-contrib/lightning/blob/master/examples/document_classification_news20.py
    #https://contrib.scikit-learn.org/lightning/generated/lightning.regression.FistaRegressor.html
    #clf =  lightning.regression.FistaRegressor(C=1.0, alpha=1.0, penalty='l1', max_iter=100, max_steps=30, eta=2.0, sigma=1e-05, callback=None, verbose=0)        
    #https://contrib.scikit-learn.org/lightning/index.html
    #pip install sklearn-contrib-lightning
    from lightning.classification import CDClassifier #Estimator for learning linear classifiers by (block) coordinate descent
    #import  lightning
    #do in parallel on many VMs/instances
    q = 0
    clf_lightning = CDClassifier(loss="squared_hinge",
                       #penalty="l2",
                       penalty="l1",
        #penalty="l1/l2", #                     
                       multiclass=False,
                       max_iter=20,
                       alpha=1e-4,
                       C=1.0 / X_train_vectorised.shape[0],
                       tol=1e-3,
                       n_jobs =5)

    clf_lightning.fit(X_train_vectorised, y_train )
    pred_train_lightning = [int(x) for x in clf_lightning.predict(X_train_vectorised)]   
    original_targt_lightning = [int(x) for x in y_train]  
    print('\n****                 train data lightning confusion_matrix     **************')
    print(confusion_matrix( original_targt_lightning , pred_train_lightning)  )
    q = 0
    pred_test_data_lightning      = [int(x) for x in clf_lightning.predict(test_data_vectorised)]   
    original_test_targt_lightning = [int(x) for x in y_test]  
    print('\n****                 test data lightning confusion_matrix     **************')
    print(confusion_matrix( original_test_targt_lightning , pred_test_data_lightning)  )
    print('F1 performance  = ', f1_score(original_test_targt_lightning , pred_test_data_lightning))
    q = 0

                        
    #





#  input your text -> imitate request from any computer or web site 
my_text =  X_test[0]

#my_text =  'From: teezee@netcom.com (TAMOOR A. ZAIDI)\nSubject: Hall Generators from USSR\nKeywords: hall generators,thrusters,USSR,JPL\nOrganization: NETCOM On-line Communication Services (408 241-9760 guest)\nLines: 21\n\nHi Folks,\n\n              Last year America bought two  "Hall Generators" which are\nused as thrusters for space vehicles from former USSR,if I could recall\ncorrectly these devices were sent to JPL,Pasadena labs for testing and\nevaluation.\n     \n              I am just curious to know  how these devices work and what\nwhat principle is involved .what became of them.There was also some\ncontroversy that the Russian actually cheated,sold inferior devices and\nnot the one they use in there space vehicles.\n\nAny info will be appreciated...\n  ok   {                         Thank{ in advance...\nTamoor A Zaidi\nLockheed Commercial Aircraft Center\nNorton AFB,San Bernardino\n\nteezee@netcom.com\nde244@cleveland.freenet.edu\n\n'
#my_text =  'Mac and was wondering if anyone in netland knows of public domain anti-aliasing utilities so that I can skip this step '



q = 0
def  inference_for_one_text_message(my_text ):
    #test_vectorised =  preprocessing_objects(X_train, [my_text])
    test_vectorised =  do_preprocessing(my_PorterStemmer, my_preprocess_documents, TFIDF_vectorizer,  [my_text] )
    
    #predict use existing model 
    pred_test_lightning = clf_lightning.predict(  test_vectorised    )
    print('for text => ', my_text )
    print('pred_test_lightning', pred_test_lightning[0])
    return pred_test_lightning[0]
from datetime import datetime
start_time =  datetime.now()
N =  500
for  i in  range(N):
    my_prediction  = inference_for_one_text_message(my_text)
end_time  =  datetime.now()
time_diff =  end_time -  start_time
print('time spent for ',  N ,  ' requests    ' , time_diff)
q = 0