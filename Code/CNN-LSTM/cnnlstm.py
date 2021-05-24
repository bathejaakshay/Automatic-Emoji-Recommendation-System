"""
A Convolution Neural Network - Long Short term memory to recommend the emojis for a
sentence using the sentence's sentiment.

This module defines the following classes:

- `cnn_lstm` , a class that initializes the Convolution Neural Network
- `cnn_lstm_run` , a class that runs the initialized cnn_lstm over the Emoji Dataset

How To Use This Module
======================
(See the individual classes, methods and attributes for details.)

1. Import it: ``import cnn_lstm``

2. Create an object of ``cnn_lstm_run``::
    
        final_model = cnn_lstm.cnn_lstm_run()

3. Finally run the ``run()`` method of ``cnn_lstm_run`` class::

    	vect , model ,mlb, new_emoji_matrix,emoji_set,top, max_len = final_model.run()
    
4. After running the above lines, you should see the model metrics (``Precision``, ``Recall``, ``F1_score``, ``accuray``) for the two approaches followed i.e:
    - Without Emoji Matrix
    - With Emoji Matrix
    
5. If you want to test the system then call following method::
	
	final_model.recommend_em(vect,model,mlb,new_emoji_matrix,emoji_set,top, max_len)
	
6. This method will ask you to input a string and will recommend you the respective emojis for the same.


Working of this Module
======================

1. `cnn_lstm`, This class declares all the methods used for data preprocessing, generating cnn_lstm model , generating emoji_matrix that are used by `cnn_lstm_run` class.
2. `cnn_lstm_run` , This class does following functions:

    a) Generates a pandas DataFrame from a text file consisting of Tweets. These Tweets consists of text and multiple emojis. See the following functions for reference:
        - `extract_sent`, extracts the sentences from tweets.
        - `extract_emojis`, extracts the emojis from the tweets.
    
    b) It then cleans the DataFrame by removing stop_words, punctuations, performing lemmatization. ::

            def remove_stopwords(self,s):
                return ' '.join(self.lemmatizer.lemmatize(c.lower()) for c in list(s.split()) if c not in self.stop_words)
    
    c) It then keeps only sentences that has atleast one of the Top 100 most frequent used emojis. Look at the function `clean` and `label` for reference.

    d) After cleaning the whole dataset it the vectorize the text sentences and labels(emojis) using keras vectorizer and sklearn's Mulilabel Binarizer respectively::

        def vectorizer(self,data_df):
        '''
        vectorizing the whole dataset -  text and labels
        
        :type data_df: pandas.core.frame.DataFrame
        :paramter data_df: DataFrame consisting all the training and testing dataset with text and labels

        returns the vectorized sentences and labels respective to all the complete dataset
        '''

        #vectorizing
        vect = Tokenizer()
        vect.fit_on_texts(data_df['text'])
        train_sent = vect.texts_to_sequences(data_df['text'])
        vocab_size = len(vect.word_index) + 1
        
        #set max_len of sentence
        max_len = 50
        train_sent_X = pad_sequences(train_sent, padding='post', maxlen=max_len)
        mlb = MultiLabelBinarizer()
        train_label_y = mlb.fit_transform(data_df['label'].apply(lambda x: tuple(x)))
        
        return vect,train_sent,vocab_size,train_sent_X,train_label_y,mlb,max_len
    
    e) After vectorizing we split the whole dataset into 80%(train_data) and 20%(test_data)::
        
        #splitting dataset into train 80% and test 20%
        X_train,X_test,y_train,y_test = train_test_split(train_sent_X,train_label_y, test_size=0.2)

    f) After splitting the whole dataset we generate a tf.keras cnn_lstm model::
        
        def make_model(self,vocab_size,top,max_len):
        '''
        making cnn_lstm model using tf.keras

        :type vocab_size: int
        :parameter vocab_size: count of unique words in the corpus

        :type top: int
        :parameter top: count of most frequent emojis

        :type max_len: int
        :paramter max_len: fixed max length of the sentence

        returns the cnn_lstm model configured using tf.keras   
        '''

        model = keras.Sequential()
        model.add(tensorflow.keras.layers.Embedding(vocab_size,top, input_length=max_len))
        model.add(tensorflow.keras.layers.Conv1D(32, 5, padding='valid', activation='relu'))
        model.add(tensorflow.keras.layers.Conv1D(64, 3, padding='valid', activation='relu'))
        model.add(tensorflow.keras.layers.Dense(128, activation='relu'))
        model.add(tensorflow.keras.layers.Flatten())
        model.add(tensorflow.keras.layers.Dense(top, activation = 'sigmoid'))
        print(model.summary())
        return model

    
    g) Now we train the model::
        #compiling and training the model 
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
        model.fit(X_train, y_train, epochs = 1, validation_data=(X_test, y_test), batch_size = 64)
    
    h) Now we follow two approaches for evaluating our model:  
        - `Without Emoji Matrix`, directly find the model metrics from the model trained by predicting it over the test dataset.
        - `With Emoji Matrix`, generate an emoji_matrix from emoji_net. Look at ``generate_emoji_matrix`` for reference. Then transform the test labels and predicted labels into their respective keywords using emoji_matrix and the find the model metrics. Look at ``find_sem`` method for reference.
    
    i) After running all above steps you should be able to see the model's test metrics with and without emoji matrix.

"""


import tensorflow
import emoji
import string
import json
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import operator
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.stem.porter import PorterStemmer
import string
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score, f1_score,accuracy_score

class cnn_lstm:
    """
    A Convolution Neural Network.

    It declares all the methods for data preprocessing, cnn_lstm model generation, emoji_matrix generation etc.
    """
    def __init__(self,stop_words,lemmatizer):
        """
        Initializing the cnn_lstm class with stop_words and lemmatizer as its instance variables

        :type stop_words: list
        :parameter stop_words: list of stop_words

        :type lemmatizer: function declaration
        :parameter lemmatizer: function used to get the stems of the words
        """
        self.stop_words = stop_words
        self.lemmatizer = lemmatizer
        
    def extract_sent(self,s):
        '''
        Extracting text sentences out off the tweet

        :type s: string
        :parameter s: tweet containing text and emoji 

        returns the text and remove the emoji from the sentence
        '''
        
        return ''.join(c for c in s if c not in emoji.UNICODE_EMOJI and c not in list(string.punctuation))
    def extract_emojis(self,s):
        '''
        Extracting emojis out off the tweet
        
        :type s: string
        :parameter s: tweet containing text and emoji 
        
        returns a set of emojis in the respective sentence
        '''
        return set([c for c in s if c in emoji.UNICODE_EMOJI])

    def remove_stopwords(self,s):
        '''
        removing stop words and performing lemmatization
        
        :type s : string
        :parameter s : tweet containing text and emoji 
        
        returns a lemmatized sentence free from stop_words 
        '''
        return ' '.join(self.lemmatizer.lemmatize(c.lower()) for c in list(s.split()) if c not in self.stop_words)


    def data_preprocessing(self,data,all_emojis):
        '''
        Generating data dict from the emoji-twitter text file , performing lemmatization and removing stop_words, punctuations 
        and converting the sentence to lower caps

        :type data: dict
        :parameter data: dict consisting of the complete dataset with text and emojis
        
        :type all_emojis: dict
        :parameter all_emojis: dict consisting of all the emojis and their frequency

        returns the preprocessed dataset i.e removing stop_words, lemmatization, extracting emojis and text 
        '''

        #Generating data dict with first 2L tweets 
        with open('dataset/emojitweetsnew.txt') as file: 
            con = 0
            for i in file:
                #Taking first 2L tweets outoff the given file 
                if con <200000:
                    if con%100000 == 0:
                        print(con)

                    #extracting sentences and removing stopwords and performing lemmatization and lower caps  
                    sent = self.extract_sent(i)
                    sent = self.remove_stopwords(sent)
                    
                    #extracting emojis from the line
                    emo = self.extract_emojis(i)
                    if len(emo)<=0:
                        continue
                    data['text'].append(sent.strip())
                    data['label'].append(emo)
                    for t in emo:
                        if t in all_emojis:
                            all_emojis[t]+=1
                        else:
                            all_emojis[t] = 1
                    con+=1
                else:
                    break
        return data, all_emojis

    def set_emo(self, emo_list, emoji_set):
        """
        function to set emo_list and emoji_set as its instance members

        :type emo_list: list
        :parameter emo_list: list of emojis 

        :emoji_set: set
        :parameter emoji_set: set of emojis
        
        """
        self.emo_list = emo_list
        self.emoji_set = emoji_set

    def new_label(self,label):
        '''
        function to keep those labels with atleast one of the most frequent emojis

        :type label: set
        :parameter label: set of emojis

        returns the set if atleast one of the emojis in set is in emoji_set 
        '''
        if any(x in self.emoji_set for x in label):
            return label
        else:
            return None

    def clean(self,label):
        '''
        function to keep only 100 most frequent as target labels, removing empty labels
        
        :type label: set
        :parameter label: set of emojis

        returns set consisting of only the 100 most frequent emojis

        '''
        return {i for i in label if i in self.emo_list}

    def vectorizer(self,data_df):
        '''
        vectorizing the whole dataset -  text and labels
        
        :type data_df: pandas.core.frame.DataFrame
        :parameter data_df: DataFrame consisting all the training and testing dataset with text and labels

        returns the vectorized sentences and labels respective to all the complete dataset
        '''

        #vectorizing
        vect = Tokenizer()
        vect.fit_on_texts(data_df['text'])
        train_sent = vect.texts_to_sequences(data_df['text'])
        vocab_size = len(vect.word_index) + 1
        
        #set max_len of sentence
        max_len = 50
        train_sent_X = pad_sequences(train_sent, padding='post', maxlen=max_len)
        mlb = MultiLabelBinarizer()
        train_label_y = mlb.fit_transform(data_df['label'].apply(lambda x: tuple(x)))
        
        return vect,train_sent,vocab_size,train_sent_X,train_label_y,mlb,max_len
    
    def make_model(self,vocab_size,top,max_len):
        '''
        making cnn_lstm model using tf.keras

        :type vocab_size: int
        :parameter vocab_size: count of unique words in the corpus

        :type top: int
        :parameter top: count of most frequent emojis

        :type max_len: int
        :parameter max_len: fixed max length of the sentence

        returns the cnn_lstm model configured using tf.keras   
        '''

        model = keras.Sequential()
        model.add(tensorflow.keras.layers.Embedding(vocab_size,top, input_length=max_len))
        model.add(tensorflow.keras.layers.Conv1D(32, 5, padding='valid', activation='relu'))
        model.add(tensorflow.keras.layers.Conv1D(64, 3, padding='valid', activation='relu'))
        model.add(tensorflow.keras.layers.Dense(128, activation='relu'))
        model.add(tensorflow.keras.layers.LSTM(32,return_sequences=True))
        model.add(tensorflow.keras.layers.Flatten())
        model.add(tensorflow.keras.layers.Dense(top, activation = 'sigmoid'))
        print(model.summary())
        return model

    def generate_emoji_matrix(self):
        '''
        Generate Emoji_matrix using Emojinet , emoji_matrix consists of the emojis as its columns and keywords as its index
        
        '''
        d = open('dataset/emojis.json')
        data = json.load(d)
        keys = set()
        emojis = set()
        for dt in data:
            for kw in dt['keywords']:
                keys.add(kw)
            uni = dt['unicode'].split(' ')
            if len(uni[0]) == 7:
                emojis.add(uni[0])

        keys = np.transpose(np.array(list(keys)))
        emojis = np.array(list(emojis))
        emoji_mat = np.zeros((len(keys), len(emojis)))
        key = dict(zip(list(keys), range(0, len(keys))))
        emoji = dict(zip(list(emojis), range(0, len(emojis))))
        for dt in data:
            uni = dt['unicode'].split(' ')[0]
            if len(uni) == 7:
                for kw in dt['keywords']:
                    emoji_mat[key[kw]][emoji[uni]] = 1
        emoji_matrix = pd.DataFrame(data = emoji_mat, index = keys, columns = emojis)
        emojis_fig = []
        for i in range(len(emojis)):
            if len(emojis[i]) == 7:
                em = emojis[i].split('+')
                emojis_fig.append(str(em[0] + "000" + em[1]))
        emoji_fig = []
        for emoj in emojis_fig:
            emoji_fig.append(emoj.replace('U', r"\U").encode('ASCII').decode('unicode-escape'))

        em_mat = pd.DataFrame(data = emoji_mat, index = keys, columns = emoji_fig)
        # em_mat.to_csv('emoji_matrix_fig.csv')
        return em_mat


    def find_sem(self,y,y_hat,emoji_matrix,emoji_set,top):
        '''
        converting predicted multilabels into cluters/groups using emoji_matrix

        :type y: numpy.ndarray
        :parameter y: Actual vector of the test instance

        :type y_hat: numpy.ndarray
        :paramter y_hat: Predicted vector of the test instance

        :type emoji_matrix: pandas.core.frame.DataFrame
        :parameter emoji_matrix: emoji matrix constructed from the Emojinet

        returns the keywords corresponding to the y and y_hat

        '''
        e_y, e_y_hat = [],[]
        for i in range(top):
            if y[i] == 1:
                e_y.append(list(emoji_set.keys())[i])
            if y_hat[i] == 1:
                e_y_hat.append(list(emoji_set.keys())[i])
        emoji_mat_y = emoji_matrix[e_y]
        emoji_mat_y_hat = emoji_matrix[e_y_hat]
        y_sen = emoji_mat_y.sum(axis = 1).idxmax()
        y_hat_sen = emoji_mat_y_hat.sum(axis = 1).idxmax()
        return y_sen, y_hat_sen

class cnn_lstm_run:
    """
    It is a class that is responsible for the execution of all the methods
    declared in the cnn_lstm class and training and tesing the model
    and finally generating the model metrics with and without the emoji matrix.
    """
    
    def __init__(self):
        pass

    def run(self):
        """
        This method creates an object of cnn_lstm class and perform all the required operation for 
        generation of model metrics like data preprocessing, training , testing.
        """
        #initializing stop words and lemmatizer
        stop_words = set(stopwords.words('english')+list(string.punctuation))
        lemmatizer = WordNetLemmatizer()

        cnn_lstm_model = cnn_lstm(stop_words,lemmatizer)

        # all_emojis is a dict that will consist of all the emojis as its key and count of that respective emoji as its value
        all_emojis = {}

        #data dict is generated from the emojitweets-01-04-2018.txt text file
        data= {'text':[],'label':[]}

        data, all_emojis = cnn_lstm_model.data_preprocessing(data, all_emojis)

        #generating a dataframe outoff the data dict
        data_df = pd.DataFrame(data = data)
        all_emojis = dict(sorted(all_emojis.items(), key=operator.itemgetter(1), reverse=True))

        #generating emoji_matrix
        # emoji_matrix = pd.read_csv('/content/drive/MyDrive/Emoji Recommendation/emoji_matrix_fig.csv', index_col=0)
        
        emoji_matrix = cnn_lstm_model.generate_emoji_matrix()
        em_mat = list(emoji_matrix.columns)
        emoj_set = {}
        for i in range(len(em_mat)):
            if em_mat[i] in all_emojis.keys():
                emoj_set[em_mat[i]] = all_emojis[em_mat[i]]


        emoj_set = dict(sorted(emoj_set.items(), key=operator.itemgetter(1), reverse=True))
        top = 100

        #emoji_set consists of Top 100 most frequent used emojis
        emoji_set = dict(list(emoj_set.items())[0:top])
        emm = list(emoji_matrix.columns)

        #deleting all the emoji columns from the emoji matrix that are not most frequent
        del_em = []
        
        for i in range(len(emm)):
            if emm[i] not in emoji_set.keys():
                del_em.append(emm[i])
        emoji_matrix = emoji_matrix.drop(del_em, axis = 1)

        emoji_matrix = emoji_matrix[(emoji_matrix.T != 0).any()]
        emoji_set = dict(sorted(emoji_set.items(), key=operator.itemgetter(1), reverse=True))
        print(emoji_matrix)

        #contains all the emojis from emoji matrix
        emo_list = list(emoji_matrix.columns)

        cnn_lstm_model.set_emo(emo_list,emoji_set)

        data_df['label'] = data_df['label'].apply(cnn_lstm_model.new_label)
        data_df = data_df.dropna()

        
        data_df['label'] = data_df['label'].apply(cnn_lstm_model.clean)

        vect,train_sent,vocab_size,train_sent_X,train_label_y,mlb, max_len = cnn_lstm_model.vectorizer(data_df)

        new_emoji_matrix = emoji_matrix[mlb.classes_]

        #removing face as its sum count is 44 so removing it will give us more general idea
        new_emoji_matrix = new_emoji_matrix.drop(index = 'face',axis = 1)

        #splitting dataset into train 80% and test 20%
        X_train,X_test,y_train,y_test = train_test_split(train_sent_X,train_label_y, test_size=0.2)


        #creating a cnn_lstm model
        model = cnn_lstm_model.make_model(vocab_size,top,max_len)

        #compiling and training the model 
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
        model.fit(X_train, y_train, epochs = 2, validation_data=(X_test, y_test), batch_size = 64)

        #finding predictions with threshold 0.2
        y_hat_t = model.predict(X_test)
        y_hat = y_hat_t
        y_hat[y_hat >= 0.2] = 1
        y_hat[y_hat < 0.2] = 0

        #finding model metrics for different threshold values

        thresholds=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        predictions=y_hat_t

        print("Model metrics without grouping/clustering")
        for val in thresholds:
            print("For threshold: ", val)
            pred=predictions.copy()
        
            pred[pred>=val]=1
            pred[pred<val]=0
        
            precision = precision_score(y_test, pred, average='samples')
            accuracy = accuracy_score(y_test, pred)
            recall = recall_score(y_test, pred, average='samples')
            f1 = f1_score(y_test, pred, average='samples')
    
            print("samples-average quality numbers")
            print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f} , Accuracy : {:.4f}".format(precision, recall, f1,accuracy))




        #converting predicted and true test multilabels into clusters/groups using emoji matrix   
        al = len(y_test)
        count = 0
        true_group = []
        pred_group = []
        for i in range(al):
            #evaluate using emoji_matrix
            y,y_h = cnn_lstm_model.find_sem(y_test[i], y_hat[i], new_emoji_matrix, emoji_set, top)
            true_group.append(y)
            pred_group.append(y_h)
            if y == y_h:
                count = count + 1
        acc = count / al

        #finding all the model metrics for the case "with grouping using emoji matrix" 
        groups = new_emoji_matrix.index
        groups = groups.to_list()
        true_test = np.zeros((len(y_test),len(groups)))
        pred_test = np.zeros((len(y_test),len(groups)))
        for i in range(len(true_group)):
            true_test[i][groups.index(true_group[i])]=1
            pred_test[i][groups.index(pred_group[i])]=1

        precision = precision_score(true_test, pred_test, average='samples')
        accuracy = accuracy_score(true_test, pred_test)
        recall = recall_score(true_test, pred_test, average='samples')
        f1 = f1_score(true_test, pred_test, average='samples')

        print("\n\nModel metrics with grouping/clustering using emoji matrix")
        print("samples-average quality numbers")
        print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f} , Accuracy : {:.4f}".format(precision, recall, f1,accuracy))   
        
        return vect, model , mlb, new_emoji_matrix,list(mlb.classes_),top, max_len
    
    def find_s(self, y,emoji_matrix,emoji_set,top):
        '''
        It converts the predicted set of emojis into its keyword using emoji matrix

        :type y: numpy.ndarray
        :parameter y: numpy input vector 
        :type emoji_matrix: pandas.frame.DataFrame
        :parameter emoji_matrix: DataFrame consisting of keywords as its index and emojis as its columns
        :type emoji_set: set
        :parameter emoji_set: set of "top" most frequent emojis
        :type top: int 
        :parameter top: "top" most frequent emojis
        
        returns a keyword corresponding to the label vector
        '''
        e_y = []
        for i in range(top):
          if y[0][i] == 1:
            e_y.append(emoji_set[i])
        emoji_mat_y = emoji_matrix[e_y]
        y_sen = emoji_mat_y.sum(axis = 1).idxmax()
        return y_sen

    def recommend_em(self,vect,model,mlb,new_emoji_matrix,emoji_set,top,max_len):
        '''
        This function generates the emojis for the input sentence

        :type vect: keras_preprocessing.text.Tokenizer
        :parameter vect: tokenize the given input sentence
        :type model: tensorflow.python.keras.engine.sequential.Sequential
        :parameter model: trained CNN-LSTM model
        :type mlb: sklearn.preprocessing._label.MultiLabelBinarizer
        :parameter mlb: To vectorize the labels
        :type new_emoji_matrix: pandas.frame.DataFrame
        :parameter new_emoji_matrix: DataFrame consisting of keywords as its index and emojis as its columns
        :type emoji_set: set
        :parameter emoji_set: set of "top" most frequent emojis
        :type top: int
        :parameter top: "top" most frequent emojis
        :type max_len: int
        :parameter max_len: maximum length of sentence after padding

        '''
        while True:
            inp = input("\nEnter tweet text: ")
            inp_s = inp
            inp = vect.texts_to_sequences([inp])
            inp = pad_sequences(inp, padding='post', maxlen=max_len)
            out = model.predict([inp])
            out[out >= 0.1] = 1
            out[out < 0.1] = 0
            out_em = ""
            for i in range(100):
                if out[0][i] == 1:
                    out_em += emoji_set[i]
            print("\nOutput without emoji_matrix")
            print(inp_s + out_em)

            sentiment = self.find_s(out, new_emoji_matrix, emoji_set,top)
            emoo = new_emoji_matrix.columns[new_emoji_matrix.loc[sentiment] ==1]
            print("Output with emoji_matrix")
            print(inp_s + "".join(emoo))
            que = int(input('\nPress 1 to exit\n 2. To enter another sentence\n'))
            if que == 1:
                break
            else:
                continue

# if __name__ == '__main__':

#     final_model = cnn_lstm_run()
#     vect , model ,mlb, new_emoji_matrix,emoji_set,top, max_len = final_model.run()
#     final_model.recommend_em(vect,model,mlb,new_emoji_matrix,emoji_set,top, max_len)
