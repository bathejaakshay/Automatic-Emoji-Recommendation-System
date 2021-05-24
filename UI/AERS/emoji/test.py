import emoji.cnnlstm
from emoji.cnnlstm import cnn_lstm
from emoji.cnnlstm import cnn_lstm_run
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle
import json



def find_s(y,emoji_matrix,emoji_set,top):
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

def recommend_em(string,vect,model,mlb,new_emoji_matrix,emoji_set,top,max_len):
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
    # while True:
    inp = string
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
    without_emo = inp_s + out_em

    sentiment = find_s(out, new_emoji_matrix, emoji_set,top)
    emoo = new_emoji_matrix.columns[new_emoji_matrix.loc[sentiment] ==1]
    print("Output with emoji_matrix")
    emo_sent_ret = inp_s + "".join(emoo)
    return without_emo,emo_sent_ret
    # que = int(input('\nPress 1 to exit\n 2. To enter another sentence\n'))
        # if que == 1:
            # break
        # else:
            # continue

if __name__ == '__main__':
    string = input('enter sentence')
    new_model = tf.keras.models.load_model('emoji/saved_model/my_model')
    print(new_model.summary())
    with open('emoji/saved_data/req_data.json','r') as f:
        v = json.load(f)
        v = json.loads(v)

    print(v.keys())
    with open('emoji/saved_data/vect.pkl', 'rb') as handle:
        vect = pickle.load(handle)
    with open('emoji/saved_data/mlb.pkl', 'rb') as handle, open('emoji/saved_data/new_emoji_matrix.pkl', 'rb') as handle2, open('emoji/saved_data/emoji_set.pkl', 'rb') as handle3:
        mlb = pickle.load(handle)
        new_emoji_matrix = pickle.load(handle2)
        emoji_set = pickle.load(handle3)
    
    print(recommend_em(string,vect,new_model,mlb,new_emoji_matrix,emoji_set,v['top'], v['max_len']))


