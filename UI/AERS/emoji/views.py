from django.shortcuts import render
from emoji.forms import FormName
# from emoji.test import main
from emoji.test import find_s
from emoji.test import recommend_em
import emoji.test
import emoji.cnnlstm
from emoji.cnnlstm import cnn_lstm
from emoji.cnnlstm import cnn_lstm_run
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle
import json
# Create your views here.

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

def index(request):
    form = FormName()
    context = {'form':form,'output': ' '}
    if request.method == 'POST':
        form = FormName(request.POST)
        if form.is_valid():
            print('sentence : ' + form.cleaned_data['sentence'])
            sentence = form.cleaned_data['sentence']
            output1,output2 = recommend_em(sentence,vect,new_model,mlb,new_emoji_matrix,emoji_set,v['top'], v['max_len'])
        context = {'form':form,'output1': output1, 'output2': output2}

    return render(request, 'emoji/index.html', context = context) 
