import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
import streamlit as st
from streamlit_folium import folium_static
# import pickle
from PIL import Image
import folium
from wordcloud import WordCloud
import itertools
import snscrape.modules.twitter as sntwitter
# import tensorflow as tf

# model = tf.lite.Interpreter(model_path="res/converted_quant_model.tflite")
# model.allocate_tensors()

# input_details = model.get_input_details()
# output_details = model.get_output_details()
# print(input_details[0]['shape'])
# print(output_details[0]['shape'])
# print(input_details[0]['dtype'])
# print(output_details[0]['dtype'])

# infile = open('res/tokenizer.pkl','rb')
# tokenizer = pickle.load(infile)
# infile.close()

TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

SEQUENCE_LENGTH = 300

POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

# loc = '30.88859095134539, 76.68799584206599, 300km'
inputT=pd.DataFrame(itertools.islice(sntwitter.TwitterSearchScraper(f'near:"Delhi" within:100km ').get_items(),600))     # since:2021-10-25 until:2021-10-26
print(inputT.content.isna().sum())
inputT=inputT[inputT.lang=='en']

stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")
def preprocess(text, stem=False):
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)
# out=inputT.text.copy()

out=inputT.content.apply(lambda x: preprocess(x))


# def decode_sentiment(score, include_neutral=True):
#     if include_neutral:        
#         label = NEUTRAL
#         if score <= SENTIMENT_THRESHOLDS[0]:
#             label = NEGATIVE
#         elif score >= SENTIMENT_THRESHOLDS[1]:
#             label = POSITIVE

#         return label
#     else:
#         return NEGATIVE if score < 0.5 else POSITIVE

# usr_neg=[]
# def predict(text_list, include_neutral=True):
    
#     # a=time.time()
#     for j,i in enumerate(text_list):
        
#         x_test = pad_sequences(tokenizer.texts_to_sequences([i]), maxlen=SEQUENCE_LENGTH)
#         # score = model.predict([x_test])[0]
#         x_test=np.array(x_test,dtype=np.float32)
#         model.set_tensor(input_details[0]['index'],x_test)
#         # print(x_test.shape)
#         model.invoke()
#         score=model.get_tensor(output_details[0]['index'])[0][0]
#         label = decode_sentiment(score, include_neutral=include_neutral)
        
        
#         if label==NEGATIVE:
#             usr_neg.append((j,score))
    # print(time.time()-a)
    

# no=np.random.randint(0,190,189)

# @st.cache
def folium_map():
    df=pd.read_csv('res/part1.csv')[['Police Station','Latitude','Longitude','Event Type']]
    dfP1=df[df['Police Station']=='PS1'].reset_index()
    dfP2=df[df['Police Station']=='PS2'].reset_index()
    dfP3=df[df['Police Station']=='PS3'].reset_index()
    dfP4=df[df['Police Station']=='PS4'].reset_index()
    m= folium.Map(location=[df.Latitude.mean(),
                            df.Longitude.mean()], zoom_start=11, control_scale=True)
    group1 = folium.FeatureGroup(name='<span style=\\"color: red;\\">PS1 C1 circle (Blue)</span>')
    group2 = folium.FeatureGroup(name='<span style=\\"color: blue;\\">PS2 C1 circle (Green)</span>')
    group3 = folium.FeatureGroup(name='<span style=\\"color: red;\\">PS3 C2 circle (Red)</span>')
    group4 = folium.FeatureGroup(name='<span style=\\"color: blue;\\">PS4 C2 circle (Black)</span>')

    for i,j in enumerate(zip(dfP1.Latitude,dfP1.Longitude)):
        location = [j[0],j[1]]
        folium.CircleMarker(location,radius=1,popup=dfP1['Event Type'][i]).add_to(group1)
    group1.add_to(m)

    for i,j in enumerate(zip(dfP2.Latitude,dfP2.Longitude)):
        location = [j[0],j[1]]
        folium.CircleMarker(location,radius=1,popup=dfP2['Event Type'][i],color='green').add_to(group2)
    group2.add_to(m)

    for i,j in enumerate(zip(dfP3.Latitude,dfP3.Longitude)):
        location = [j[0],j[1]]
        folium.CircleMarker(location,radius=1,popup=dfP3['Event Type'][i],color='red').add_to(group3)
    group3.add_to(m)

    for i,j in enumerate(zip(dfP4.Latitude,dfP4.Longitude)):
        location = [j[0],j[1]]
        folium.CircleMarker(location,radius=1,popup=dfP4['Event Type'][i],color='black').add_to(group4)
    group4.add_to(m)

    folium.map.LayerControl('topright', collapsed=False).add_to(m)
    # return folium_static(m)
    folium_static(m)
folium_map()

with st.expander("Word Cloud"):
        # st.header('WordCloud')
        tw_mask = np.array(Image.open("res/images.jpg").convert('1'))

        def transform_format(val):
            if val == 0:
                return val
            else:
                return 255
        transformed_wine_mask = np.ndarray((tw_mask.shape[0],tw_mask.shape[1]), np.int32)

        for i in range(len(tw_mask)):
            transformed_wine_mask[i] = list(map(transform_format, tw_mask[i]))
        wc = WordCloud(background_color="white", max_words=1000, mask=transformed_wine_mask,width=300,contour_color='firebrick',contour_width=2)
                    
        wc.generate(' '.join(out.values))
        st.image(wc.to_array(),width=370)




# predict(out.values)

# neg_df=pd.DataFrame(usr_neg, columns =['ind','score'])
# neg_df.set_index('ind',inplace=True)
# 
# final_df=inputT.join(neg_df,how='left').sort_values('score',ascending=True).dropna(subset=['score'])


# print(final_df.content.isna().sum())
# st.write('Date','Tweet ID','Score','Username','Name','Location')
# def fun(x):
#     print(x['date'],x['id'])
#     st.write(x['date'],round(x['score'],3),x['user']['username'],x['user']['displayname'],x['user']['location'])
# final_df.apply(fun,axis=1)
# for i in inputT.iloc[usr_neg]:
    # 
# tweet = st.text_input('Movie title', 'hey i am eating pizza.Come and join me.')
# predict(tweet)
# res=predict(tweet)




