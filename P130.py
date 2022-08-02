import pandas as pd
import streamlit as st
import docx2txt
import pdfplumber
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot  as plt
import plotly.express as px
stop=set(stopwords.words('english'))
import pickle
vectors = pickle.load(open('vect.pkl','rb'))
model = pickle.load(open('xgb.pkl','rb'))


nltk.download('wordnet')
nltk.download('stopwords')


resume = []

def display(doc_file):
    if doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume.append(docx2txt.process(doc_file))
    else :
        with pdfplumber.open(doc_file) as pdf:
            pages=pdf.pages[0]
            resume.append(pages.extract_text())
    return resume

def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"")
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words)





def mostcommon_words(cleaned,i):
    tokenizer = RegexpTokenizer(r'\w+')
    words=tokenizer.tokenize(cleaned)
    mostcommon=FreqDist(cleaned.split()).most_common(i)
    return mostcommon

def display_wordcloud(mostcommon):
    wordcloud=WordCloud(width=1000, height=600, background_color='black').generate(str(mostcommon))
    a=px.imshow(wordcloud)
    st.plotly_chart(a)

def display_words(mostcommon_small):
    x,y=zip(*mostcommon_small)
    chart=pd.DataFrame({'keys': x,'values': y})
    fig=px.bar(chart,x=chart['keys'],y=chart['values'],height=700,width=700)
    st.plotly_chart(fig)





def main():
    st.title('DOCUMENT CLASSIFICATION')
    upload_file = st.file_uploader('Hey,Upload Your Resume ',
                                type= ['docx','pdf'],accept_multiple_files=True)
    if st.button("Process"):
        for doc_file in upload_file:
            if doc_file is not None:
                file_details = {'filename':[doc_file.name],
                               'filetype':doc_file.type.split('.')[-1].upper(),
                               'filesize':str(doc_file.size)+' KB'}
                file_type=pd.DataFrame(file_details)
                st.write(file_type.set_index('filename'))
                displayed=display(doc_file)

                cleaned=preprocess(display(doc_file))
                predicted= model.predict(vectors.transform([cleaned]))

                string='The Uploaded Resume is belongs to '+predicted[0]
                st.header(string)

                st.subheader('WORDCLOUD')
                display_wordcloud(mostcommon_words(cleaned,100))

                st.header('Frequency of 20 Most Common Words')
                display_words(mostcommon_words(cleaned,20))

if __name__ == '__main__':
    main()