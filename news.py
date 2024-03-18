import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from flask import Flask,request,render_template
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
st = PorterStemmer()
sw = set(stopwords.words('english'))
nltk.download('stopwords')
voc_size = 6500
 
app = Flask(__name__)
model = keras.models.load_model('C:/Users/odhia/OneDrive/Desktop/streamlit tut/fakenews.keras')

@app.route('/')
def home():
    return render_template('news.html')

@app.route('/detect',methods=['POST'])
def predict_func():
    text = request.form['title']
    
    text = text.lower()
    text = re.sub('[^a-zA-Z]',' ',text)
    
    text = ' '.join(i for i in text.split() if i not in sw)
    text = ' '.join([st.stem(word) for word in text.split()])
    
    text = [tf.keras.preprocessing.text.one_hot(text,voc_size)]
    text = pad_sequences(text,padding = 'pre',maxlen = 30)
    
    pred = model.predict(text)
    
    if pred > 0.5:
        msg = 'This is a Real News'
    else:
        msg = 'This is a Fake News!!'
        
    return render_template('news.html',predict_message = msg)


if __name__ == '__main__':
    app.run(debug=True)