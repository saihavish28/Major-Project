from flask import Flask, render_template, Response, request
import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

database = "test3.db"

def create_table():
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS register(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, mail TEXT, phoneno TEXT, address TEXT, password TEXT)")
    conn.commit()
    conn.close()

create_table()

dataset = pd.read_csv('dataset/tweet_sentiment.csv')
dataset['cleaned_tweets'] = dataset['cleaned_tweets'].fillna('').astype(str)


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/register', methods=["GET","POST"])
def register():
    if request.method == "POST":
        name = request.form['name']
        mail = request.form['mail']
        phoneno = request.form['phoneno']
        password = request.form['password']
        address = request.form['address']

        conn = sqlite3.connect(database)
        cursor = conn.cursor()
        cursor.execute("SELECT mail, phoneno FROM register WHERE mail=? OR phoneno=?", (mail, phoneno,))
        registered = cursor.fetchall()
        if registered:
            return render_template('index.html', show_alert1=True)
        else:
            cursor.execute("INSERT INTO register(name, mail, phoneno, address, password) VALUES(?,?,?,?,?)", (name, mail, phoneno, address, password))
            conn.commit()
            return render_template('index.html', show_alert2=True)
    return render_template('index.html')

@app.route('/login', methods=["GET","POST"])
def login():
    global mail
    if request.method == "POST":
        mail = request.form['mail']
        password = request.form['password']
        conn = sqlite3.connect(database)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM register WHERE mail=? AND password=?", (mail, password))
        data = cursor.fetchone()
        if data is None:
            return render_template('index.html', show_alert5=True)
        else:
            conn = sqlite3.connect(database)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM register WHERE mail=?", (mail,))
            results = cursor.fetchone()
            conn.commit()
            return render_template('comment.html')
    return render_template('index.html', show_alert5=True)



tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(dataset['cleaned_tweets'])
maxlen = 100  # define your maximum sequence length

model1 = load_model("model.h5")

@app.route("/comment", methods=["GET", "POST"])
def comment():
    if request.method == "POST":
        user_comment = request.form["comment"]  # Rename variable to avoid confusion with function name
        sequence = tokenizer.texts_to_sequences([user_comment])  # Use user's comment
        padded_sequence = pad_sequences(sequence, maxlen=maxlen)
        prediction = model1.predict(padded_sequence)
        sentiment_label = np.argmax(prediction)
        
        if sentiment_label == 0:
            result = "Happy"
        elif sentiment_label == 1:
            result = "Sad"
        else:
            result = "Neutral"
            
        return render_template('final.html', result=result)
        
    return render_template('comment.html')



if __name__ == "__main__":
    app.run(port=300)
