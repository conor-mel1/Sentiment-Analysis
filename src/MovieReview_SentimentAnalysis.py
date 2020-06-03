

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem.porter import PorterStemmer
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import optimizers
import tensorflow as tf
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('punkt')

#Method used to preprocess the data set
def preprocess(dataset): 
    #convert the sentences to word tokens
    tokens = word_tokenize(dataset)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    #stem each word
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]
    #join all the word tokens back into single sentence
    stemmed = " ".join(stemmed)

    return stemmed


#Create the Y labels based on the 12,500 positive examples first followed by 12,500 negative examples
labels =[]
i = 0
for i in range (25000):
    if i<=12499:
        labels.append(1)
    else:
        labels.append(0)

labels =np.array(labels)
print(labels)

#Read the txt file using pandas splitting rows by new line 
train = pd.read_csv('full_train.txt', sep = "\n", header=None)

#Rename dataframe columns
train['labels'] = labels
train =train.rename(columns={0:"Review", "labels" :"Labels"})

#Splitting the reviews and labels, converting to numpy arrays
X= train.iloc[:,0].values
Y = train.iloc[:,1].values

preprocessed_sentences = []
for sentence in X:
    new_sentence = preprocess(sentence)
    preprocessed_sentences.append(new_sentence)
    
text_tokenizer = Tokenizer()
text_tokenizer.fit_on_texts(preprocessed_sentences)
word_dict = text_tokenizer.word_index # Assigning unique integer to each word in input text
X_RNN = text_tokenizer.texts_to_sequences(preprocessed_sentences)

num_source_tokens = len(text_tokenizer.word_index)
max_source_seq_length = max(len(text_to_word_sequence(x)) for x in preprocessed_sentences)
print(num_source_tokens)
print(max_source_seq_length)

#pad each sequence with 0's until max length
X_RNN = pad_sequences(X_RNN)

# Splitting data into training data and test data
# 90/10 split chosen for this
y_RNN = Y
X_RNN_train, X_RNN_test, y_RNN_train, y_RNN_test = train_test_split(X_RNN,y_RNN, train_size =0.9) 
print(len(X_RNN_train))

#Create acceptor RNN model to classify reviews using tensorflow
#3 layers, the original embedding layer, bidirectional LSTM layer and fully connected output layer
model = tf.keras.Sequential([
    #size of vocab and output dimensions
    tf.keras.layers.Embedding(70000, 256,input_length=1440),
    #LSTM with 128 nodes
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
    #dense output layer binary therefore only 1
    tf.keras.layers.Dense(1)
])
#compile as binary output with Stochastic gradient descent optimiser, comparing accuracy for each epoch
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer='sgd',
              metrics=['accuracy'])
#give summary of model
model.summary()

#Train the model over 100 epochs in batchs of 64 
epochs = 100
Batch_size = 64
history = model.fit(x=X_RNN_train,y=y_RNN_train, batch_size=Batch_size, epochs=epochs,validation_split=0.2)

#Evaluate model with test data
results = model.evaluate(x= X_RNN_test, y=y_RNN_test)
print('test loss, test acc:', results)

#Save model as h5 file
model.save("movie_reviewer.h5")

#Plotting loss of validation and training data after training the model 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc = 'upper left')
plt.show()

#Ploting accuracy of model for train and validation data
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc = 'upper left')
plt.show()
