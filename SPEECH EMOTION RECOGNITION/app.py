import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
import warnings
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import OneHotEncoder
import speech_recognition as sr

warnings.filterwarnings('ignore')

# Load Dataset
@st.cache_data
def load_data():
    paths = []
    labels = []
    for dirname, _, filenames in os.walk("C:\\Users\\sanka\\Downloads\\TESS Toronto emotional speech set data"):
        for filename in filenames:
            paths.append(os.path.join(dirname, filename))
            label = filename.split('_')[-1]
            label = label.split('.')[0]
            labels.append(label.lower())
    df = pd.DataFrame()
    df['speech'] = paths
    df['label'] = labels
    return df

df = load_data()

# Feature Extraction
@st.cache_data
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))
X = np.array([x for x in X_mfcc])
X = np.expand_dims(X, -1)

enc = OneHotEncoder()
y = enc.fit_transform(df[['label']]).toarray()

# LSTM Model
@st.cache_resource
def create_model():
    model = Sequential([
        LSTM(256, return_sequences=False, input_shape=(40,1)),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(7, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = create_model()

# Define pages
def data_processing_page():
    st.title('Data Processing')
    st.write('### Dataset Preview')
    st.dataframe(df.head())

def eda_page():
    st.title('Exploratory Data Analysis')
    st.write('### Label Distribution')
    fig, ax = plt.subplots()
    sns.countplot(df['label'], ax=ax)
    st.pyplot(fig)

    st.write('### Waveform and Spectrogram')
    emotion = st.selectbox('Choose an emotion', df['label'].unique())
    path = np.array(df['speech'][df['label']==emotion])[0]
    
    data, sampling_rate = librosa.load(path)
    wave_fig, wave_ax = plt.subplots()
    wave_ax.set(title=f'{emotion} Waveplot', xlabel='Time (s)', ylabel='Amplitude')
    wave_ax.plot(np.linspace(0, len(data)/sampling_rate, num=len(data)), data)
    
    spec_fig, spec_ax = plt.subplots()
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    spec_ax.set(title=f'{emotion} Spectrogram')
    img = librosa.display.specshow(xdb, sr=sampling_rate, x_axis='time', y_axis='hz', ax=spec_ax)
    spec_fig.colorbar(img, ax=spec_ax)
    
    st.pyplot(wave_fig)
    st.pyplot(spec_fig)

def classification_page():
    st.title('Speech Emotion Classification')

    # Train Model
    if st.button('Train Model'):
        history = model.fit(X, y, validation_split=0.2, epochs=50, batch_size=64)
        st.write('### Training Accuracy and Loss')

        fig, ax = plt.subplots()
        ax.plot(history.history['accuracy'], label='Train Accuracy')
        ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax.set(title='Accuracy', xlabel='Epochs', ylabel='Accuracy')
        ax.legend()
        st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='Train Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set(title='Loss', xlabel='Epochs', ylabel='Loss')
        ax.legend()
        st.pyplot(fig)

    # Recognize and Classify Emotion
    def classify_emotion(model, audio_path):
        mfcc = extract_mfcc(audio_path)
        mfcc = np.expand_dims(mfcc, axis=0)
        mfcc = np.expand_dims(mfcc, axis=-1)
        pred = model.predict(mfcc)
        emotion = enc.inverse_transform(pred)
        return emotion[0][0]

    st.write('### Speech Recognition and Emotion Classification')
    if st.button('Record and Classify Emotion'):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write('Speak Anything...')
            audio = recognizer.listen(source)
            try:
                audio_path = 'temp.wav'
                with open(audio_path, 'wb') as f:
                    f.write(audio.get_wav_data())
                text = recognizer.recognize_google(audio)
                st.write(f'You said: {text}')
                emotion = classify_emotion(model, audio_path)
                st.write(f'Predicted Emotion: {emotion}')
            except Exception as e:
                st.write(f'Sorry, could not recognize your speech. Error: {e}')

# Sidebar for navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Data Processing', 'EDA', 'Speech Emotion Classification'])

if page == 'Data Processing':
    data_processing_page()
elif page == 'EDA':
    eda_page()
elif page == 'Speech Emotion Classification':
    classification_page()
