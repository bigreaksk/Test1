import streamlit as st
import seaborn as sns
df = sns.load_dataset('iris')
df

# create sidebar menu for user can select classifiers
classifier = st.sidebar.selectbox('Classifier', ('KNN', 'SVM', 'Decision Tree', 'Random Forest', 'Neural Network'))
