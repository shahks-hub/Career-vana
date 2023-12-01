import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

import re
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TfidfVectorizer


data_visualize_K = pd.read_csv('data/LocalPayNYC.csv')
data_visualize_B = pd.read_csv('data/emotionalEmployment.csv')

tabs = st.sidebar.radio("Select a tab", ('Geographic', 'Psychographic', 'Demographic', 'Find Your Perfect Career Sector'))

# Main content
st.title("Careers influenced by various factors over the years")

# Psychographic tab
if tabs == 'Psychographic':
    st.header("Psychographic Section")
    
    selected_factor = st.selectbox('Select a factor', ['_agreeableness', '_conscientiousness', '_emotional_stability', '_extroversion', '_openness'])

    # Display multiple box plots
    st.subheader("Box Plots")
    fig_box1 = px.box(data_visualize_B, x=selected_factor, y='50-100k', title=f"{selected_factor} $50-100k")
    st.plotly_chart(fig_box1)

    fig_box2 = px.box(data_visualize_B, x=selected_factor, y='100-150k', title=f"{selected_factor} $100-150k")
    st.plotly_chart(fig_box2)

    fig_box3 = px.box(data_visualize_B, x=selected_factor, y='150-200k', title=f"{selected_factor} $150-200k")
    st.plotly_chart(fig_box3)

    fig_box4 = px.box(data_visualize_B, x=selected_factor, y='200-500k', title=f"{selected_factor} $200-500k")
    st.plotly_chart(fig_box4)

    fig_box5 = px.box(data_visualize_B, x=selected_factor, y='+500k', title=f"{selected_factor} $500k+")
    st.plotly_chart(fig_box5)

    # Display histogram
    st.subheader("Histogram")
    fig_bar = px.histogram(data_visualize_B, x=selected_factor, color="_employment", title=f"Employment by {selected_factor}")
    st.plotly_chart(fig_bar)

elif tabs == 'Geographic':
    st.header("Geographic Section")

    # Load data
    df = pd.read_csv('data/mock_data.csv')

   
    sectors = st.multiselect('Select Employment Sectors', df['employment_sector'].unique())
    if not sectors:
        filtered_df = df
    else:
        filtered_df = df[df['employment_sector'].isin(sectors)]

    
    state_counts = filtered_df['state'].value_counts().reset_index()
    state_counts.columns = ['state', 'count']
    state_avg_salary = filtered_df.groupby('state')['salary'].mean().reset_index()


    map_color = st.selectbox('Select Map Color', ['Viridis', 'Cividis', 'Plasma', 'Inferno'])

    
    map_option = st.selectbox('Select what to display on the map', ['Number of Entries', 'Average Salary'])

    
if map_option == 'Number of Entries':
    fig = px.choropleth(state_counts, 
                        locations='state', 
                        color='count',  
                        color_continuous_scale=map_color,
                        scope="usa",
                        title='Number of Entries per State')
    st.plotly_chart(fig)
elif map_option == 'Average Salary':
    fig = px.choropleth(state_avg_salary, 
                        locations='state',  
                        locationmode="USA-states", 
                        color='salary',  
                        color_continuous_scale=map_color,
                        scope="usa",
                        title='Average Salary per State')
    st.plotly_chart(fig)



    

# Demographic tab
elif tabs == 'Demographic':
    st.header("Demographic Section")
    selected_factor = st.selectbox('Select a factor', ['gender', 'ethnicity', 'race'])

    # Display box plot and pie chart
    st.subheader("Box Plot")
    fig_box = px.box(data_visualize_K, x=selected_factor, y="upper_pay_band_bound", title=f"Pay Distribution by {selected_factor}")
    st.plotly_chart(fig_box)

    st.subheader("Pie Chart")
    fig_pie = px.histogram(data_visualize_K, x=selected_factor, color="job_category", title=f"Distribution of Job Categories by {selected_factor}")
    st.plotly_chart(fig_pie)

# Find Your Perfect Career Sector tab
elif tabs == 'Find Your Perfect Career Sector':

    # Load the model and initialize TfidfVectorizer
    filename = 'finalized_model.sav'
    filename2 = 'finalized_vector.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    loaded_vector = pickle.load(open(filename2,'rb'))
   

    # Define functions for text preprocessing
    def make_lower(a_string):
        return a_string.lower()

    def remove_punctuation(a_string):    
        a_string = re.sub(r'[^\w\s]', '', a_string)
        return a_string

  
    # Text preprocessing function using TfidfVectorizer
    def text_pipeline(input_string):
        input_string = make_lower(input_string)
        input_string = remove_punctuation(input_string) 
        return input_string

    # Streamlit app section
    st.title("Find Your Perfect Career Sector")

    # Text input area
    user_input = st.text_area("Enter your text here:")

    if st.button("Predict"):
        # Process the user input
        processed_input = text_pipeline(user_input)
        X = loaded_vector.transform([processed_input])
        predictions_proba = loaded_model.predict_proba(X)
        classes = loaded_model.classes_
        proba = predictions_proba[0]
        combined = list(zip(classes, proba))
        sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
        top_predictions = sorted_combined[:3]
        for i, (predicted_class, probability) in enumerate(top_predictions, start=1):
            st.write(f"Prediction {i}: Career path '{predicted_class}'")



   
        