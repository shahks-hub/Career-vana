import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle 
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from sklearn.feature_extraction.text import TfidfVectorizer  



data_visualize_K = pd.read_csv('data/LocalPayNYC.csv')
data_visualize_B = pd.read_csv('data/b_data.csv')

tabs = st.sidebar.radio("Select a tab", ('Geographic', 'Psychographic', 'Demographic', 'Find Your Perfect Career Sector', 'Generate Cover Letter'))

# Main content
st.title("Careers influenced by various factors over the years")






# Psychographic tab
if tabs == 'Psychographic':
    st.header("Psychographic Section")
    
    # Create a list of available factors (columns) in the CSV file
    available_factors = data_visualize_B.columns.tolist()

    # Let the user choose factors (columns) from the CSV file
    col1, col2, col3 = st.columns(3)  # Create two columns for side-by-side display

    with col1:
        selected_factor1 = st.selectbox('Select a factor (1)', available_factors, key='factor1')
        unique_values_factor1 = data_visualize_B[selected_factor1].unique()
        st.write(f"Unique values in the selected {selected_factor1} column:", unique_values_factor1)

    with col2:
        selected_factor2 = st.selectbox('Select a factor (2)', available_factors, key='factor2')
        unique_values_factor2 = data_visualize_B[selected_factor2].unique()
        st.write(f"Unique values in the selected {selected_factor2} column:", unique_values_factor2)
        
    with col3:
        selected_factor3 = st.selectbox('Select a factor (3)', available_factors, key='factor3')
        unique_values_factor3 = data_visualize_B[selected_factor3].unique()
        st.write(f"Unique values in the selected {selected_factor3} column:", unique_values_factor3)

    # Display box plots
    st.subheader("Box Plots")
    fig_box = px.box(data_visualize_B, x=selected_factor1, y=selected_factor2, title=f"{selected_factor1} {selected_factor2}")
    st.plotly_chart(fig_box)

    # Display histogram
    st.subheader("Bar Chart")
    fig_bar = px.histogram(data_visualize_B,x=selected_factor1, y=selected_factor2, color=selected_factor3, title=f"Employment by {selected_factor1}", barmode='group')   
    st.plotly_chart(fig_bar)
    
    # Display line graph
    st.subheader("Line Graph")
    fig_line = px.line(data_visualize_B, x=selected_factor1, y=selected_factor2, color=selected_factor3, title=f"{selected_factor1} {selected_factor2}")
    st.plotly_chart(fig_line)
    
    # Display Scatter 3D graph
    st.subheader("Scatter 3D Plot")
    fig_scatter_3d = px.scatter_3d(data_visualize_B, x=selected_factor1, y=selected_factor2, z=selected_factor3, title=f"{selected_factor1} vs {selected_factor2} vs {selected_factor3}")
    fig_scatter_3d.update_layout(height=800, width=1000)
    st.plotly_chart(fig_scatter_3d)












elif tabs == 'Geographic':
    st.write("hello")

# Demographic tab
elif tabs == 'Demographic':
    st.header("Demographic Section")
    selected_factor = st.selectbox('Select a factor', ['gender', 'ethnicity', 'race'])
    if selected_factor == 'gender':
        data_visualize_K.loc[~data_visualize_K['gender'].isin(['Male', 'Female']), 'gender'] = 'Other gender'
        description_box = "Observations: People identifying as male tend to have the highest salaries followed by women and other genders."
        description_bar = "Observations: Highest percentage of all genders are professionals which could also mean that there is just more data on professionals. Some noteworthy comparisons is that there are more men in skilled craft than female and other gender. women least common professions are skilled craft and service maintenance. men least common are technicians and protective service while other genders least common are administrative support, technicians and skilled craft  "
    elif selected_factor == 'ethnicity':
        description_box = "Non-hispanic or latino tend to be paid the highest. there may be data bias because of the people choosing not to report their ethnicity"
        description_bar = "Ignoring the common professionals,Non-hispanic or latino populations tend to work as paraprofessionals and officials/administrators and least in protective service and skilled craft. while hispanic or latinos are generally the same the difference in paraprofessionals and officials in less which means either of those categories are more common, while in non-hispanic/latinos more tend to lean towards paraprofessionals than officials/administrators   "
    elif selected_factor == 'race':
        description_box = "highest paid race is White while lowest being native hawaian. again this could just correspond to their populations in the US. Moreover this the chart is only showing upper pay bound the average pay could be a different story."
        description_bar = "some notable observations would be white working more in skilled craft than other races, asians and black winning the technicians profession, asians and white significantly more administrators/officials than paraprofessionals. "


        

    # Display box plot and pie chart
    st.subheader("Box Plot")
    fig_box = px.box(data_visualize_K, x=selected_factor, y="upper_pay_band_bound", title=f"Pay Distribution by {selected_factor}")
    st.write(description_box)
    st.plotly_chart(fig_box)

    st.subheader("Bar Plot")
    fig_bar = px.bar(data_visualize_K, x=selected_factor,color="job_category", barmode ="group",title=f"Distribution of Job Categories by {selected_factor}")
    st.write(description_bar)
    st.plotly_chart(fig_bar)

# Find Your Perfect Career Sector tab
elif tabs == 'Find Your Perfect Career Sector':
    monster_df = pd.read_csv('data/monster.csv')

    # Load the model and initialize TfidfVectorizer
    filename = 'pickled_models/finalized_model.sav'
    filename2 = 'pickled_models/finalized_vector.sav'
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

            # Filter dataset by predicted sectors
        selected_sectors = [pred[0] for pred in top_predictions]
        filtered_df = monster_df[monster_df['sector'].isin(selected_sectors)]
        st.subheader("Top States in Predicted Sectors")
        for sector in selected_sectors:
            sector_df = filtered_df[filtered_df['sector'] == sector]
            states_count = sector_df['location'].apply(lambda x: x.split(',')[1].strip() if len(x.split(',')) >= 2 else x.strip()).value_counts()
            top_states = states_count.head(10)
        
        # Create pie chart
            fig_pie = px.pie(values=top_states.values, names=top_states.index, title=f"Top 10 States in '{sector}'")
            st.plotly_chart(fig_pie)
 
     # Display top 10 job titles in each predicted sector
        st.subheader("Top 10 Job Titles in Predicted Sectors")
        for sector in selected_sectors:
            top_jobs = filtered_df[filtered_df['sector'] == sector]['job_title'].value_counts().head(10)
            st.write(f"Top 10 Job Titles in '{sector}':")
            st.write(top_jobs)
    

elif tabs == 'Generate Cover Letter':
    st.subheader('Add your Resume and job description to get a tailored cover letter and updated resume.')
    job_desc = st.text_area("Copy paste the job description you're interested in")

    # File uploader section
    uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx"])
    # Check if a file was uploaded
    if uploaded_file is not None:
        st.write("File uploaded successfully!")
    
    # Process the uploaded file, for example, you can read its contents
        file_contents = uploaded_file.read()
    
    # # Display the file contents
    #     st.write(file_contents)