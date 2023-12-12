import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer  
import openai
import os
from openai import OpenAI
from PyPDF2 import PdfReader 
import requests
import base64
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())



tabs = st.sidebar.radio("Select a tab", ( 'CareerProphet', 'JobProphet'))

# Main content
st.title(":blue[Predict], Explore, :violet[*Achieve!*] :sparkles:")
st.subheader("*with* :rainbow[Career-vana]")


# Find Your Perfect Career Sector tab
if tabs == 'CareerProphet':
    monster_df = pd.read_csv('data/monster_jobs.csv')

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
    st.subheader("Find Your Perfect Career Sector 	:100:")

    # Text input area
    user_input = st.text_area("Enter your interests here: ")

    if st.button("Predict 	:crystal_ball:"):
        # Process the user input
        processed_input = text_pipeline(user_input)
        X = loaded_vector.transform([processed_input])
        predictions_proba = loaded_model.predict_proba(X)
        classes = loaded_model.classes_
        proba = predictions_proba[0]
        combined = list(zip(classes, proba))
        sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
        top_predictions = sorted_combined[:2]
        for i, (predicted_class, probability) in enumerate(top_predictions, start=1):
            st.write(f":rainbow[*Prediction* {i}:] Career path :blue['{predicted_class}']")

        
         # Filter dataset by predicted sectors
        selected_sectors = [pred[0] for pred in top_predictions]
        filtered_df = monster_df[monster_df['sector'].isin(selected_sectors)]
         # Display dropdown for selecting sectors
        selected_sector = top_predictions[0][0]

        st.subheader("Top :red[States] in Predicted :red[Sectors]")

        sector_df = filtered_df[filtered_df['sector'] == selected_sector]
        states_count = sector_df['states'].value_counts()
        states_count = states_count.reset_index()
        states_count.columns = ['states', 'count']
        merged_df = pd.merge(sector_df[['states']], states_count, on='states', how='inner')
        

        
        fig_heatmap = px.choropleth(
            merged_df,
            locations='states',
            locationmode="USA-states",
            color='count',
            scope="usa",
            color_continuous_scale="Reds",
            title=f"Heatmap for '{selected_sector}' Jobs by State"
        )

        fig_heatmap.update_layout(
            title_text=f"Heatmap for '{selected_sector}' Jobs by State",
            geo=dict(
                lakecolor='LightBlue',
                landcolor='LightGreen',
            ),
         )

        st.plotly_chart(fig_heatmap)

        
       ###MAKE PIE CHARTS FOR PREDICTED SECTORS 
        # st.subheader("Top :red[States] in Predicted :red[Sectors]")
        for sector in selected_sectors:
            sector_df = filtered_df[filtered_df['sector'] == sector]
            states_count = sector_df['location_state'].value_counts().head(10)
        
       
            fig_pie = px.pie(values=states_count.values, names=states_count.index, title=f"Top 10 States in '{sector}'")
            st.plotly_chart(fig_pie)


     # Display top 10 job titles in each predicted sector
        st.subheader("Top 10 :red[Job Titles] in Predicted :red[Sectors]")
        for sector in selected_sectors:
            top_jobs = filtered_df[filtered_df['sector'] == sector]['job_title'].value_counts().head(10)
            st.write(f"Top 10 Job Titles in '{sector}':")
            st.write(top_jobs)


        

        



    



elif tabs == 'JobProphet':
    API_URL = "https://api-inference.huggingface.co/models/runaksh/ResumeClassification_distilBERT"
    API_TOKEN = os.getenv('API_TOKEN')  
    openai.api_key  = os.getenv('OPENAI_API_KEY')
    client = OpenAI()
   

    st.subheader('Resume classifier 	:memo: and Cover Letter Generator 	:printer:')
    
    job_desc = st.text_area("Copy paste the job description you're interested in")

   
    uploaded_file = st.file_uploader("Upload your resume", type=["pdf"])
    
     #extract text from pdf
    if uploaded_file is not None:
        st.write("File uploaded successfully! ")
        reader = PdfReader(uploaded_file)
        page = reader.pages[0]
        text = page.extract_text()

    def query(payload):
        response = requests.post(API_URL, headers={"Authorization": f"Bearer {API_TOKEN}"}, json=payload)
        return response.json()


    if st.button("Unlock Career Suggestions 	:unlock:"):
        if text:
            answer = query(text)
            labels = [item['label'] for item in answer[0][:3]]
                # Displaying labels with corresponding sectors
            for idx, label in enumerate(labels, start=1):
                st.write(f"Sector :blue[{idx}:] :rainbow[{label}]")
        else:
                st.write("Please upload a valid PDF file to extract text.")

    
    if st.button("Generate AI Crafted Cover Letter 	:robot_face:"):

            if job_desc and uploaded_file is not None:
                prompt = f"Create a personalized cover letter based on the provided job description: {job_desc} and resume: {text} . Incorporate relevant details such as previous experience, skills, education, contact information (email and address) from the resume. Extract the company name and the position requirements from the job description to craft a tailored cover letter that highlights the qualifications in the resume and aligns with the job role."

                messages = [{"role": "user", "content": prompt}]
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0, 
                )
                answer = response.choices[0].message.content


                css_styles = """
                <style>
                    .cover-letter {
                        font-family: Arial, sans-serif;
                        font-size: 12pt;
                        line-height: 1.6;
                        margin-bottom: 20px;
                        padding: 20px;
                        border: 1px solid #ccc;
                        border-radius: 5px;
                        background-color: #f9f9f9;
                        color: #333;
                    }
                    .cover-letter h1 {
                        font-size: 18pt;
                        margin-bottom: 20px;
                    }
                 </style>
                """
                
                st.markdown(css_styles, unsafe_allow_html=True)
                st.subheader(":rainbow[Your Tailored Cover Letter]")
                st.download_button('Download cover letter', answer)
                st.markdown(f'<div class="cover-letter">{answer}</div>', unsafe_allow_html=True)
                    
   
    if st.checkbox("Visualize Job Trends 	:bar_chart: :chart_with_upwards_trend:"):
                
                data_visualize_K = pd.read_csv('data/LocalPayNYC.csv')
                st.header(":rainbow[Job Trends by demographics,] :red[Target City: NYC]")
            
                selected_factor = st.selectbox('Select a factor', ['gender', 'ethnicity', 'race'])
                
                if selected_factor == 'gender':
                    data_visualize_K.loc[~data_visualize_K['gender'].isin(['Male', 'Female']), 'gender'] = 'Other gender'
                    description_box = "People identifying as male tend to have the highest salaries followed by women and other genders."
                    description_bar = "Highest percentage of all genders are professionals which could also mean that there is just more data on professionals. Some noteworthy comparisons is that there are more men in skilled craft than female and other gender. women least common professions are skilled craft and service maintenance. men least common are technicians and protective service while other genders least common are administrative support, technicians and skilled craft  "
                elif selected_factor == 'ethnicity':
                    description_box = "Non-hispanic or latino tend to be paid the highest. there may be data bias because of the people choosing not to report their ethnicity"
                    description_bar = "Ignoring the common professionals,Non-hispanic or latino populations tend to work as paraprofessionals and officials/administrators and least in protective service and skilled craft. while hispanic or latinos are generally the same the difference in paraprofessionals and officials in less which means either of those categories are more common, while in non-hispanic/latinos more tend to lean towards paraprofessionals than officials/administrators   "
                elif selected_factor == 'race':
                    description_box = "highest paid race is White while lowest being native hawaian. again this could just correspond to their populations in the US. Moreover this the chart is only showing upper pay bound the average pay could be a different story."
                    description_bar = "some notable observations would be white working more in skilled craft than other races, asians and black winning the technicians profession, asians and white significantly more administrators/officials than paraprofessionals. "


                st.subheader("The :blue[Bar] is high :blue[Plot]")
                fig_bar = px.bar(data_visualize_K, x=selected_factor,color="job_category", barmode ="group",title=f"Distribution of Job Categories by {selected_factor}")
               
                st.plotly_chart(fig_bar)
                with st.expander("See explanation"):
                    st.write(description_bar)

                # Display box plot and pie chart
                st.subheader("Don't fit in the :blue[Box Plot]")
                fig_box = px.box(data_visualize_K, x=selected_factor, y="upper_pay_band_bound", title=f"Pay Distribution by {selected_factor}")
                st.plotly_chart(fig_box)
                with st.expander("See explanation"):
                    st.write(description_box)




