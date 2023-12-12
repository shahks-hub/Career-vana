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
import pyperclip
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())



tabs = st.sidebar.radio("Select a tab", ( 'CareerProphet', 'JobProphet'))

# Main content
st.title(":blue[Predict], Explore, :violet[*Achieve!*] :sparkles:")
st.subheader("*with* :rainbow[Career-vana]")
with st.expander("How to Navigate our Website"):
    st.write("Add your career interests in the CareerProphet tab, go to the job titles and pick one you're interested in, expand that one and copy your favorite job description. Then move on to the JobProphet tab and paste it in the job description input box!")

##Helper functions

#function to load data
@st.cache_data  
def load_data(url):
    df = pd.read_csv(url)
    return df

#function to apply professional style to the generated cover letter
@st.cache_data
def get_css_styles():
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
    return css_styles 

#function to extract text from a pdf resume
@st.cache_data
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    page = reader.pages[0]
    text = page.extract_text()
    return text


##Loading all data in dataframe
data_visualize_K = load_data('data/LocalPayNYC.csv')
monster_df = load_data('data/monster_jobs.csv')


#Find Your Perfect Career Sector tab

#This tab utilizes a pre-trained machine learning model,
 #which has been pickled after being trained on a comprehensive dataset of job descriptions. 
 #The model's objective is to predict career sectors based on user-provided interests. 
##Additionally, it presents insightful visualizations derived from the predicted sectors, 
#showcasing common job titles and prevalent locations associated with those sectors.


if tabs == 'CareerProphet':
   

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
  
    # Text preprocessing pipeline function for TfidfVectorizer
    def text_pipeline(input_string):
        input_string = make_lower(input_string)
        input_string = remove_punctuation(input_string) 
        return input_string

    # Streamlit app section
    st.subheader("Find Your Perfect Career Sector 	:100:")
    with st.expander("See explanation 	:hibiscus:"):
        st.write("This tab utilizes a pre-trained machine learning model,which has been pickled by us after being trained on a comprehensive dataset of job descriptions and sectors using Natural language processing. The model's objective is to predict career sectors based on user-provided interests. Additionally, it presents insightful visualizations derived from the predicted sectors while matching predicted sectors to the training dataset, showcasing common job titles and prevalent locations associated with those sectors.")

    # Text input area
    user_input = st.text_area("Enter your interests here: ")

    if st.button("Predict 	:crystal_ball:"):
        # Process the user input and pick the top predictions
        processed_input = text_pipeline(user_input)
        X = loaded_vector.transform([processed_input])
        predictions_proba = loaded_model.predict_proba(X)
        classes = loaded_model.classes_
        proba = predictions_proba[0]
        combined = list(zip(classes, proba))
        sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
        top_predictions = sorted_combined[:1]
        for i, (predicted_class, probability) in enumerate(top_predictions, start=1):
            st.write(f":rainbow[*Prediction* :] Career path :blue['{predicted_class}']")

        
         # Filter dataset by predicted sectors
        selected_sectors = [pred[0] for pred in top_predictions]
        filtered_df = monster_df[monster_df['sector'].isin(selected_sectors)]

         # Display dropdown for selecting sectors
        selected_sector = top_predictions[0][0]

        st.subheader("Top :red[States] in Predicted :red[Sector]")
       
        # make a new df for the map based on location counts for the predicted sector
        sector_df = filtered_df[filtered_df['sector'] == selected_sector]
        states_count = sector_df['states'].value_counts()
        states_count = states_count.reset_index()
        states_count.columns = ['states', 'count']
        merged_df = pd.merge(sector_df[['states']], states_count, on='states', how='inner')
        

        # Map plotting
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
       
      
        sector_df = filtered_df[filtered_df['sector'] == selected_sector]
        states_count = sector_df['location_state'].value_counts().head(10)
        fig_pie = px.pie(values=states_count.values, names=states_count.index, title=f"Top 10 States in '{selected_sector}'")
        st.plotly_chart(fig_pie)

   
                # Display top 10 job titles in each predicted sector
        st.subheader("Top 10 :red[Job Titles] in Predicted :red[Sector]")
           
        top_jobs = filtered_df[filtered_df['sector'] == selected_sector]['job_title'].value_counts().head(10)
        st.write(f"Top 10 Job Titles in '{selected_sector}':")

        for job_title in top_jobs.index:
                    # Fetch job descriptions for the selected sector and job title
            job_descs = filtered_df[(filtered_df['sector'] == selected_sector) & (filtered_df['job_title'] == job_title)]['job_description'].values
                    
                    # Create an expandable section for each job title
            with st.expander(f"{job_title} - {len(job_descs)} Descriptions"):
                for idx, job_desc in enumerate(job_descs, start=1):
                    desc_words = job_desc.split()[:50]
                    truncated_desc = ' '.join(desc_words)
                    st.write(f"Description {idx}: {truncated_desc}...")

       

                
                        


        

        

#JobProphet tab

#This tab offers a multifaceted functionality by accepting user inputs in the form of a job description and a PDF resume.
#It extracts textual content from the uploaded PDF file and employs the OpenAI API to process this combined information.
#Subsequently, the tab provides an analysis of job trends based on demographics, visualizing salaries and sectors.
#Moreover, it uses a Hugging Face model to classify resumes,
#making it possible to unlock career suggestions based on the resume content. 
#The combination of these functionalities facilitates a comprehensive understanding of job trends while utilizing 
#cutting-edge AI models for resume classification and career suggestion generation.



elif tabs == 'JobProphet':
    API_URL = "https://api-inference.huggingface.co/models/runaksh/ResumeClassification_distilBERT"
    API_TOKEN = os.getenv('API_TOKEN')  
    openai.api_key  = os.getenv('OPENAI_API_KEY')
    client = OpenAI()
   

    st.subheader('Resume classifier 	:memo: and Cover Letter Generator 	:printer:')
    with st.expander("See explanation 	:hibiscus:"):
        st.write("This tab offers a multifaceted functionality by accepting user inputs in the form of a job description and a PDF resume.It extracts textual content from the uploaded PDF file and employs the OpenAI API to process this combined information.Subsequently, the tab provides an analysis of job trends based on demographics, visualizing salaries and sectors. Moreover, it uses a Hugging Face model to classify resumes, making it possible to unlock career suggestions based on the resume content. The combination of these functionalities facilitates a comprehensive understanding of job trends while utilizing cutting-edge machine learning models for resume classification and career suggestion generation.")
    
    job_desc = st.text_area("Copy paste the job description you're interested in")

   
    uploaded_file = st.file_uploader("Upload your resume", type=["pdf"])
    
     #extract text from pdf
    if uploaded_file is not None:
        st.write("File uploaded successfully! ")
        text = extract_text_from_pdf(uploaded_file)
       
    # query to send the resume content to the hugging face Inference API 
    def query(payload):
        response = requests.post(API_URL, headers={"Authorization": f"Bearer {API_TOKEN}"}, json=payload)
        return response.json()

    # loading predictions from the hugging face model
    if st.button("Unlock Career Suggestions 	:unlock:"):
        if text:
            answer = query(text)
            labels = [item['label'] for item in answer[0][:3]]
                # Displaying labels with corresponding sectors
            for idx, label in enumerate(labels, start=1):
                st.write(f"Sector :blue[{idx}:] :rainbow[{label}]")
        else:
                st.write("Please upload a valid PDF file to extract text.")

    # generates tailored cover letter with open AI API
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


                css = get_css_styles()
                st.markdown(css, unsafe_allow_html=True)
                st.subheader(":rainbow[Your Tailored Cover Letter]")
                st.download_button('Download cover letter :envelope_with_arrow:', answer)
                st.markdown(f'<div class="cover-letter">{answer}</div>', unsafe_allow_html=True)
                    
   # displays visualizations by gender, ethnicity, race
    if st.checkbox("Visualize Job Trends 	:bar_chart: :chart_with_upwards_trend:"):
               
               
                st.header(":rainbow[Job Trends by demographics,] :red[Target City: NYC]")
            
                selected_factor = st.selectbox('Select a factor', ['gender', 'ethnicity', 'race'])
                
                descriptions = {
                    'gender': {
                        'box': "People identifying as male tend to have the highest salaries followed by women and other genders.",
                        'bar': "Highest percentage of all genders are professionals which could also mean that there is just more data on professionals. Some noteworthy comparisons is that there are more men in skilled craft than female and other gender. women least common professions are skilled craft and service maintenance. men least common are technicians and protective service while other genders least common are administrative support, technicians and skilled craft  "
                    },
                    'ethnicity': {
                        'box': "Non-hispanic or latino tend to be paid the highest. there may be data bias because of the people choosing not to report their ethnicity",
                        'bar': "Ignoring the common professionals,Non-hispanic or latino populations tend to work as paraprofessionals and officials/administrators and least in protective service and skilled craft. while hispanic or latinos are generally the same the difference in paraprofessionals and officials in less which means either of those categories are more common, while in non-hispanic/latinos more tend to lean towards paraprofessionals than officials/administrators   "
                    },
                    'race': {
                        'box': "highest paid race is White while lowest being native hawaian. again this could just correspond to their populations in the US. Moreover this the chart is only showing upper pay bound the average pay could be a different story.",
                        'bar': "some notable observations would be white working more in skilled craft than other races, asians and black winning the technicians profession, asians and white significantly more administrators/officials than paraprofessionals. "
                    }
                }



                #combining all other options in the gender columns as "Other gender"
                if selected_factor == 'gender':
                    data_visualize_K.loc[~data_visualize_K['gender'].isin(['Male', 'Female']), 'gender'] = 'Other Gender'
                    
                #display bar plot
                st.subheader("The :blue[Bar] is high :blue[Plot]")
                fig_bar = px.bar(data_visualize_K, x=selected_factor,color="job_category", barmode ="group",title=f"Distribution of Job Categories by {selected_factor}")
                st.plotly_chart(fig_bar)
                with st.expander("See explanation"):
                    st.write(descriptions[selected_factor]['bar'])

                # Display box plot
                st.subheader("Don't fit in the :blue[Box Plot]")
                fig_box = px.box(data_visualize_K, x=selected_factor, y="upper_pay_band_bound", title=f"Pay Distribution by {selected_factor}")
                st.plotly_chart(fig_box)
                with st.expander("See explanation"):
                    st.write(descriptions[selected_factor]['box'])




