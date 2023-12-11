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







data_visualize_K = pd.read_csv('data/LocalPayNYC.csv')
data_visualize_B = pd.read_csv('data/b_dataMod.csv')

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
        selected_factor1 = st.selectbox('This factor appears on X-Axis', available_factors, key='factor1')
        unique_values_factor1 = data_visualize_B[selected_factor1].unique()
        st.write(f"Unique values in the selected {selected_factor1} column:", unique_values_factor1)

    with col2:
        selected_factor2 = st.selectbox('This factor appears on Y-Axis', available_factors, key='factor2')
        unique_values_factor2 = data_visualize_B[selected_factor2].unique()
        st.write(f"Unique values in the selected {selected_factor2} column:", unique_values_factor2)
        
    with col3:
        selected_factor3 = st.selectbox('This factor provides color', available_factors, key='factor3')
        unique_values_factor3 = data_visualize_B[selected_factor3].unique()
        st.write(f"Unique values in the selected {selected_factor3} column:", unique_values_factor3)

    # Display box plots
    st.subheader("Box Plots")
    fig_box = px.box(data_visualize_B, x=selected_factor1, y=selected_factor2, color=selected_factor3, title=f"{selected_factor1} {selected_factor2}")
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
    st.subheader("Line 3D Chart")
    fig_line_3d = px.line_3d(data_visualize_B, x=selected_factor1, y=selected_factor2, z=selected_factor3, title=f"{selected_factor1} vs {selected_factor2} vs {selected_factor3}")
    fig_line_3d.update_layout(height=800, width=1000)
    st.plotly_chart(fig_line_3d)
    
    # Display Scatter 3D graph
    st.subheader("Scatter 3D Plot")
    fig_scatter_3d = px.scatter_3d(data_visualize_B, x=selected_factor1, y=selected_factor2, z=selected_factor3, title=f"{selected_factor1} vs {selected_factor2} vs {selected_factor3}")
    fig_scatter_3d.update_layout(height=800, width=1000)
    st.plotly_chart(fig_scatter_3d)




elif tabs == 'Geographic':
    st.header("Geographic Section")

    # Load data
    monster_jobs_df = pd.read_csv('data/monster_jobs.csv')

    state_name_to_code = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
        'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
        'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
        'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
        'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
        'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
        'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
        'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
        'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
        'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
        'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
        'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
        'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC'
    }

    monster_jobs_df['state_code'] = monster_jobs_df['cleaned_states'].map(state_name_to_code)

    df_sector = monster_jobs_df[['state_code', 'sector']]
    unique_sectors = df_sector['sector'].unique()
    colors = px.colors.qualitative.Plotly
    color_discrete_map_sector = {sector: color for sector, color in zip(unique_sectors, colors)}

    # Set up Streamlit dropdowns
    state_dropdown_sector = st.selectbox('Select a State:', list(state_name_to_code.keys()), key='state')
    sector_dropdown = st.selectbox('Select a Sector:', unique_sectors, key='sector')

    

      # Please dear god
    def update_map_sector(selected_state, selected_sector):
      

        
        filtered_df_sector = df_sector[(df_sector['state_code'] == selected_state) & (df_sector['sector'] == selected_sector)]

        
        fig_sector = px.choropleth(
            filtered_df_sector,
            locations='state_code',
            locationmode="USA-states",
            color='sector',
            color_discrete_map=color_discrete_map_sector,
            scope="usa"
        )
        fig_sector.update_layout(title_text=f'Sector: {selected_sector} by State in USA')

        st.plotly_chart(fig_sector)
   
    update_map_sector(state_name_to_code[state_dropdown_sector], sector_dropdown)

        

    

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
    monster_df = pd.read_csv('data/monster_jobs.csv')

    # Load the model and initialize TfidfVectorizer
    filename = 'pickled_models/finalized_model.sav'
    filename2 = 'pickled_models/finalized_vector.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    loaded_vector = pickle.load(open(filename2,'rb'))

    state_name_to_code = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
        'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
        'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
        'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
        'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
        'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
        'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
        'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
        'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
        'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
        'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
        'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
        'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC'
    }

    monster_df['state_code'] = monster_df['cleaned_states'].map(state_name_to_code)

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
        top_predictions = sorted_combined[:2]
        for i, (predicted_class, probability) in enumerate(top_predictions, start=1):
            st.write(f"Prediction {i}: Career path '{predicted_class}'")



         ####VISUALIZATIONS START HERE
            # Filter dataset by predicted sectors
        selected_sectors = [pred[0] for pred in top_predictions]
        filtered_df = monster_df[monster_df['sector'].isin(selected_sectors)]
         # Display dropdown for selecting sectors
        selected_sector = top_predictions[0][0]

        st.subheader(f"Heatmap for '{selected_sector}' Jobs by State")

        sector_df = filtered_df[filtered_df['sector'] == selected_sector]
        states_count = sector_df['cleaned_states'].value_counts()


        fig_heatmap = px.choropleth(
            sector_df,
            locations='state_code',
            locationmode="USA-states",
            color='cleaned_states',
            scope="usa",
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

        
       ###MAKE PIE CHARTS FOR PREDICTED SECTORS - henry you can try to use this logic to make maps instead
        st.subheader("Top States in Predicted Sectors")
        for sector in selected_sectors:
            sector_df = filtered_df[filtered_df['sector'] == sector]
            states_count = sector_df['cleaned_states'].value_counts().head(10)
        
       
            fig_pie = px.pie(values=states_count.values, names=states_count.index, title=f"Top 10 States in '{sector}'")
            st.plotly_chart(fig_pie)


     # Display top 10 job titles in each predicted sector
        st.subheader("Top 10 Job Titles in Predicted Sectors")
        for sector in selected_sectors:
            top_jobs = filtered_df[filtered_df['sector'] == sector]['job_title'].value_counts().head(10)
            st.write(f"Top 10 Job Titles in '{sector}':")
            st.write(top_jobs)
    



elif tabs == 'Generate Cover Letter':
    API_URL = "https://api-inference.huggingface.co/models/Sachinkelenjaguri/resume_classifier"
    API_TOKEN = os.getenv('API_TOKEN')  
    openai.api_key  = os.getenv('OPENAI_API_KEY')
    client = OpenAI()
    MAX_SEQUENCE_LENGTH = 512


    st.subheader('Add your Resume and job description to get a tailored cover letter')
    
    job_desc = st.text_area("Copy paste the job description you're interested in")

   
    uploaded_file = st.file_uploader("Upload your resume", type=["pdf"])
    
     #extract text from pdf
    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        reader = PdfReader(uploaded_file)
        page = reader.pages[0]
        text = page.extract_text()

   
    def get_sector_name(label):
        label_to_sector = {
            'LABEL_0': 'Advocate',
            'LABEL_1': 'Arts',
            'LABEL_2': 'Automation Testing',
            'LABEL_3': 'Blockchain Engineer',
            'LABEL_4': 'Business Analyst',
            'LABEL_5': 'Civil Engineer',
            'LABEL_6': 'Data Science',
            'LABEL_7': 'Database Management',
            'LABEL_8': 'DevOps Engineer',
            'LABEL_9': 'DotNet Developer',
            'LABEL_10': 'ETL Developer',
            'LABEL_11': 'Electrical Engineering',
            'LABEL_12': 'HR',
            'LABEL_13': 'Hadoop Developer',
            'LABEL_14': 'Health and fitness',
            'LABEL_15': 'Java Developer',
            'LABEL_16': 'Mechanical Engineer',
            'LABEL_17': 'Network Security Engineer',
            'LABEL_18': 'Operations Manager',
            'LABEL_19': 'Project Management',
            'LABEL_20': 'Python Developer',
            'LABEL_21': 'SAP Developer',
            'LABEL_22': 'Sales',
            'LABEL_23': 'Testing Operations',
            'LABEL_24': 'Web Designing'
        }
        return label_to_sector.get(label, 'Chilling/Unknown')


    def extract_top_5(classification_responses):
        if isinstance(classification_responses, list):
            top_5_classifications = []
            for response in classification_responses:
                response_json = response.json()
                sorted_responses = sorted(response_json, key=lambda x: x['score'], reverse=True)
                top_5_classifications.extend(sorted_responses[:5])

            top_5_classifications.sort(key=lambda x: x['score'], reverse=True)
    
            unique_sectors = set()
            top_5_unique_sectors = []
            for classification in top_5_classifications:
                sector = get_sector_name(classification['label'])
                if sector not in unique_sectors:
                    unique_sectors.add(sector)
                    top_5_unique_sectors.append({'sector': sector})
                    if len(top_5_unique_sectors) == 5:
                        break
    
            return top_5_unique_sectors
        else:
            response_json = classification_responses.json()
            sorted_responses = sorted(response_json, key=lambda x: x['score'], reverse=True)
            top_5_classifications = sorted_responses[:5]
        
            unique_sectors = set()
            top_5_unique_sectors = []
            for classification in top_5_classifications:
                sector = get_sector_name(classification['label'])
                if sector not in unique_sectors:
                    unique_sectors.add(sector)
                    top_5_unique_sectors.append({'sector': sector})
                    if len(top_5_unique_sectors) == 5:
                        break
    
            return top_5_unique_sectors



 
    if st.button("Step 1: Make sure your resume aligns to your sector "):

            if uploaded_file:
                data = {"text" : text}
                if len(text) > MAX_SEQUENCE_LENGTH:
                    chunks = [text[i:i+MAX_SEQUENCE_LENGTH] for i in range(0, len(text), MAX_SEQUENCE_LENGTH)]
                    responses = []
                    for chunk in chunks:
                        
                        response = requests.post(API_URL, headers={"Authorization": f"Bearer {API_TOKEN}"}, json={"text": chunk})
                        responses.append(response)

                    top_5_classifications = extract_top_5(responses)  # Function to extract top 5 classifications
                  
                else:
                   
                    response = requests.post(API_URL, headers={"Authorization": f"Bearer {API_TOKEN}"}, json=data)
                    top_5_classifications = extract_top_5(response) 


                st.subheader(f"Top 5 Resume Classifications:")
                for classification in top_5_classifications:
                    st.markdown(f'<div style="background-color:#FAF0E6; font-size: 20px; font-weight: bold; color: #ff69b4; padding:10px; border-radius:5px;">{classification["sector"]}</div>', unsafe_allow_html=True)
                    
            else:
                st.write("Please upload a resume first.")



    
    if st.button("Step 2: Generate Cover Letter"):
            if job_desc and uploaded_file is not None:
                prompt = f"Create a personalized cover letter based on the provided job description: {job_desc} and resume: {text} . Incorporate relevant details such as previous experience, skills, education, contact information (email and address) from the resume. Extract the company name and the position requirements from the job description to craft a tailored cover letter that highlights the qualifications in the resume and aligns with the job role."

                # prompt = f"Take this job description: {job_desc} and resume: {text} and write a cover letter tailored to it. Extract previous experience, skills, education, email, address from the resume and extract the company name and position from the job description."
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
                st.subheader("Your Tailored Cover Letter")
                st.markdown(f'<div class="cover-letter">{answer}</div>', unsafe_allow_html=True)


                if st.button("Download as PDF"):
                    pdf_content = answer
                    pdf_filename = "cover_letter.pdf"
                    b64 = base64.b64encode(pdf_content.encode()).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="{pdf_filename}">Download Cover Letter as PDF</a>'
                    st.markdown(href, unsafe_allow_html=True)


    


     # # Dark transparent background for astheics
        # background_color = "#0E1117"  
        # fig_sector.update_layout(
        #     paper_bgcolor=background_color,
        #     plot_bgcolor=background_color,
        #     geo=dict(
        #         bgcolor=background_color,
        #         lakecolor='LightBlue',  
        #         landcolor='LightGreen',  
        #     ),
        #     font=dict(
        #         family="Arial, sans-serif",
        #         size=12,
        #         color="#FFFFFF"  
        #     )
        # )