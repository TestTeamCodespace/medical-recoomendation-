import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set up the Streamlit app configuration
st.set_page_config(page_title="Medical Recommendation", page_icon="⛑️", layout="centered")

# Custom CSS for styling the app with animations and gradients
st.markdown(
    """
    <style>
    /* Use Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    /* Animated gradient background */
    body {
        background: linear-gradient(90deg, #ff4b4b, #ffcc00, #764ba2);
        background-size: 300% 300%;
        animation: gradientAnimation 15s ease infinite;
        font-family: 'Roboto', sans-serif;
    }

    @keyframes gradientAnimation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Premium button style with animation */
    .stButton > button {
        background: linear-gradient(135deg, #ffcc00, #ff4b4b);
        color: white;
        border-radius: 12px;
        font-size: 18px;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
        transition: all 0.4s ease;
        font-family: 'Roboto', sans-serif;
        font-weight: bold;
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2, #ff4b4b);
        transform: scale(1.05);
        box-shadow: 0 12px 20px rgba(0, 0, 0, 0.3);
    }

    /* Animated headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Roboto', sans-serif;
        color: black;
        animation: fadeIn 2s ease-in-out;
    }

    h3 {
        background: -webkit-linear-gradient(#ffcc00, #ff4b4b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(-20px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    /* Sidebar styling with gradient */
    .css-1lcbmhc {
        background-color: #3e4a61;
        color: white;
        border-radius: 10px;
        padding: 20px;
        animation: fadeIn 1.5s ease-in-out;
    }

    .css-1lcbmhc h1 {
        color: #ffffff;
        font-family: 'Roboto', sans-serif;
        font-weight: bold;
    }

    /* Smooth transition for tables */
    .dataframe {
        border-collapse: collapse;
        width: 100%;
        margin: 10px 0;
        background-color: white;
        border-radius: 10px;
        animation: fadeIn 1s ease-in-out;
    }

    .dataframe th, .dataframe td {
        text-align: left;
        padding: 10px;
    }

    .dataframe th {
        background-color: #4CAF50;
        color: white;
    }

    .dataframe tr:nth-child(even) {
        background-color: #f2f2f2;
    }

    /* Expander customization */
    .stExpander {
        border: 1px solid #ccc;
        border-radius: 10px;
        animation: fadeIn 1.5s ease-in-out;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar configuration for symptom selection
st.sidebar.title("Medical Recommendation App ")
st.sidebar.markdown("### Select your symptoms to get health recommendations:")

# Load CSV data for symptoms, descriptions, precautions, medications, workouts, and diets
try:
    symptoms_dict = pd.read_csv("DataSets/symptoms_dict.csv")  # Symptoms dictionary
    diseases_list = pd.read_csv("DataSets/diseases_list.csv")  # Diseases dictionary
    precautions_df = pd.read_csv("DataSets/precautions_df.csv")
    workout_df = pd.read_csv("DataSets/workout_df.csv")
    description_df = pd.read_csv("DataSets/description.csv")
    medications_df = pd.read_csv("DataSets/medications.csv")
    diets_df = pd.read_csv("DataSets/diets.csv")

    svc_model = pickle.load(open('svc.pkl', 'rb'))  # Load the trained SVM model
    num_symptoms = pd.read_csv('DataSets/Symptom-severity.csv')
    symptoms_list = num_symptoms['Symptom'].tolist()
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()

# Function to predict disease based on symptoms
def get_predicted_disease(symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms:
        if symptom in symptoms_dict['Symptom'].values:
            input_vector[symptoms_dict[symptoms_dict['Symptom'] == symptom].index[0]] = 1
    predicted_code = svc_model.predict([input_vector])[0]
    predicted_disease = diseases_list[diseases_list['Disease_Code'] == predicted_code]['Disease_Name'].values[0]
    return predicted_disease

# Helper function to fetch recommendations for a disease
def get_recommendations_for_disease(predicted_disease):
    desc = description_df[description_df['Disease'] == predicted_disease]['Description'].values[0]
    pre = precautions_df[precautions_df['Disease'] == predicted_disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values[0]
    med = medications_df[medications_df['Disease'] == predicted_disease]['Medication'].tolist()
    die = diets_df[diets_df['Disease'] == predicted_disease]['Diet'].tolist()
    wrkout = workout_df[workout_df['disease'] == predicted_disease]['workout'].tolist()
    return desc, pre, med, die, wrkout

# Function to get recommendations based on selected symptoms
def get_recommendations(selected_symptoms):
    predicted_disease = get_predicted_disease(selected_symptoms)
    desc, pre, med, die, wrkout = get_recommendations_for_disease(predicted_disease)
    return {
        'Disease': predicted_disease,
        'Description': desc,
        'Precautions': pre,
        'Medications': med,
        'Workout': wrkout,
        'Diets': die
    }

# Display multiselect dropdown with search functionality in the sidebar
selected_symptoms = st.sidebar.multiselect("Select symptoms:", symptoms_list)

# Button to trigger recommendation
if st.sidebar.button(" Recommend me"):
    if selected_symptoms:
        st.header(f"Predicted Disease: {get_predicted_disease(selected_symptoms)}")
        recommended_info = get_recommendations(selected_symptoms)
        # Display recommended information in collapsible sections
        with st.expander("Description"):
            st.markdown(f"<p style='color: #ffcc00'>{recommended_info['Description']}</p>", unsafe_allow_html=True)
        with st.expander("Precautions"):
            st.table(pd.DataFrame({'Start doing': recommended_info['Precautions']}))
        with st.expander("Medications"):
            st.table(pd.DataFrame({'Medications': recommended_info['Medications']}))
        with st.expander("Workouts"):
            st.table(pd.DataFrame({'Workouts': recommended_info['Workout']}))
        with st.expander("Diets"):
            st.table(pd.DataFrame({'Diets': recommended_info['Diets']}))

# Footer or closing message
st.markdown("### Powered by AI & Medical Data | Made by Harsh Attri ", unsafe_allow_html=True)
