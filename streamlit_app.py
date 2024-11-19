import streamlit as st
import pandas as pd
from openai import OpenAI
import json
import requests
from typing import Dict, List
import os
from datetime import datetime
from docx import Document

# Streamlit configuration
st.set_page_config(page_title="International Student Assistant", layout="wide")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'university_data' not in st.session_state:
    st.session_state.university_data = None
if 'expenses_data' not in st.session_state:
    st.session_state.expenses_data = None
if 'employment_data' not in st.session_state:
    st.session_state.employment_data = None

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["open-key"])
OPENWEATHER_API_KEY = st.secrets["open-weather"]

def read_docx(file_path: str) -> str:
    """Read content from a Word document"""
    try:
        doc = Document(file_path)
        full_text = []
        for paragraph in doc.paragraphs:
            full_text.append(paragraph.text)
        return '\n'.join(full_text)
    except Exception as e:
        st.error(f"Error reading Word document: {str(e)}")
        return ""

def load_data():
    """Load all necessary data files"""
    try:
        # Load CSV files
        expenses_df = pd.read_csv("/workspaces/COMPASS/data/Avg_Living_Expenses.csv")
        employment_df = pd.read_csv("/workspaces/COMPASS/data/Employment_Projections.csv")
        
        # Load Word document content
        university_data = read_docx("/workspaces/COMPASS/data/University_Data.docx")
        
        if not university_data:
            st.warning("University data could not be loaded. Operating with limited information.")
            university_data = "University information temporarily unavailable."
        
        return university_data, expenses_df, employment_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

def get_state_weather(location: str) -> Dict:
    """Fetch weather data for a given location"""
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location},US&appid={OPENWEATHER_API_KEY}&units=imperial"
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            return {
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"]
            }
        else:
            return {"error": "Weather data not available"}
    except Exception as e:
        return {"error": str(e)}

def fetch_living_expenses(state: str, expenses_df: pd.DataFrame) -> Dict:
    """Fetch living expenses data for a given state"""
    try:
        state_data = expenses_df[expenses_df['State'].str.contains(state, case=False, na=False)]
        if len(state_data) > 0:
            return state_data.iloc[0].to_dict()
        return {"error": "State not found"}
    except Exception as e:
        return {"error": str(e)}

def fetch_job_market_trends(field: str, employment_df: pd.DataFrame) -> Dict:
    """Fetch job market trends for a specific field"""
    try:
        field_data = employment_df[employment_df['Occupation Title'].str.contains(field, case=False, na=False)]
        if len(field_data) > 0:
            return field_data.iloc[0].to_dict()
        return {"error": "Field not found"}
    except Exception as e:
        return {"error": str(e)}

def get_chatbot_response(user_input: str, context: Dict, university_data: str, 
                        expenses_df: pd.DataFrame, employment_df: pd.DataFrame) -> str:
    """Get response from OpenAI API"""
    try:
        # Get relevant data based on context
        state_weather = get_state_weather(context['preferred_state'])
        living_expenses = fetch_living_expenses(context['preferred_state'], expenses_df)
        job_trends = fetch_job_market_trends(context['study_field'], employment_df)
        
        # Create messages list for chat completion
        messages = [
            {
                "role": "system",
                "content": f"""You are an International Student Assistant helping students choose universities in the United States.
                Use the following information to provide detailed responses:
                
                Student Context:
                - Field of Study: {context['study_field']}
                - Preferred State: {context['preferred_state']}
                - Monthly Budget: ${context['budget_range'][0]}-${context['budget_range'][1]}
                
                State Weather: {json.dumps(state_weather, indent=2)}
                Living Expenses: {json.dumps(living_expenses, indent=2)}
                Job Market Trends: {json.dumps(job_trends, indent=2)}
                
                University Information: {university_data[:2000]}  # Truncated for context window
                
                Provide specific, relevant information based on the student's interests and needs.
                Focus on practical advice and accurate information about universities, living costs, and career prospects."""
            }
        ]
        
        # Add chat history (last 5 messages)
        for msg in st.session_state.chat_history[-5:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add user's current question
        messages.append({"role": "user", "content": user_input})
        
        # Get response from OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Extract the response content
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    st.title("International Student Assistant")
    
    # Load data if not already loaded
    if not st.session_state.data_loaded:
        university_data, expenses_df, employment_df = load_data()
        if university_data is not None and expenses_df is not None and employment_df is not None:
            st.session_state.university_data = university_data
            st.session_state.expenses_data = expenses_df
            st.session_state.employment_data = employment_df
            st.session_state.data_loaded = True
    
    # Sidebar inputs
    st.sidebar.header("Your Preferences")
    study_field = st.sidebar.selectbox(
        "Field of Study", 
        ["Computer Science", "Engineering", "Business", "Data Science", "Other"]
    )
    preferred_state = st.sidebar.selectbox(
        "Preferred State", 
        ["California", "New York", "Texas", "Massachusetts", "Other"]
    )
    budget_range = st.sidebar.slider(
        "Monthly Budget (USD)", 
        1000, 5000, (2000, 3000)
    )
    
    # Main chat interface
    st.write("Chat with me about universities, scholarships, and life in the US!")
    
    # Chat input
    if user_input := st.chat_input("Type your message here..."):
        # Save user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get context
        context = {
            "study_field": study_field,
            "preferred_state": preferred_state,
            "budget_range": budget_range
        }
        
        # Get chatbot response
        if st.session_state.data_loaded:
            response = get_chatbot_response(
                user_input=user_input,
                context=context,
                university_data=st.session_state.university_data,
                expenses_df=st.session_state.expenses_data,
                employment_df=st.session_state.employment_data
            )
            
            # Save assistant response
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            st.error("Data not loaded properly. Please check the data files.")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

if __name__ == "__main__":
    main()