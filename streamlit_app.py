import sys
import sqlite3
import platform

# SQLite fix for older versions
if sqlite3.sqlite_version_info < (3, 35, 0):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pysqlite3-binary"])
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import pandas as pd
import openai
import chromadb
from chromadb.config import Settings
import uuid
import requests
import docx2txt
from pathlib import Path
import json
from datetime import datetime

# Set page config
st.set_page_config(page_title="University Assistant", layout="wide")
st.title("International Student University Assistant")

# Initialize session states
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_chat' not in st.session_state:
    st.session_state.show_chat = False
if 'preferences' not in st.session_state:
    st.session_state.preferences = {}
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'initial_recommendations' not in st.session_state:
    st.session_state.initial_recommendations = None

# Constants
US_REGIONS = {
    "Northeast": ["Maine", "New Hampshire", "Vermont", "Massachusetts", "Rhode Island", "Connecticut", "New York", "Pennsylvania", "New Jersey"],
    "Southeast": ["Maryland", "Delaware", "Virginia", "West Virginia", "Kentucky", "Tennessee", "North Carolina", "South Carolina", "Georgia", "Florida", "Alabama", "Mississippi", "Arkansas", "Louisiana"],
    "Midwest": ["Ohio", "Indiana", "Illinois", "Michigan", "Wisconsin", "Minnesota", "Iowa", "Missouri", "North Dakota", "South Dakota", "Nebraska", "Kansas"],
    "Southwest": ["Texas", "Oklahoma", "New Mexico", "Arizona"],
    "West": ["Colorado", "Wyoming", "Montana", "Idaho", "Washington", "Oregon", "Utah", "Nevada", "California", "Alaska", "Hawaii"]
}

# Enhanced system prompt
SYSTEM_PROMPT = """You are a highly knowledgeable university advisor for international students. Your role is to provide detailed, actionable advice that helps students make informed decisions about their education in the United States. 

When providing initial recommendations:
1. Focus on the top 3 universities that best match the student's preferences
2. For each university provide:
   - Brief overview of the program strength
   - Specific costs and potential scholarships
   - Location benefits and climate match
   - Notable features or advantages
3. Keep the initial recommendations concise but informative

When answering follow-up questions:
1. Provide detailed, specific information about asked universities
2. Compare and contrast options when relevant
3. Include practical next steps and actionable advice
4. Focus on international student perspective

Remember to:
- Consider the complete student context (field, budget, location preferences)
- Be realistic about challenges and opportunities
- Provide evidence-based recommendations
- Maintain an encouraging and supportive tone"""

# Set API keys from Streamlit secrets
OPENAI_API_KEY = st.secrets["open-key"]
OPENWEATHER_API_KEY = st.secrets["open-weather"]

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

# User Authentication Functions
def initialize_user_db():
    """Initialize user database if it doesn't exist"""
    if not os.path.exists('user_data'):
        os.makedirs('user_data')
    if not os.path.exists('user_data/users.json'):
        with open('user_data/users.json', 'w') as f:
            json.dump({}, f)

def load_user_data(user_id):
    """Load user data from JSON file"""
    try:
        with open('user_data/users.json', 'r') as f:
            users = json.load(f)
        return users.get(user_id, None)
    except FileNotFoundError:
        initialize_user_db()
        return None

def save_user_data(user_id, data):
    """Save user data to JSON file"""
    try:
        with open('user_data/users.json', 'r') as f:
            users = json.load(f)
    except FileNotFoundError:
        users = {}
    
    users[user_id] = data
    
    with open('user_data/users.json', 'w') as f:
        json.dump(users, f, indent=4)

def save_current_session():
    """Save current session data for the user"""
    if hasattr(st.session_state, 'user_id') and st.session_state.user_id:
        user_data = {
            'preferences': st.session_state.preferences,
            'chat_history': st.session_state.chat_history[-10:],  # Save last 10 conversations
            'initial_recommendations': st.session_state.initial_recommendations,
            'last_login': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_user_data(st.session_state.user_id, user_data)

def login_page():
    """Display login page and handle user authentication"""
    st.write("### Welcome to University Assistant")
    st.write("Please enter your username to continue")
    
    user_id = st.text_input("Username:", key="user_id_input",
                           help="Enter your username to access your saved preferences")
    
    if user_id:
        user_data = load_user_data(user_id)
        
        if user_data:
            st.success(f"Welcome back, {user_id}! 👋")
            # Load user preferences and chat history
            st.session_state.preferences = user_data.get('preferences', {})
            st.session_state.chat_history = user_data.get('chat_history', [])
            st.session_state.initial_recommendations = user_data.get('initial_recommendations', None)
            st.session_state.user_id = user_id
            st.session_state.authenticated = True
            
            # Display last login time
            last_login = user_data.get('last_login', 'First time login')
            st.info(f"Last login: {last_login}")
            
            # Update last login time
            user_data['last_login'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_user_data(user_id, user_data)
            
            if st.button("Continue to Assistant"):
                st.experimental_rerun()
        else:
            st.info("New user detected! Let's set up your preferences.")
            st.session_state.user_id = user_id
            st.session_state.authenticated = True
            st.session_state.show_chat = False
            
            if st.button("Set Up Preferences"):
                st.experimental_rerun()
    
    return st.session_state.get('authenticated', False)

# Data Loading Functions
@st.cache_data
def load_data():
    """Load all necessary data files"""
    living_expenses_df = pd.read_csv("/workspaces/COMPASS/data/Avg_Living_Expenses.csv")
    employment_projections_df = pd.read_csv("/workspaces/COMPASS/data/Employment_Projections.csv")
    university_text = docx2txt.process("/workspaces/COMPASS/data/University_Data.docx")
    return living_expenses_df, employment_projections_df, university_text

def get_living_expenses_info(query):
    """Function to query living expenses data"""
    try:
        df = load_data()[0]  # Get living expenses DataFrame
        return str(df.query(query) if 'query' in query.lower() else df[df['State'].str.contains(query, case=False)])
    except Exception as e:
        return f"Error querying living expenses: {str(e)}"

def get_employment_info(query):
    """Function to query employment projections data"""
    try:
        df = load_data()[1]  # Get employment DataFrame
        return str(df[df['Occupation Title'].str.contains(query, case=False, na=False)])
    except Exception as e:
        return f"Error querying employment data: {str(e)}"

def get_state_weather(location):
    """Fetch weather data for a location"""
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": f"{location},US",
            "appid": OPENWEATHER_API_KEY,
            "units": "imperial"
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return {
                "location": location,
                "temperature": data["main"]["temp"],
                "description": data["weather"][0]["description"],
                "humidity": data["main"]["humidity"]
            }
        return f"Error getting weather for {location}"
    except Exception as e:
        return f"Error: {str(e)}"

def get_initial_recommendations(preferences, retrieval_chain):
    """Generate initial university recommendations based on preferences"""
    recommendation_prompt = f"""
    Based on the student's preferences, provide the top 3 recommended universities:

    STUDENT PROFILE:
    - Field of Study: {preferences['field_of_study']}
    - Yearly Budget: ${preferences['tuition_range'][0]:,} - ${preferences['tuition_range'][1]:,}
    - Preferred Regions: {', '.join(preferences['preferred_regions'])}
    - Weather Preference: {', '.join(preferences['preferred_weather'])}
    
    Additional Notes: {preferences.get('additional_notes', 'None provided')}

    Please provide:
    1. Top 3 universities that best match these preferences
    2. For each university include:
       - Program strengths and unique features
       - Estimated costs (tuition, living expenses)
       - Location benefits and climate
       - Notable scholarships or funding opportunities
    3. Brief explanation of why each university is recommended

    Format the response in a clear, easy-to-read manner with university names in bold.
    Keep the response focused and concise while maintaining important details.
    """
    
    try:
        response = retrieval_chain({"question": recommendation_prompt})
        return response['answer']
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"

@st.cache_resource
def initialize_chromadb(living_expenses_df, employment_projections_df, university_text):
    """Initialize or load ChromaDB"""
    persist_directory = "./chroma_db"
    
    # If ChromaDB already exists, load it
    if os.path.exists(persist_directory):
        try:
            return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        except Exception as e:
            st.warning(f"Error loading existing ChromaDB: {e}. Creating new database.")
            import shutil
            shutil.rmtree(persist_directory, ignore_errors=True)
    
    # Create new ChromaDB
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=persist_directory
    ))
    
    # Create collection
    collection = client.create_collection(name="university_assistant")
    
    # Process and add living expenses data
    for _, row in living_expenses_df.iterrows():
        doc = (f"State: {row['State']}, Living Expenses - Index: {row['Index']}, "
               f"Grocery: {row['Grocery']}, Housing: {row['Housing']}, "
               f"Utilities: {row['Utilities']}, Transportation: {row['Transportation']}, "
               f"Health: {row['Health']}, Miscellaneous: {row.get('Misc.', 'N/A')}")
        
        embedding = embeddings.embed_query(doc)
        
        collection.add(
            embeddings=[embedding],
            documents=[doc],
            metadatas=[{"type": "living_expenses", "state": row["State"]}],
            ids=[f"expenses_{uuid.uuid4()}"]
        )
    
    # Process and add employment projections data
    for _, row in employment_projections_df.iterrows():
        doc = (f"Occupation: {row['Occupation Title']}, "
               f"Employment Change (2023-2033): {row['Employment Change, 2023-2033']}, "
               f"Median Annual Wage: ${row['Median Annual Wage 2023']}, "
               f"Required Education: {row['Typical Entry-Level Education']}")
        
        embedding = embeddings.embed_query(doc)
        
        collection.add(
            embeddings=[embedding],
            documents=[doc],
            metadatas=[{"type": "employment", "occupation": row['Occupation Title']}],
            ids=[f"employment_{uuid.uuid4()}"]
        )
    
    # Process and add university data
    chunks = university_text.split('\n\n')
    for i, chunk in enumerate(chunks):
        if chunk.strip():
            embedding = embeddings.embed_query(chunk)
            
            collection.add(
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"type": "university", "index": i}],
                ids=[f"university_{uuid.uuid4()}"]
            )
    
    return Chroma(
        client=client,
        collection_name="university_assistant",
        embedding_function=embeddings
    )

def create_tools():
    """Create tools for the agent"""
    return [
        Tool(
            name="Weather Info",
            func=get_state_weather,
            description="Get weather information for a US state or city"
        ),
        Tool(
            name="Living Expenses",
            func=get_living_expenses_info,
            description="Get information about living expenses in different US states"
        ),
        Tool(
            name="Employment Info",
            func=get_employment_info,
            description="Get employment projections and job market trends"
        )
    ]
def display_preferences_form():
    """Display and handle the preferences form"""
    st.write("### Your Preferences")
    st.write(f"Setting up preferences for: **{st.session_state.user_id}**")
    
    with st.form("student_preferences"):
        col1, col2 = st.columns(2)
        
        with col1:
            field_of_study = st.text_input(
                "What is your field of study?",
                value=st.session_state.preferences.get('field_of_study', ''),
                help="Enter your specific field of study or research interest"
            )
            
            tuition_range = st.slider(
                "What is your yearly tuition budget (USD)?",
                10000, 70000, 
                value=st.session_state.preferences.get('tuition_range', (20000, 40000)),
                step=5000,
                help="Select your minimum and maximum yearly tuition budget"
            )
            
        with col2:
            preferred_regions = st.multiselect(
                "Which US regions are you interested in?",
                options=list(US_REGIONS.keys()),
                default=st.session_state.preferences.get('preferred_regions', []),
                help="Select one or more regions you're interested in"
            )
            
            preferred_weather = st.multiselect(
                "What type of weather do you prefer?",
                ["Warm", "Cold", "Moderate", "Sunny", "Rainy"],
                default=st.session_state.preferences.get('preferred_weather', []),
                help="Select your preferred weather conditions"
            )

        additional_notes = st.text_area(
            "Any additional preferences or requirements?",
            value=st.session_state.preferences.get('additional_notes', ''),
            help="Enter any other factors that are important to you (e.g., campus size, research opportunities, etc.)"
        )

        submit_button = st.form_submit_button("Save Preferences & Get Recommendations")
        
        if submit_button and field_of_study and preferred_regions and preferred_weather:
            st.session_state.preferences = {
                "field_of_study": field_of_study,
                "tuition_range": tuition_range,
                "preferred_regions": preferred_regions,
                "preferred_weather": preferred_weather,
                "additional_notes": additional_notes
            }
            
            # Generate initial recommendations
            with st.spinner('Generating your personalized university recommendations...'):
                # Initialize components for recommendations
                living_expenses_df, employment_projections_df, university_text = load_data()
                vector_store = initialize_chromadb(
                    living_expenses_df,
                    employment_projections_df,
                    university_text
                )
                
                retrieval_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vector_store.as_retriever(),
                    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
                    verbose=True
                )
                
                # Get recommendations
                recommendations = get_initial_recommendations(st.session_state.preferences, retrieval_chain)
                st.session_state.initial_recommendations = recommendations
            
            st.session_state.show_chat = True
            save_current_session()
            return True
            
        elif submit_button:
            st.error("Please fill in all required fields (Field of Study, Regions, and Weather preferences)")
    return False

def display_chat_interface(agent, retrieval_chain):
    """Display the chat interface"""
    st.write("### Your University Assistant")
    
    # Display initial recommendations
    if st.session_state.initial_recommendations:
        with st.expander("📚 Your Personalized University Recommendations", expanded=True):
            st.write(st.session_state.initial_recommendations)
            st.write("---")
            st.info("💡 Feel free to ask specific questions about these universities or explore other options!")
    
    # User profile and history
    with st.expander("👤 Your Profile and Past Conversations", expanded=False):
        st.write("#### Current Preferences")
        st.write(f"**Username:** {st.session_state.user_id}")
        st.write(f"**Field of Study:** {st.session_state.preferences['field_of_study']}")
        st.write(f"**Yearly Tuition Budget:** ${st.session_state.preferences['tuition_range'][0]:,} - ${st.session_state.preferences['tuition_range'][1]:,}")
        st.write(f"**Preferred Regions:** {', '.join(st.session_state.preferences['preferred_regions'])}")
        st.write(f"**Preferred Weather:** {', '.join(st.session_state.preferences['preferred_weather'])}")
        
        if st.session_state.preferences.get('additional_notes'):
            st.write(f"**Additional Notes:** {st.session_state.preferences['additional_notes']}")
        
        st.write("#### Recent Questions")
        if st.session_state.chat_history:
            for i, (role, message) in enumerate(st.session_state.chat_history[-5:]):
                if role == "You":
                    st.write(f"🗣️ **You asked:** {message[:100]}...")
    
    # Chat interface
    st.write("### Ask Your Questions")
    user_input = st.text_input(
        "Your question:",
        key="user_input",
        help="Ask about specific universities, programs, costs, campus life, etc.",
        placeholder="e.g., What are the admission requirements for the recommended universities?"
    )
    
    if user_input:
        # Create enhanced context including initial recommendations
        all_states = [state for region in st.session_state.preferences['preferred_regions'] 
                     for state in US_REGIONS[region]]
        
        context = f"""
        STUDENT CONTEXT:
        Field of Study: {st.session_state.preferences['field_of_study']}
        Budget: ${st.session_state.preferences['tuition_range'][0]:,} - ${st.session_state.preferences['tuition_range'][1]:,}
        Preferred Regions: {', '.join(st.session_state.preferences['preferred_regions'])}
        States Available: {', '.join(all_states)}
        Weather Preference: {', '.join(st.session_state.preferences['preferred_weather'])}
        
        PREVIOUSLY RECOMMENDED UNIVERSITIES:
        {st.session_state.initial_recommendations}
        
        CURRENT QUESTION: {user_input}
        
        Previous conversation context:
        {' '.join([f"{role}: {msg}" for role, msg in st.session_state.chat_history[-3:]])}
        
        Please provide:
        1. Specific answer to the question
        2. Relevant details about any universities mentioned
        3. Additional recommendations if applicable
        4. Practical next steps
        """
        
        try:
            with st.spinner('Researching your question...'):
                agent_response = agent.run(context)
                
                # Format and enhance the response
                formatted_response = agent_response
                
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("Assistant", formatted_response))
                save_current_session()
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Display chat history with improved formatting
    st.write("### Conversation History")
    for role, message in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                👤 <b>You:</b> {message}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background-color: #e8f4f9; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                🤖 <b>Assistant:</b> {message}
            </div>
            """, unsafe_allow_html=True)
    
    # Control buttons
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            save_current_session()
            st.experimental_rerun()
    with col2:
        if st.button("Update Preferences"):
            st.session_state.show_chat = False
            st.experimental_rerun()
    with col3:
        if st.button("Logout"):
            save_current_session()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()

def main():
    """Main application function"""
    try:
        # Initialize user database
        initialize_user_db()
        
        # Handle authentication
        if not st.session_state.authenticated:
            if login_page():
                st.experimental_rerun()
            return
        
        # Load data and initialize components
        with st.spinner('Loading data and initializing components...'):
            # Load data
            living_expenses_df, employment_projections_df, university_text = load_data()
            
            # Initialize ChromaDB
            vector_store = initialize_chromadb(
                living_expenses_df, 
                employment_projections_df, 
                university_text
            )
        
        # Initialize memory and chains
        if 'memory' not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        
        # Create tools and initialize agent with custom prompt
        tools = create_tools()
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=st.session_state.memory,
            verbose=True
        )
        
        # Create retrieval chain with custom prompt
        retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
            memory=st.session_state.memory,
            verbose=True,
            return_source_documents=True
        )
        
        # Display appropriate interface based on state
        if not st.session_state.show_chat:
            if 'initial_recommendations' in st.session_state and st.session_state.initial_recommendations:
                # Show recommendations if they exist
                st.write("### Your Current Recommendations")
                st.write(st.session_state.initial_recommendations)
                if st.button("Update Preferences"):
                    if display_preferences_form():
                        save_current_session()
                        st.experimental_rerun()
            else:
                # Show preferences form for new users or updates
                if display_preferences_form():
                    save_current_session()
                    st.experimental_rerun()
        else:
            # Show chat interface
            display_chat_interface(agent, retrieval_chain)
            save_current_session()
        
        # Add footer with helpful information
        st.markdown("---")
        with st.expander("ℹ️ Tips for using the Assistant"):
            st.write("""
            - Ask specific questions about recommended universities
            - Inquire about admission requirements, deadlines, and application processes
            - Ask about scholarships and financial aid opportunities
            - Get information about campus life and student services
            - Learn about housing options and living costs
            - Explore career opportunities and job placement rates
            """)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if 'user_id' in st.session_state:
            save_current_session()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.write("Please refresh the page and try again.")