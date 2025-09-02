# from http.client import responses
#
# import pandas as pd
# import streamlit as st
# from langchain.agents import AgentType
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from langchain_ollama import ChatOllama
# from sqlalchemy import True_
#
# #streamlit web configuration
#
# st.set_page_config(
#     page_title = "DF Chat",
#     page_icon = "💬",
#     layout = "centered"
# )
#
# def read_data(file):
#     if file.name.endswith(".csv"):
#         return pd.read_csv(file)
#     else:
#         return pd.read_excel(file)
#
# # streamlit page title
#
# st.title("🤖 Dataframe Chatbot - Ollama")
#
# #initialize chat history in streamlit session state
#
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
#
# #initiate df in session state
#
# if "df" not in st.session_state:
#     st.session_state.df = None
#
# #file upload widget
#
# uploaded_file = st.file_uploader("Choose a file", type=["csv","xlsx","xls"])
#
# if uploaded_file:
#     st.session_state.df = read_data(uploaded_file)
#     st.write("Dataframe Preview:")
#     st.dataframe(st.session_state.df.head())
#
# #display chat history
#
# for message in st.session_state.chat_history:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#
# # input field for user's message
# user_prompt = st.chat_input("Ask LLM...")
#
# if user_prompt:
#     st.chat_message("user").markdown(user_prompt)
#     st.session_state.chat_history.append({"role":"user" , "content":user_prompt})
#
#     #loading the llm
#     llm = ChatOllama(model="gemma:2b", temperature=0)
#
#     pandas_df_agent = create_pandas_dataframe_agent(
#         llm,
#         st.session_state.df,
#         verabose=True,
#         agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#         allow_dangerous_code=True
#     )
#
#     message = [
#         {"role":"system", "content": "ou are a helpful agent"},
#         *st.session_state.chat_history
#     ]
#
#     response = pandas_df_agent.invoke(user_prompt)
#
#     assistant_response = response["output"]
#
#     st.session_state.chat_history.append({"role":"assistant", "content": assistant_response})
#
#     #display LLM response
#     with st.chat_message("assistant"):
#         st.markdown(assistant_response)


import pandas as pd
import streamlit as st
from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_ollama import ChatOllama

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DF Chat Advanced",
    page_icon="📊",
    layout="centered"
)


# --- 2. CACHED FUNCTIONS for Performance ---

@st.cache_resource
def get_llm():
    """Initializes and caches the LLM."""
    return ChatOllama(model="llama3", temperature=0)


@st.cache_resource
def get_agent(_df):
    """Creates and caches the dataframe agent."""
    llm = get_llm()

    def safe_parsing_error_callback(error):
        return f"⚠️ Parsing error: {error}. Returning best guess instead."


    return create_pandas_dataframe_agent(
        llm,
        _df,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True,
        handle_parsing_errors=True,
        return_intermediate_steps=False
    )


@st.cache_data(show_spinner="Reading file...")
def read_data(file):
    """Reads and caches a CSV or Excel file."""
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


def format_chat_to_markdown(history):
    """Formats the chat history into a Markdown string for download."""
    md = "# Chat History\n\n"
    for message in history:
        md += f"**{message['role'].capitalize()}**: {message['content']}\n\n---\n\n"
    return md


# --- 3. SESSION STATE INITIALIZATION ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "df" not in st.session_state:
    st.session_state.df = None

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("📁 Settings")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls"],
        help="Upload a file to start chatting with your data."
    )
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.header("📤 Export")
    chat_md = format_chat_to_markdown(st.session_state.chat_history)
    st.download_button(
        label="Download Chat History",
        data=chat_md,
        file_name="chat_history.md",
        mime="text/markdown",
        use_container_width=True,
        disabled=not st.session_state.chat_history
    )

# --- 5. MAIN APP ---
st.title("🤖 DataFrame Chatbot - Ollama")
st.markdown(
    "Welcome! I'm an AI assistant that can help you analyze your data.")

if uploaded_file:
    st.session_state.df = read_data(uploaded_file)
    if st.session_state.df is not None:
        if "new_file" not in st.session_state or st.session_state.new_file != uploaded_file.name:
            st.session_state.new_file = uploaded_file.name
            st.session_state.chat_history = []
            with st.chat_message("assistant"):
                st.markdown("File uploaded! Here's a preview. What would you like to know?")
        st.dataframe(st.session_state.df.head())

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display the thought process if it exists
        if "steps" in message and message["steps"]:
            with st.expander("Show thought process 🧠"):
                for step in message["steps"]:
                    action, observation = step
                    st.markdown("**Action:**")
                    st.code(action.tool_input, language="python")
                    st.markdown("**Observation:**")
                    st.info(observation)


def run_agent(prompt):
    """Handles the agent execution and chat history."""
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤔"):
            try:
                pandas_df_agent = get_agent(st.session_state.df)
                # Pass input as a dictionary for better compatibility
                response = pandas_df_agent.invoke({"input": prompt})
                assistant_response = response.get("output", "Sorry, I couldn't process that.")
                intermediate_steps = response.get("intermediate_steps", [])

                st.markdown(assistant_response)
                # Display thought process in the expander
                if intermediate_steps:
                    with st.expander("Show thought process 🧠"):
                        for step in intermediate_steps:
                            action, observation = step
                            st.markdown("**Action:**")
                            st.code(action.tool_input, language="python")
                            st.markdown("**Observation:**")
                            st.info(str(observation))

                # Store the response and steps in session state
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": assistant_response,
                    "steps": intermediate_steps
                })

            except Exception as e:
                error_message = f"An error occurred: {e}. Please try rephrasing your question."
                st.error(error_message)
                st.session_state.chat_history.append({"role": "assistant", "content": error_message, "steps": []})


# Handle user input
if st.session_state.df is not None:
    # Show dynamic example prompts only if chat is empty
    if not st.session_state.chat_history:
        st.markdown("**Example questions:**")
        numeric_cols = st.session_state.df.select_dtypes(include='number').columns.tolist()
        examples = ["How many rows are in the dataset?", "What are the columns?"]
        if numeric_cols:
            examples.append(f"What is the average of the '{numeric_cols[0]}' column?")

        cols = st.columns(len(examples))
        for i, example in enumerate(examples):
            if cols[i].button(example, use_container_width=True, key=f"example_{i}"):
                run_agent(example)

    if user_prompt := st.chat_input("Ask a question about your data..."):
        run_agent(user_prompt)

else:
    st.info("👋 Please upload a file in the sidebar to get started!")

