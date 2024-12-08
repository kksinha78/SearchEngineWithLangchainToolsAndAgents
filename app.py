import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain.utilities import ArxivAPIWrapper,WikipediaAPIWrapper # this wrapper will use wikipedia API to cinduct searches and fetch page summaries.By defualt it will return page summaries of the top -k results.it limit the document content by doc_content_chars_max
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun #DuckDuckGoSearchRun helps to ssearch anything from internet
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent

from langchain.callbacks import StreamlitCallbackHandler # this will allow to communicate with al tools within themselves
from dotenv import load_dotenv

#Load the GROQ api key
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

#Arxiv and wikipedia tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results =1, doc_content_chars_max=500)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

wikipedia_wrapper=WikipediaAPIWrapper(top_k_results =1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

search = DuckDuckGoSearchRun(name="search") # tool for search from internet

# app title
st.title("   Langchain Chat with search")

"""
In this example, we are using 'StreamlitCallbackHandler' to display the thoughts and actions of an agent in an interactive Streamlit app.
"""
# Lets create session state to have chat history
if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assisstant","content":"Hi, I am a chatbot who can search the web.How can I help you?"}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

#Create prompt
if prompt:=st.chat_input(placeholder="What is Machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192",streaming = True)

    # tools = [arxiv,wiki,search]
    tools = [arxiv,wiki] # all above 3 tools are in list

    #lets convert all these tools into agant so that we will be able to invoke these agents

    search_agent = initialize_agent(tools,llm,agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)

    with st.chat_message("assisstant"):
        st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts = False)
        response = search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assisstant',"content":response})
        st.write(response)


#Sedebar settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API key:",type="password")

