# # %%
# import os
# import getpass
# import pathlib
# import textwrap
# import pickle
# import google.generativeai as genai
# import warnings
# from IPython.display import Markdown
# import urllib

# # Function to convert text to markdown
# def to_markdown(result):
#     text = result.replace('•', '  *')
#     return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# # Initialize the language model
# from langchain_google_genai import ChatGoogleGenerativeAI

# llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key="AIzaSyAJ9gS_ch1anqk2G20hk0_Gxu_wT6nK-1w")

# # Test the LLM invocation
# result = llm.invoke("What are the usecases of LLMs?")
# print(to_markdown(result.content))

# # Configure the larger model
# llm = ChatGoogleGenerativeAI(
#     model="gemini-pro", 
#     google_api_key="AIzaSyAJ9gS_ch1anqk2G20hk0_Gxu_wT6nK-1w",
#     temperature=0.7, 
#     convert_system_message_to_human=True
# )

# persist_directory = 'ncc'
# print(result.content)

# # Load PDF and display content
# from langchain_community.document_loaders import PyPDFLoader

# pdf_loader = PyPDFLoader(r"D:\Convoloo_Intern\meeting02\02_Report_Gemini_LLAMA_OpenAI.pdf")
# file = pdf_loader.load()

# def load_db(file):
#     for page in file:
#         print(page.page_content[:500])  # Print first 500 characters of each page

# load_db(file)

# # Creating a template for the RAG
# from langchain.prompts import ChatPromptTemplate

# template = """You are an assistant for carrying out question-answering tasks. Answer according to the context provided. If you don't know the answer, just say that you don't know.
# Question: {question}
# Context: {context}
# Answer with specific details about Gemini and LLAMA models:
# """
# prompt = ChatPromptTemplate.from_template(template)

# # Split text
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200, add_start_index=True)

# context = "\n\n".join(str(p.page_content) for p in file)
# all_splits = text_splitter.split_text(context)
# print(len(all_splits))

# # Convert the text splits into Document objects
# from langchain.schema import Document  # Assuming Document is imported from langchain.schema

# documents = [Document(page_content=chunk) for chunk in all_splits]

# # Ensure that the documents contain the relevant information
# # for i, doc in enumerate(documents[:5]):  # Check first 5 chunks for relevant content
# #     print(f"Document {i} content preview: {doc.page_content[:500]}")

# # Initialize embeddings and vector store
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyAJ9gS_ch1anqk2G20hk0_Gxu_wT6nK-1w")

# from langchain_community.vectorstores import FAISS

# # Assuming you use FAISS for the vector store
# vectorstore = FAISS.from_documents(documents, embedding=embeddings)

# # Create the retriever and the retrieval chain
# from langchain.chains import RetrievalQA

# retriever = vectorstore.as_retriever()

# rag_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     chain_type="map_reduce",  # Adjust chain type as needed
#     input_key="question",
#     return_source_documents=True
# )

# # Test the RAG chain
# question = "What are the key points about Gemini and LLAMA models?"
# result = rag_chain({"question": question})
# print(result)

# # from langchain_google_genai import GoogleGenerativeAI

# # client = GoogleGenerativeAI(google_api_key="AIzaSyAJ9gS_ch1anqk2G20hk0_Gxu_wT6nK-1w")

# # # List available models
# # available_models = client.list_models()
# # print(available_models)


# # %%
# from langchain.chains import create_sql_query_chain
# from langchain_google_genai import GoogleGenerativeAI
# llm_sql = GoogleGenerativeAI(model="models/text-bison-001", google_api_key="AIzaSyAJ9gS_ch1anqk2G20hk0_Gxu_wT6nK-1w", temperature=0.7)
# from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
# from operator import itemgetter
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough

# answer_prompt = PromptTemplate.from_template(
#     """Given the following user question, corresponding SQL query, and SQL result, 
#     answer the user question in a contructive sentence that mentions the question asked.
#     Question: {question}
#     SQL Query: {query}
#     SQL  Result: {result}
#     Answer:"""
# )

# # %%
# from langchain_community.utilities import SQLDatabase
# import sqlite3
# db_uri = "sqlite:///D:\Convoloo_Intern\Project\patientHealthData.db"
# db = SQLDatabase.from_uri(db_uri)

# from langchain.chains import create_sql_query_chain
# generate_query = create_sql_query_chain(llm_sql, db)
# query = generate_query({"question": "How many patient are there?"})
# # %%
#%%%
import os
import pathlib
import urllib
import google.generativeai as genai
import textwrap
import pickle
from IPython.display import Markdown
import getpass
#from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv())
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_openai import ChatOpenAI
#%%
llm = ChatGoogleGenerativeAI(model = "gemini-pro", google_api_key= "AIzaSyAJ9gS_ch1anqk2G20hk0_Gxu_wT6nK-1w",
                             temperature=0.7, convert_system_message_to_human=True)
#llm_2 = ChatOpenAI(model="gpt-3.5-turbo-0125")
from langchain_community.document_loaders import PyPDFLoader
#%%
pdf_loader = PyPDFLoader(r"D:\Convoloo_Intern\Project\Report_AI.pdf")
#Indexing splits
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1200, chunk_overlap = 200,  add_start_index=True)
file = pdf_loader.load_and_split(text_splitter)
#%%
from langchain.prompts import ChatPromptTemplate
templates =  """You are an assistant for carrying out question-answering tasks. Answer according to the context provided. If you don't know the answer, just say that you don't know.
Question: {question}
Context: {context}
Answer:
"""
#extract the text
#context = "\n\n".join(str(p.page_content)  for p in file)
#all_splits = text_splitter.split_documents(context)
#%%
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key = "AIzaSyAJ9gS_ch1anqk2G20hk0_Gxu_wT6nK-1w",
                                          transport="rest")
from langchain.chains import ConversationalRetrievalChain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
vectorstore= FAISS.from_documents(file, embeddings)
retriever = vectorstore.as_retriever(k = 1)
#%%
prompt = ChatPromptTemplate.from_template(templates)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
rag_chain = (
    {"context": retriever |format_docs, "question": RunnablePassthrough()}
     | prompt
     | llm
     | StrOutputParser())
from langchain.chains import create_sql_query_chain
from langchain_google_genai import GoogleGenerativeAI
#from langchain_openai import ChatOpenAI
#%%
llm_sql = GoogleGenerativeAI(model = "models/text-bison-001", google_api_key= "AIzaSyAJ9gS_ch1anqk2G20hk0_Gxu_wT6nK-1w", temperature=0.7)
#llm_sql2 = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key = os.environ['OPENAI_API_KEY'], temperature=0.7)
# execute query
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result,
    answer the user question in a contructive sentence that mentions the question asked.
    Question: {question}
    SQL Query: {query}
    SQL  Result: {result}
    Answer:"""
)
# print(final_prompt.format(input="",table_info="some table info"))
# llm_sql is Gemini llm_sql2 is openai
# %%
from langchain_community.utilities import SQLDatabase
import sqlite3
# db_uri = "mysql+mysqlconnector://root:123456@localhost:3306/chinook"
# db=SQLDatabase.from_uri(db_uri)
# print(db.run("SELECT * FROM artist limit 10;"))
db_uri = "sqlite:///D:\Convoloo_Intern\Project\patientHealthData.db"
db=SQLDatabase.from_uri(db_uri)
# %%
# columns_query = """
# SELECT table_name, column_name
# FROM information_schema.columns
# WHERE table_schema = 'chinook' AND column_name = 'patient';
# """
# result = db.run(columns_query)
# if result:
#     print(result)
# else:
#     print("No columns named 'patient' found.")
# %%
# Query to get column names from a specific table, e.g., 'artist'
# columns_query = """
# SELECT COLUMN_NAME
# FROM information_schema.columns
# WHERE table_schema = 'chinook' AND table_name = 'artist';
# """
# columns = db.run(columns_query)
# print(columns)


# # %%
# search_query = """
# SELECT *
# FROM artist
# WHERE ArtistId LIKE '%patient%' OR Name LIKE '%patient%'
# LIMIT 10;
# """

# # Execute the query
# results = db.run(search_query)
# print(results)


# %%
from langchain.chains import create_sql_query_chain
generate_query = create_sql_query_chain(llm_sql, db)
query = generate_query.invoke({"question": "How many patients are there?"})
query

# %%
from langchain_core.prompts import PromptTemplate
template = '''Given an input question, first create a syntactically correct sqlite query to run, then look at the results of the query and return the answer.
Use the following format:
Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"
Only use the following tables:
{table_info}
\nCREATE TABLE "patientHealthData" (\n\t"Patient_ID" BIGINT, \n\t"Date" TEXT, \n\t"Time_of_Day" TEXT, \n\t"Glucose_Level" FLOAT, \n\t"Systolic_Pressure" BIGINT, \n\t"Diastolic_Pressure" BIGINT, \n\t"Dietary_Info" TEXT, \n\t"Water_Intake (Liters)" FLOAT, \n\t"Steps_Taken" BIGINT\n)\n\n/*\n3 rows from patientHealthData table:\nPatient_ID\tDate\tTime_of_Day\tGlucose_Level\tSystolic_Pressure\tDiastolic_Pressure\tDietary_Info\tWater_Intake (Liters)\tSteps_Taken\n1\t2023-12-16\tMorning\t193.34\t127\t73\t[(\'Banana\', 175, 6.140936728763672, 3.2275577289707345, 1.1544553954637582), (\'Chicken Breast\', 95, \t1.47\t7446\n1\t2023-12-16\tBreakfast\t152.22\t90\t87\t[(\'Banana\', 189, 1.7434071790485473, 1.3737664750151823, 8.684217427301824), (\'Chicken Breast\', 100,\t1.73\t4186\n1\t2023-12-16\tLunch\t127.47\t110\t84\t[(\'Banana\', 110, 4.763203826908784, 8.637806704142006, 8.732422419551787), (\'Chicken Breast\', 193, 4\t1.77\t2770\n*/.
Return {top_k} 5 result per select statement.
Question: {input}'''

#%%
prompt_info = PromptTemplate.from_template(template)
rephrase_answer = answer_prompt | llm_sql | StrOutputParser()
generate_query = create_sql_query_chain(llm_sql, db, prompt = prompt_info)
execute_query = QuerySQLDataBaseTool(db=db)
chain = (
    RunnablePassthrough.assign(query = generate_query).assign(
       result = itemgetter("query") | execute_query
    )
    | rephrase_answer
)
print(chain.invoke({"question": "How mnay patient are there?"}))

# %%
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
# from langchain_community.utilities import SQLDatabase
# import sqlite3
# import pandas as pd
# from sqlalchemy import Table, MetaData
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import DataFrameLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.prompts import ChatPromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain.chains import create_sql_query_chain
# from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
# from operator import itemgetter

# google_api_key= "AIzaSyAJ9gS_ch1anqk2G20hk0_Gxu_wT6nK-1w"

# %%
# sqlite_db_path = 'D:\Convoloo_Intern\Project\patientHealthData.db'

# conn = sqlite3.connect(sqlite_db_path)

# tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)


# for table in tables['name']:
#     df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
#     df.to_csv(f'{table}.csv', index=False)

# conn.close()

# # %%
# DB_HOST = "34.42.45.239"#"34.172.207.78"
# DB_PORT = "5432"
# DB_USER = "postgres"
# DB_PASSWORD = "jiaowoBruce921#"
# DB_NAME = "test"

# DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# db = SQLDatabase(engine)

# # %%
# # Initialize the LLM
# llm = ChatGoogleGenerativeAI(
#     model="gemini-pro",
#     google_api_key="AIzaSyAJ9gS_ch1anqk2G20hk0_Gxu_wT6nK-1w",
#     temperature=0.7,
#     convert_system_message_to_human=True
# )

# # Load the PDF
# pdf_loader = PyPDFLoader(r"D:\Convoloo_Intern\meeting02\02_Report_Gemini_LLAMA_OpenAI.pdf")

# # Split the text
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200, add_start_index=True)
# file = pdf_loader.load_and_split(text_splitter)

# # Define the prompt template
# templates = """You are an assistant for carrying out question-answering tasks. Answer according to the context provided. If you don't know the answer, just say that you don't know.
# Question: {question}
# Context: {context}
# Answer:
# """

# # Initialize embeddings
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/embedding-001",
#     google_api_key="AIzaSyAJ9gS_ch1anqk2G20hk0_Gxu_wT6nK-1w",
#     transport="rest"
# )

# # Create vectorstore and retriever
# vectorstore = FAISS.from_documents(file, embeddings)
# retriever = vectorstore.as_retriever(k=1)

# # Define the format_docs function
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# # Define the RAG chain
# prompt = ChatPromptTemplate.from_template(templates)
# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# # %%
# llm_sql = GoogleGenerativeAI(
#     model="models/text-bison-001",
#     google_api_key="AIzaSyAJ9gS_ch1anqk2G20hk0_Gxu_wT6nK-1w",
#     temperature=0.7
# )

# # Update db_uri to connect to Google Cloud SQL (MySQL example)
# db_uri = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# db = SQLDatabase.from_uri(db_uri)

# # Create SQL query chain
# generate_query = create_sql_query_chain(llm_sql, db)
# query = generate_query.invoke({"question": "How many patients are there?"})

# answer_prompt = PromptTemplate.from_template(
#     """Given the following user question, corresponding SQL query, and SQL result,
#     answer the user question in a constructive sentence that mentions the question asked.
#     Question: {question}
#     SQL Query: {query}
#     SQL Result: {result}
#     Answer:"""
# )

# template = '''Given an input question, first create a syntactically correct SQL query to run, then look at the results of the query and return the answer.
# Use the following format:
# Question: "Question here"
# SQLQuery: "SQL Query to run"
# SQLResult: "Result of the SQLQuery"
# Answer: "Final answer here"
# Only use the following tables:
# {table_info}
# \nCREATE TABLE "patientHealthData" (\n\t"Patient_ID" BIGINT, \n\t"Date" TEXT, \n\t"Time_of_Day" TEXT, \n\t"Glucose_Level" FLOAT, \n\t"Systolic_Pressure" BIGINT, \n\t"Diastolic_Pressure" BIGINT, \n\t"Dietary_Info" TEXT, \n\t"Water_Intake (Liters)" FLOAT, \n\t"Steps_Taken" BIGINT\n)\n\n/*\n3 rows from patientHealthData table:\nPatient_ID\tDate\tTime_of_Day\tGlucose_Level\tSystolic_Pressure\tDiastolic_Pressure\tDietary_Info\tWater_Intake (Liters)\tSteps_Taken\n1\t2023-12-16\tMorning\t193.34\t127\t73\t[(\'Banana\', 175, 6.140936728763672, 3.2275577289707345, 1.1544553954637582), (\'Chicken Breast\', 95, \t1.47\t7446\n1\t2023-12-16\tBreakfast\t152.22\t90\t87\t[(\'Banana\', 189, 1.7434071790485473, 1.3737664750151823, 8.684217427301824), (\'Chicken Breast\', 100,\t1.73\t4186\n1\t2023-12-16\tLunch\t127.47\t110\t84\t[(\'Banana\', 110, 4.763203826908784, 8.637806704142006, 8.732422419551787), (\'Chicken Breast\', 193, 4\t1.77\t2770\n*/.
# Return {top_k} 5 result per select statement.
# Question: {input}'''

# prompt_info = PromptTemplate.from_template(template)
# rephrase_answer = answer_prompt | llm_sql | StrOutputParser()
# execute_query = QuerySQLDataBaseTool(db=db)

# chain = (
#     RunnablePassthrough.assign(query=generate_query).assign(
#         result=itemgetter("query") | execute_query
#     )
#     | rephrase_answer
# )

# print(chain.invoke({"question": "How many patients are there?"}))