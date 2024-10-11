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
pdf_loader = PyPDFLoader(r"D:\Convoloo_Intern\meeting02\02_Report_Gemini_LLAMA_OpenAI.pdf")
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
db_uri = "sqlite:///D:\Convoloo_Intern\Project\patientHealthData.db"
db=SQLDatabase.from_uri(db_uri)

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
