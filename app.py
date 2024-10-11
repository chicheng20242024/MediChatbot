# %%
# !pip install chainlit

# %%
import logging
import json
import os
import getpass
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes
from RagChain import chain, rag_chain
from appointment import booking_agent
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import PyPDFLoader
import nest_asyncio
from langserve import add_routes
import chainlit as cl
import requests

app = FastAPI(
    title='Medical Assistant Chatbot',
    version='1.0',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

class RouteQuery(BaseModel):
    """Route a user query to the most relevant action"""
    action: Literal["general_query", "sql_query", "appointment_booking", "web_search"] = Field(
        ...,
        description="Determines the appropriate action based on the user's question",
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key="AIzaSyAJ9gS_ch1anqk2G20hk0_Gxu_wT6nK-1w",
    temperature=0,
    convert_system_message_to_human=True
)

system_prompt = """
You are an AI assistant responsible for directing user queries to the appropriate service. 
The user may ask general questions about medical information, request patient data from a database, 
or book a doctor’s appointment using the Google Calendar API. 

For each query, you must respond in two parts:
1. The action type ('general_query', 'sql_query', 'appointment_booking', or 'web_search') followed by a colon.
2. The user's original question or relevant information following the action.

Example: 'general_query: What are the symptoms of diabetes?'

Please structure your responses accordingly.
"""
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)
router = prompt_template | llm

class UserQuery(BaseModel):
    question: str

def google_custom_search(query):
    api_key = "AIzaSyB36kIDp5DrxzedYx38BGJrLJswLJpB4hg" 
    search_engine_id = "943bbbfb7e98b480f" 
    
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": search_engine_id,
        "q": query
    }
    
    response = requests.get(search_url, params=params)
    response.raise_for_status()
    
    search_results = response.json()
    if "items" in search_results:
        top_result = search_results["items"][0]
        return f"Web Search Result: {top_result['snippet']} (Source: {top_result['link']})"
    else:
        return "No relevant information found in web search."

def choose_route(result):
    # Handle greetings and simple interactions first
    greetings = ["hi", "hello", "hey"]
    user_query = result.content.strip().lower()
    if any(greeting in user_query for greeting in greetings):
        return "How can I assist you today?"

    # Check if the model's response is as expected
    if not result or not result.content:
        return "Error: No response from the model."

    try:
        # Attempt to parse the model's response
        result_content = result.content.strip().split(":", 1)
        if len(result_content) < 2:
            return "The response from the model is not in the expected format."

        action = result_content[0].strip().lower()
        user_query = result_content[1].strip().rstrip('?')  # Remove any trailing question mark for consistency

        # Log the action and query for debugging
        print(f"Action: {action}")
        print(f"User Query: {user_query}")

        # Handle the action appropriately
        if "general_query" in action:
            output = rag_chain.invoke({"question": user_query})
            if not output:
                output = google_custom_search(user_query)
                output += "\n(Note: This information was retrieved from a web search. Please verify the source before using it.)"
        elif "sql_query" in action:
            output = chain.invoke({"question": user_query})
            if not output:
                output = google_custom_search(user_query)
                output += "\n(Note: This information was retrieved from a web search. Please verify the source before using it.)"
        elif "appointment_booking" in action:
            output = booking_agent.invoke({"question": user_query})
        elif "web_search" in action:
            output = google_custom_search(user_query)
            output += "\n(Note: This information was retrieved from a web search. Please verify the source before using it.)"
        else:
            output = "Could not determine the action."
        
        if isinstance(output, dict):
            output = json.dumps(output)  # Convert dict to JSON string if necessary

        return output
    except Exception as e:
        # Log the exception if any
        print(f"Error processing the request: {str(e)}")
        return "Sorry, there was a problem processing your request."
# %%
agent_chain = router | RunnableLambda(choose_route)

# %%
# Chainlit interaction handlers
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("runnable", agent_chain)

# %%
@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")
    try:
        response = runnable.invoke({"question": message.content})
        await cl.Message(content=f"Received: {response}").send()
    except Exception as e:
        logging.error(f"Error processing message: {e}")
        await cl.Message(content="Sorry, there was a problem processing your request.").send()

# FastAPI and Chainlit app launch would typically be done externally from this script, not within the same file.

# %%
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# %%
# def choose_route(result):
#     if not result or not result.content:
#         return "Error: No response from the model."

#     result_content = result.content.strip().lower().split(":")
    
#     if len(result_content) < 2:
#         logging.error(f"Unexpected format: {result_content}")
#         return "The response from the model is not in the expected format."

#     action = result_content[0].strip()
#     user_query = result_content[1].strip()

#     print(f"Action: {action}")
#     print(f"User Query: {user_query}")

#     if "general_query" in action:
#         if not isinstance(user_query, str):
#             logging.error(f"Unexpected query format: {user_query}")
#             return "The query is not in the expected format."
#         output = rag_chain.invoke({"question": user_query})
#         if not output:
#             # Perform web search if no information is found
#             output = google_custom_search(user_query)
#             output += "\n(Note: This information was retrieved from a web search. Please verify the source before using it.)"
#     elif "sql_query" in action:
#         if not isinstance(user_query, str):
#             logging.error(f"Unexpected query format: {user_query}")
#             return "The query is not in the expected format."
#         output = chain.invoke({"question": user_query})
#         if not output:
#             # Perform web search if no information is found in the database
#             output = google_custom_search(user_query)
#             output += "\n(Note: This information was retrieved from a web search. Please verify the source before using it.)"
#     elif "appointment_booking" in action:
#         if not isinstance(user_query, str):
#             logging.error(f"Unexpected query format: {user_query}")
#             return "The query is not in the expected format."
#         output = booking_agent.invoke({"question": user_query})
#     else:
#         logging.error(f"Unable to determine action from model output: {action}")
#         output = "Error: Could not determine the action."

#     if isinstance(output, dict):
#         logging.error(f"Unexpected output format: {output}")
#         return "The output from the model is not in the expected format."

#     return output