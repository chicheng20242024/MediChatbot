
# %%
from datetime import datetime, timedelta
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import pickle

from typing import Optional, Type
from langchain.tools import BaseTool
from langchain.agents import initialize_agent
from pydantic import BaseModel, Field
from langchain.agents import AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
SCOPES = ['https://www.googleapis.com/auth/calendar']

creds = None
if os.path.exists('token.pickle'):
    with open("token.pickle","rb") as token:
        creds = pickle.load(token)

if not creds or creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
r"D:\Convoloo_Intern\Project\credentials.json", SCOPES)
        creds = flow.run_local_server(prot=0)
    with open("token.picke","wb") as token:
        pickle.dump(creds,token)

service = build ("calendar","v3",credentials=creds)

now = datetime.utcnow().isoformat() + 'Z'

# %%
def create_event(doc_email, client_mail, client_name, start_time):
    start_time = start_time.replace('Z', "")
    end_time = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S') + timedelta(hours=1)

    event = {
        'summary': f'Appointment with {client_name}',
        'location': 'Virtual',
        'description': f'appointment with {client_name}',
        'start': {
            'dateTime': start_time,
            'timeZone': 'Europe/London'
        },
        'end': {
            'dateTime': end_time.strftime('%Y-%m-%dT%H:%M:%S'),
            'timeZone': 'Europe/London',
        },
        'attendees': [
            {'email': doc_email},
            {'email': client_mail},
        ],
        'reminders': { 
            'useDefault': False,
            'overrides': [
                {'method': 'email', 'minutes': 24 * 60},
                {'method': 'popup', 'minutes': 10},
            ],
        },
        'conferenceData': {
            'createRequest': {
                'requestId': f"{now}",
                'conferenceSolutionKey': {
                    'type': 'hangoutsMeet'
                },
            },
        },
    }

    event = service.events().insert(calendarId='primary', body=event, conferenceDataVersion=1).execute()
    return f"Event created: {event.get('htmlLink')}"


# %%

class BookingInput(BaseModel):
    """Input for the booking tool"""
    doc_email: str = Field (..., description="Email of the doctor")
    client_email: str = Field (..., description="Email of the client")
    client_name: str = Field (..., description="Name of the client")
    start_time: str = Field (..., description="Start time and date of the booking")
    
import json

class BookingTool(BaseTool):
    name = "create_event"
    description = "this tool is used to book an appointment with the doctor"
    
    def _run(self, doc_email, client_email, client_name, start_time):
        event_link = create_event(doc_email, client_email, client_name, start_time)
        return event_link
    
    args_schema: Optional[Type[BaseModel]] = BookingInput

tools = [BookingTool()]

# %%
from langchain.agents import initialize_agent

llm = ChatGoogleGenerativeAI(model = "gemini-pro",
                             google_api_key="AIzaSyAJ9gS_ch1anqk2G20hk0_Gxu_wT6nK-1w",
                             temperature="0",
                             convert_system_message_to_human=True)

llm
# %%
booking_agent = initialize_agent(tools,llm,
                                 agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                                 verbose = True,
                                 agent_kwargs={
                                     "SystemMessage": "Extract information in string, make reservations, and return the link for the reservation"
                                 })

if __name__ == "__main__":
    booking_agent.invoke("Can you please book an appointment for 30th August, 2024 at 11 am. My email is brucecheng921@gmail.com, and the doctor's email is xxx@gmail.com")
# %%
