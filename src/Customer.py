import pandas as pd
import uuid
from datetime import datetime

from src.utils import sentimentAnalysis

class Customer:
    def __init__(self, id:int) -> None:
        self.id = id
        
    def get_latest_convo(self, no_of_convos:int):
        df_chat = pd.read_csv("Data/chat_convo.csv", sep='#')
        df_email = pd.read_csv("Data/email_convo.csv", sep='#')
        df_chat["date"] = pd.to_datetime(df_chat["date"])
        df_email["date"] = pd.to_datetime(df_email["date"])
        df = pd.concat([df_chat, df_email])
        df = df[df["customer_id"] == self.id].sort_values(by="date", ascending=False).head(no_of_convos)
        return df
    
    def add_chat_convo(self, message:str):
        date = datetime.now()
        sentiment = sentimentAnalysis(message).replace("\n", " ")
        record = f"{uuid.uuid4()}#{self.id}#{message}#{date}#{sentiment}"
        record = record.replace("\n", " ")
        record = f"\n{record}"
        
        with open("Data/chat_convo.csv", "a") as f:
            f.write(record)
    
    def add_email_convo(self, message:str):
        date = datetime.now()
        sentiment = sentimentAnalysis(message).replace("\n", " ")
        record = f"\n{uuid.uuid4()}#{self.id}#{message}#{date}#{sentiment}"
        record = record.replace("\n", " ")
        record = f"\n{record}"
        with open("Data/email_convo.csv", "a") as f:
            f.write(record)
            
    def add_customer_feedback(self,feedback:str):
        date = datetime.now()
        sentiment = sentimentAnalysis(feedback).replace("\n", " ")
        record = f"\n{uuid.uuid4()}#{self.id}#{feedback}#{date}#{sentiment}"
        record = record.replace("\n", " ")
        record = f"\n{record}"
        with open("Data/feedback.csv", "a") as f:
            f.write(record)
            
    def get_latest_service_feedback_Analysys(self):
        df = pd.read_csv("Data/feedback.csv", sep='#')
        df = df[df["customer_id"] == self.id]
        if(len(df) == 0):
            return None
        return df.iloc[0]["Sentiment"]
        

        
        