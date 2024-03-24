import pandas as pd
from datetime import datetime

from src.Customer import Customer
from src.utils import createPredictionPrompt, predictChurn

class RelationshipManager:
    def __init__(self, id:int) -> None:
        self.id = id
    
    def get_customers(self):
        df = pd.read_csv('Data/RM_Customer.csv')
        df = df[df['RM_ID'] == self.id]
        return df['customer_id'].to_list()
    
    def createData(self, customer_id):
        df = pd.read_csv('Data/train1.csv')
        df = df[df['Customer_id'] == customer_id]
        df['Last_active_date'] = pd.to_datetime(df['Last_active_date'],format='%d-%M-%Y')
        today = datetime.today()
        df['Months_since_last_activity'] = (today - df['Last_active_date']) // pd.Timedelta(days=30)
        df = df.drop(columns=['Last_active_date'])
        
        sentiments_df = Customer(customer_id).get_latest_convo(5)
        available_sentiments = len(sentiments_df)
        for i in range(5):
            col_name = f'Chat_Analysis_{i+1}'
            if(i<available_sentiments):
                df[col_name] = sentiments_df.iloc[i]['Sentiment'] if (sentiments_df.iloc[i]['Sentiment'] != None or pd.isna(sentiments_df.iloc[i]['Sentiment'])) else 'Positive' 
            else:
                df[col_name] = 'Positive'
        
        customer_feedback = Customer(customer_id).get_latest_service_feedback_Analysys()
        if(customer_feedback is None):
            df['Customer_service_feedback_Analysis'] = 'Positive'
        else:
            df['Customer_service_feedback_Analysis'] = customer_feedback
            
        return df
    
    def get_churn_report(self):
        with open("Data/churn_report.csv", "w") as f:
                f.write(f"customer_id#churn#probability#reason#preventive_steps\n")
        customers = self.get_customers()
        for customer_id in customers:
            data = self.createData(customer_id)
            prompt = createPredictionPrompt(data.iloc[0])
            print(prompt)
            pred = predictChurn(data=prompt)
            tmp = str(pred).split('Prediction: ')            
            tmp = tmp[1].split('Probability: ')
            churn = tmp[0].replace('\n', '')
            tmp = tmp[1].split('Reason: ')
            probability = tmp[0].replace('\n', '')
            tmp = tmp[1].split('Preventive Steps: ')
            reason = tmp[0].replace('\n', '')
            preventive_steps = tmp[1].replace('\n', '')
            with open("Data/churn_report.csv", "a") as f:
                f.write(f"{customer_id}#{churn}#{probability}#{reason}#{preventive_steps}\n")
            with open("Data/churn_history_report.csv", "a") as f:
                f.write(f"{customer_id}#{churn}#{probability}#{reason}#{preventive_steps}\n")
            
    