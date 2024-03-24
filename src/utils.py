from datetime import datetime
import pandas as pd
from langchain_openai  import OpenAI
from dotenv import load_dotenv
import os


load_dotenv() 
api_key = os.getenv('OPENAI_API_KEY')

llm = OpenAI(temperature=0, api_key=api_key)

def createPredictionPrompt(data):
    data = pd.DataFrame([data], index=[0])
    #print(data['Age'])
    text = f"""
    Based on the provided customer details, Analyze and predict whether the customer is likely to churn or not (0 for not churn, 1 for churn) , the reason and if customer may churn then what can be done to retain. the customer details are given below.
    Customer Details =  Age: {data['Age'][0]}, Gender: {data['Gender'][0]}, Total_Amount_invested: {data['Total_Amount_invested'][0]}, Duration_of_client_relationship: {data['Duration_of_client_relationship'][0]}, Months_since_last_activity: {data['Months_since_last_activity'][0]} (less than 5 Months_since_last_activity is considered good, customer will not churn.), Customer_rating_for_service: {data['Customer_rating_for_service'][0]} Customer_rating_for_service is in range 1 to 5, greaterthan to 3 is very good, lessthan 3 is bad, and 3 is good, Chat_Analysis_1: {data['Chat_Analysis_1'][0]}, Chat_Analysis_2: {data['Chat_Analysis_2'][0]}, Chat_Analysis_3: {data['Chat_Analysis_3'][0]}, Chat_Analysis_4: {data['Chat_Analysis_4'][0]}, Chat_Analysis_5: {data['Chat_Analysis_5'][0]}, Total_Returns_percentage_in_CAGR: {data['Total_Returns_percentage_in_CAGR'][0]}, Last_1_year_returns_percentage: {data['Last_1_year_returns_percentage'][0]}, Total_Number_of_Complaints_Raised_Last_Year: {data['Total_Number_of_Complaints_Raised'][0]}, Number_of_Unresolved_Issues: {data['Number_of_Unresolved_Issues'][0]}, Average_Salary_per_Month: {data['Average_Salary_per_Month'][0]}, Net_Promoter_Score: {data['Net_Promoter_Score'][0]} (range 1 to 10),  Average_monthly_Investment: {data['Average_monthly_Investment'][0]}, Active_Loans: {data['Active_Loans'][0]}, Customer_service_feedback_Analysis: {data['Customer_service_feedback_Analysis'][0]}, Customer_Category: {data['Customer_Category'][0]} For UHNW customer, Unacceptable if latest chat analysis is negative if it is Neutral then it is good, Unacceptable if more than 2 complaints were raised last year, Unacceptable if there is 1 unresolved issue. Weightage: 3 times. For HNW customer, Unacceptable if latest 2 chat analyses are negative, Unacceptable if more than 3 complaints were raised last year, Unacceptable if there are 2 unresolved issues.Weightage: 2 times. For MA customers, Unacceptable if latest 3 chat analyses are negative, Unacceptable if more than 5 complaints were raised last year, Unacceptable if there are 5 unresolved issues, Weightage: 1 time. Average good Customer_rating_for_service is 7. Neutral chat analysis and service feedback is considered acceptable for UHNW, HNW, MA. 
    Result should always be in below format.
    Prediction: 'prediction'
    Probability: 'probability of customer may churn based on the red flags in range 0.0 to 1.0'
    Reason: 'reason why customer may churn or not churn in paragraph format'
    Preventive Steps: 'steps to be taken to retain the customer to satisfy in paragraph format'
    """
    return text

def getCRMData():
    df = pd.read_csv('Data/train1.csv')

    df['Last_active_date'] = pd.to_datetime(df['Last_active_date'])
    today = datetime.today()
    df['Months_since_last_activity'] = (today - df['Last_active_date']) // pd.Timedelta(days=30)
    df = df.drop(columns=['Last_active_date'])
    return df



def predictChurn(data, llm=llm):
     
    pred = llm.predict(data)
    print(pred)
    return pred

def sentimentAnalysis(data, llm=llm):
    text = f"""
    Analyze and predict the sentiment of the Customer from the following convesation of Customer with Assistant.
    Text: {data}
    Output: 'sentiment (Positive / Neutral / Negative)'
    """
    pred = llm.predict(text)
    pred = pred.split('Sentiment:')[1]
    return pred
