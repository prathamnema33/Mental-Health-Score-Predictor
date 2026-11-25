import joblib
import __main__
import pandas as pd
from fastapi import FastAPI
import pydantic
from pydantic import BaseModel,Field,BeforeValidator,AfterValidator
from typing import Optional,Literal
from fastapi.middleware.cors import CORSMiddleware

def group_countries(X):

    X_df = pd.DataFrame(X).copy()
    top_countries = ['Other', 'India', 'USA', 'Canada', 'Australia', 'UK', 'Germany', 'Mexico', 'Turkey', 'France']
    if 'Country' in X_df.columns:
        X_df['Country'] = X_df['Country'].apply(lambda x: x if x in top_countries else 'Other')
    return X_df

__main__.group_countries = group_countries

model=joblib.load('Mental_Health_Predictor.pkl')
class Student_Data(BaseModel):
    Age:int=Field(gt=0)
    Gender:Literal['Male',"Female"]=Field(...,description='Select a gender')
    Country:Literal['Afghanistan', 'Albania', 'Andorra', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Belarus', 'Belgium', 'Bhutan', 'Bolivia', 'Bosnia', 'Brazil', 'Bulgaria', 'Canada', 'Chile', 'China', 'Colombia', 'Costa Rica', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Ecuador', 'Egypt', 'Estonia', 'Finland', 'France', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Hong Kong', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Latvia', 'Lebanon', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malaysia', 'Maldives', 'Malta', 'Mexico', 'Moldova', 'Monaco', 'Montenegro', 'Morocco', 'Nepal', 'Netherlands', 'New Zealand', 'Nigeria', 'North Macedonia', 'Norway', 'Oman', 'Other', 'Pakistan', 'Panama', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia', 'San Marino', 'Serbia', 'Singapore', 'Slovakia', 'Slovenia', 'South Africa', 'South Korea', 'Spain', 'Sri Lanka', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Thailand', 'Trinidad', 'Turkey', 'UAE', 'UK', 'USA', 'Ukraine', 'Uruguay', 'Uzbekistan', 'Vatican City', 'Venezuela', 'Vietnam', 'Yemen']=Field(...,description='Select Country')
    Academic_level:Literal['Graduate', 'High School', 'Undergraduate']=Field(...,description='Select Academic Level')
    Most_Used_Platform:Literal['Facebook', 'Instagram', 'KakaoTalk', 'LINE', 'LinkedIn', 'Snapchat', 'TikTok', 'Twitter', 'VKontakte', 'WeChat', 'WhatsApp', 'YouTube']=Field(...,description='Select Social Media Platform ')
    Purpose_of_Use:Literal['Education', 'Entertainment', 'Networking', 'News']=Field(...,description='Select Purpose of Use')
    Avg_Daily_Usage_Hours:float=Field(ge=0,le=24)
    Daily_Unlocks:int=Field(ge=0)
    Study_Hours:float=Field(ge=0,le=24)
    Physical_Activity_Hours:float=Field(ge=0,le=24)
    Sleep_Hours_Per_Night:float=Field(ge=0,le=24)
    Stress_Level:Literal['Low','Medium','High','Very High']=Field(...,description='Select The stress Level')
class PredictionResponse(BaseModel):
    predict_mental_health_score:float
app=FastAPI()
@app.get('/')
def home():
    return {"Message":'Hello World'}
@app.post('/predict',response_model=PredictionResponse)
def predict(data:Student_Data):
    input_rows={
        'Age':data.Age,
        'Gender':data.Gender,
        'Country':data.Country,
        'Academic_Level':data.Academic_level,
        'Most_Used_Platform':data.Most_Used_Platform,
        'Purpose_Of_Use':data.Purpose_of_Use,
        'Avg_Daily_Usage_Hours':data.Avg_Daily_Usage_Hours,
        'Daily_Unlocks':data.Daily_Unlocks,
        'Study_Hours':data.Study_Hours,
        'Physical_Activity_Hours':data.Physical_Activity_Hours,
        'Sleep_Hours_Per_Night':data.Sleep_Hours_Per_Night,
        'Stress_Level':data.Stress_Level
    }
    prediction=model.predict(pd.DataFrame([input_rows]))[0]
    return PredictionResponse(predict_mental_health_score=round(float(prediction),2))

