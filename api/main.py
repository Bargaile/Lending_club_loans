import joblib
import uvicorn, os
import pandas as pd
from fastapi import FastAPI, HTTPException
from xgboost import XGBClassifier
from lightgbm import LGBMRegressor
from classes import *


app = FastAPI()

loan_predictor = joblib.load("model_xgb_loans.joblib")
grade_predictor = joblib.load("model_xgb_grade.joblib")
sub_grade_predictor = joblib.load("model_xgb_sub.joblib")
interest_rate_predictor = joblib.load("model_lgbm_int.joblib")


@app.get("/")
def home():
    return {"text": "Lending Club loan predictions"}

@app.post("/loan_prediction")
async def create_application(loan_pred: LoanPrediction):

    loan_df = pd.DataFrame()

    if loan_pred.purpose not in purpose_dict:
        raise HTTPException(status_code=404, detail="Purpose of the loan not found")

    if loan_pred.addr_state not in addr_state_dict:
        raise HTTPException(status_code=404, detail="Please insert a valid abbreviation of the US State title")

    loan_df["loan_amnt"] = [loan_pred.loan_amnt]
    loan_df["purpose"] = [loan_pred.purpose]
    loan_df["fico_score"] = [loan_pred.fico_score]
    loan_df["dti"] = [loan_pred.dti]
    loan_df["addr_state"]= [loan_pred.addr_state]
    loan_df["emp_length"] = [loan_pred.emp_length]
    loan_df["year"] = [loan_pred.year]
    loan_df["month"] = [loan_pred.month]


    prediction = loan_predictor.predict(loan_df)
    if prediction[0] == 0:
        prediction = "Your loan application is rejected"
    else:
        prediction = "Your loan application is approved"

    return {"prediction": prediction}


@app.post("/grade_prediction")
async def create_application(grade_pred: GradePrediction):

    grade_df = pd.DataFrame()

    if grade_pred.term not in term_dict:
        raise HTTPException(status_code=404, detail="Please insert term of 36 or 60 months")

    if grade_pred.emp_title not in emp_title_dict:
        raise HTTPException(status_code=404, detail="Please choose another employment title")
    
    if grade_pred.home_ownership not in home_ownership_dict:
        raise HTTPException(status_code=404, detail="This home ownership status is not found")
    
    if grade_pred.purpose not in purpose_dict:
        raise HTTPException(status_code=404, detail="Please select a valid loan purpose")
    
    if grade_pred.addr_state not in addr_state_dict:
        raise HTTPException(status_code=404, detail="Please insert a valid abbreviation of the US State title")
    
    if grade_pred.application_type not in application_type_dict:
        raise HTTPException(status_code=404, detail="Please select a valid application type")

    grade_df["loan_amnt"] = [grade_pred.loan_amnt]
    grade_df["term"] = [grade_pred.term]
    grade_df["emp_title"] = [grade_pred.emp_title]
    grade_df["emp_length"]= [grade_pred.emp_length]
    grade_df["home_ownership"] = [grade_pred.home_ownership]
    grade_df["annual_inc"] = [grade_pred.annual_inc]
    grade_df["purpose"] = [grade_pred.purpose]
    grade_df["addr_state"] = [grade_pred.addr_state]
    grade_df["dti"] = [grade_pred.dti]
    grade_df["delinq_2yrs"] = [grade_pred.delinq_2yrs]
    grade_df["inq_last_6mths"] = [grade_pred.inq_last_6mths]
    grade_df["open_acc"] = [grade_pred.open_acc]
    grade_df["pub_rec"] = [grade_pred.pub_rec]
    grade_df["total_acc"] = [grade_pred.total_acc]
    grade_df["application_type"] = [grade_pred.application_type]
    grade_df["inq_last_12m"] = [grade_pred.inq_last_12m]
    grade_df["acc_open_past_24mths"] = [grade_pred.acc_open_past_24mths]
    grade_df["avg_cur_bal"] = [grade_pred.avg_cur_bal]
    grade_df["mo_sin_old_il_acct"] = [grade_pred.mo_sin_old_il_acct]
    grade_df["mort_acc"] = [grade_pred.mort_acc]
    grade_df["pct_tl_nvr_dlq"] = [grade_pred.pct_tl_nvr_dlq]
    grade_df["pub_rec_bankruptcies"] = [grade_pred.pub_rec_bankruptcies]
    grade_df["fico_score"] = [grade_pred.fico_score]
    grade_df["last_fico_score"] = [grade_pred.last_fico_score]
    grade_df["month"] = [grade_pred.month]
    grade_df["year"] = [grade_pred.year]
    grade_df["earliest_cr_line_year"] = [grade_pred.earliest_cr_line_year]

    prediction = grade_predictor.predict(grade_df)
    if prediction[0] == 0:
        prediction = "You are likely to get grade 'A'"
    elif prediction[0] == 1:
        prediction = "You are likely to get grade 'B'"
    elif prediction[0] == 2:
        prediction = "You are likely to get grade 'C'"
    elif prediction[0] == 3:
        prediction = "You are likely to get grade 'D'"
    elif prediction[0] == 4:
        prediction = "You are likely to get grade 'E'"
    elif prediction[0] == 5:
        prediction = "You are likely to get grade 'F'"
    else:
        prediction = "You are likely to get grade 'G'"

    return {"prediction": prediction}


@app.post("/sub_grade_prediction")
async def create_application(sub_grade_pred: SubGradePrediction):


    sub_df = pd.DataFrame()

    if sub_grade_pred.term not in term_dict:
        raise HTTPException(status_code=404, detail="Please insert term of 36 or 60 months")

    if sub_grade_pred.emp_title not in emp_title_dict:
        raise HTTPException(status_code=404, detail="Please choose another employment title")
    
    if sub_grade_pred.home_ownership not in home_ownership_dict:
        raise HTTPException(status_code=404, detail="This home ownership status is not found")
    
    if sub_grade_pred.purpose not in purpose_dict:
        raise HTTPException(status_code=404, detail="Please select a valid loan purpose")
    
    if sub_grade_pred.addr_state not in addr_state_dict:
        raise HTTPException(status_code=404, detail="Please insert a valid abbreviation of the US State title")
    
    if sub_grade_pred.application_type not in application_type_dict:
        raise HTTPException(status_code=404, detail="Please select a valid application type")

    sub_df["loan_amnt"] = [sub_grade_pred.loan_amnt]
    sub_df["term"] = [sub_grade_pred.term]
    sub_df["emp_title"] = [sub_grade_pred.emp_title]
    sub_df["emp_length"]= [sub_grade_pred.emp_length]
    sub_df["home_ownership"] = [sub_grade_pred.home_ownership]
    sub_df["annual_inc"] = [sub_grade_pred.annual_inc]
    sub_df["purpose"] = [sub_grade_pred.purpose]
    sub_df["addr_state"] = [sub_grade_pred.addr_state]
    sub_df["dti"] = [sub_grade_pred.dti]
    sub_df["delinq_2yrs"] = [sub_grade_pred.delinq_2yrs]
    sub_df["inq_last_6mths"] = [sub_grade_pred.inq_last_6mths]
    sub_df["open_acc"] = [sub_grade_pred.open_acc]
    sub_df["pub_rec"] = [sub_grade_pred.pub_rec]
    sub_df["total_acc"] = [sub_grade_pred.total_acc]
    sub_df["application_type"] = [sub_grade_pred.application_type]
    sub_df["inq_last_12m"] = [sub_grade_pred.inq_last_12m]
    sub_df["acc_open_past_24mths"] = [sub_grade_pred.acc_open_past_24mths]
    sub_df["avg_cur_bal"] = [sub_grade_pred.avg_cur_bal]
    sub_df["mo_sin_old_il_acct"] = [sub_grade_pred.mo_sin_old_il_acct]
    sub_df["mort_acc"] = [sub_grade_pred.mort_acc]
    sub_df["pct_tl_nvr_dlq"] = [sub_grade_pred.pct_tl_nvr_dlq]
    sub_df["pub_rec_bankruptcies"] = [sub_grade_pred.pub_rec_bankruptcies]
    sub_df["fico_score"] = [sub_grade_pred.fico_score]
    sub_df["last_fico_score"] = [sub_grade_pred.last_fico_score]
    sub_df["month"] = [sub_grade_pred.month]
    sub_df["year"] = [sub_grade_pred.year]
    sub_df["earliest_cr_line_year"] = [sub_grade_pred.earliest_cr_line_year]

    prediction = sub_grade_predictor.predict(sub_df)
    if prediction[0] == 0:
        prediction = "You're likely to get sub_grade A1"
    elif prediction[0] == 1:
        prediction = "YYou're likely to get sub_grade A2"
    elif prediction[0] == 2:
        prediction = "You're likely to get sub_grade A3"
    elif prediction[0] == 3:
        prediction = "You're likely to get sub_grade A4"    
    else:
        prediction = "You're likely to get sub_grade A5"

    return {"prediction": prediction}


@app.post("/interest_rate_prediction")
async def create_application(int_rate_pred: InterestRatePrediction):

    rate_df = pd.DataFrame()

    if int_rate_pred.term not in term_dict:
        raise HTTPException(status_code=404, detail="Please insert term of 36 or 60 months")

    if int_rate_pred.emp_title not in emp_title_dict:
        raise HTTPException(status_code=404, detail="Please choose another employment title")
    
    if int_rate_pred.home_ownership not in home_ownership_dict:
        raise HTTPException(status_code=404, detail="This home ownership status is not found")
    
    if int_rate_pred.purpose not in purpose_dict:
        raise HTTPException(status_code=404, detail="Please select a valid loan purpose")
    
    if int_rate_pred.addr_state not in addr_state_dict:
        raise HTTPException(status_code=404, detail="Please insert a valid abbreviation of the US State title")
    
    if int_rate_pred.application_type not in application_type_dict:
        raise HTTPException(status_code=404, detail="Please select a valid application type")

    rate_df["loan_amnt"] = [int_rate_pred.loan_amnt]
    rate_df["term"] = [int_rate_pred.term]
    rate_df["emp_title"] = [int_rate_pred.emp_title]
    rate_df["emp_length"]= [int_rate_pred.emp_length]
    rate_df["home_ownership"] = [int_rate_pred.home_ownership]
    rate_df["annual_inc"] = [int_rate_pred.annual_inc]
    rate_df["purpose"] = [int_rate_pred.purpose]
    rate_df["addr_state"] = [int_rate_pred.addr_state]
    rate_df["dti"] = [int_rate_pred.dti]
    rate_df["delinq_2yrs"] = [int_rate_pred.delinq_2yrs]
    rate_df["inq_last_6mths"] = [int_rate_pred.inq_last_6mths]
    rate_df["open_acc"] = [int_rate_pred.open_acc]
    rate_df["pub_rec"] = [int_rate_pred.pub_rec]
    rate_df["total_acc"] = [int_rate_pred.total_acc]
    rate_df["application_type"] = [int_rate_pred.application_type]
    rate_df["inq_last_12m"] = [int_rate_pred.inq_last_12m]
    rate_df["acc_open_past_24mths"] = [int_rate_pred.acc_open_past_24mths]
    rate_df["avg_cur_bal"] = [int_rate_pred.avg_cur_bal]
    rate_df["mo_sin_old_il_acct"] = [int_rate_pred.mo_sin_old_il_acct]
    rate_df["mort_acc"] = [int_rate_pred.mort_acc]
    rate_df["pct_tl_nvr_dlq"] = [int_rate_pred.pct_tl_nvr_dlq]
    rate_df["pub_rec_bankruptcies"] = [int_rate_pred.pub_rec_bankruptcies]
    rate_df["fico_score"] = [int_rate_pred.fico_score]
    rate_df["last_fico_score"] = [int_rate_pred.last_fico_score]
    rate_df["month"] = [int_rate_pred.month]
    rate_df["year"] = [int_rate_pred.year]
    rate_df["earliest_cr_line_year"] = [int_rate_pred.earliest_cr_line_year]

    prediction = interest_rate_predictor.predict(rate_df)
    rounded_prediction = round(prediction[0],2)

    return {'prediction': rounded_prediction}

# uvicorn main:app --reload
