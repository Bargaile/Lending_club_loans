from pydantic import BaseModel


class LoanPrediction(BaseModel):
  loan_amnt: float
  purpose: str
  fico_score: float
  dti: float
  addr_state: str
  emp_length: int
  year: int
  month: int




class GradePrediction(BaseModel):
  loan_amnt:float
  term: str
  emp_title: str
  emp_length:int
  home_ownership:str
  annual_inc:float
  purpose:str
  addr_state:str
  dti:float
  delinq_2yrs:float
  inq_last_6mths:float
  open_acc:float
  pub_rec:float
  total_acc:float
  application_type:str
  inq_last_12m:float
  acc_open_past_24mths:float
  avg_cur_bal:float
  mo_sin_old_il_acct:float
  mort_acc:float
  pct_tl_nvr_dlq:float
  pub_rec_bankruptcies:float
  fico_score:float
  last_fico_score:float
  month:int
  year:int
  earliest_cr_line_year:int

  
class SubGradePrediction(BaseModel):
  loan_amnt:float
  term: str
  emp_title: str
  emp_length:int
  home_ownership:str
  annual_inc:float
  purpose:str
  addr_state:str
  dti:float
  delinq_2yrs:float
  inq_last_6mths:float
  open_acc:float
  pub_rec:float
  total_acc:float
  application_type:str
  inq_last_12m:float
  acc_open_past_24mths:float
  avg_cur_bal:float
  mo_sin_old_il_acct:float
  mort_acc:float
  pct_tl_nvr_dlq:float
  pub_rec_bankruptcies:float
  fico_score:float
  last_fico_score:float
  month:int
  year:int
  earliest_cr_line_year:int

class InterestRatePrediction(BaseModel):
  loan_amnt:float
  term: str
  emp_title: str
  emp_length:int
  home_ownership:str
  annual_inc:float
  purpose:str
  addr_state:str
  dti:float
  delinq_2yrs:float
  inq_last_6mths:float
  open_acc:float
  pub_rec:float
  total_acc:float
  application_type:str
  inq_last_12m:float
  acc_open_past_24mths:float
  avg_cur_bal:float
  mo_sin_old_il_acct:float
  mort_acc:float
  pct_tl_nvr_dlq:float
  pub_rec_bankruptcies:float
  fico_score:float
  last_fico_score:float
  month:int
  year:int
  earliest_cr_line_year:int
  

purpose_dict = {
  "debt_consolidation": "debt_consolidation",
  "credit_card":"credit_card",
  "other":"other",
  "home_improvement":"home_improvement",
  "major_purchase":"major_purchase",
  "medical":"medical",
  "car":"car",
  "small_business":"small_business",
  "vacation":"vacation",
  "moving":"moving",
  "house":"house",
  "renewable_energy":"renewable_energy",
  "wedding":"wedding",
  "educational":"educational",
  }

addr_state_dict = {
  "AL": "AL",
  "AK": "AK",
  "AZ": "AZ",
  "AR": "AR",
  "CA": "CA",
  "CO": "CO",
  "CT": "CT",
  "DE": "DE",
  "FL": "FL",
  "GA": "GA",
  "HI": "HI",
  "ID": "ID",
  "IL": "IL",
  "IN": "IN",
  "IA": "IA",
  "KS": "KS",
  "KY": "KY",
  "LA": "LA",
  "ME": "ME",
  "MD": "MD",
  "MA": "MA",
  "MI": "MI",
  "MN": "MN",
  "MS": "MS",
  "MO": "MO",
  "MT": "MT",
  "NE": "NE",
  "NV": "NV",
  "NH": "NH",
  "NJ": "NJ",
  "NM": "NM",
  "NY": "NY",
  "NC": "NC",
  "ND": "ND",
  "OH": "OH",
  "OK": "OK",
  "OR": "OR",
  "PA": "PA",
  "RI": "RI",
  "SC": "SC",
  "SD": "SD",
  "TN": "TN",
  "TX": "TX",
  "UT": "UT",
  "VT": "VT",
  "VA": "VA",
  "WA": "WA",
  "WV": "WV",
  "WI": "WI",
  "WY": "WY",
  "DC": "DC",
  "AS": "AS",
  "GU": "GU",
  "MP": "MP",
  "PR": "PR",
  "UM": "UM",
  "VI": "VI",
}


term_dict = {
  " 36 months": " 36 months", 
  " 60 months": " 60 months",
}


emp_title_dict = {
  "Business":"Business",
  "Other":"Other",
  "IT, Tech sector":"IT, Tech sector",
  "Private services, production, transportation":"Private services, production, transportation",
  "Law, enforcement":"Law, enforcement",
  "Education, arts, culture":"Education, arts, culture",
  "Medicine, pharmacy":"Medicine, pharmacy",
  "Government sector":"Government sector",
}

home_ownership_dict = {
  "mortgage": "mortgage",
  "rent": "rent",
  "own": "own",
  "other":"other",
}

application_type_dict = {
    "Individual":"Individual",
    "Joint App":"Joint App",
}
