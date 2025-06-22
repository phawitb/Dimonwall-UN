# main.py

from fastapi import FastAPI, HTTPException
from typing import List
import pandas as pd
import os

app = FastAPI()

# Path to prediction results
CSV_PATH = os.path.join("output", "predicted_results.csv")

# Load data once when server starts
try:
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
except Exception as e:
    print(f"❌ Failed to load CSV: {e}")
    df = pd.DataFrame()

# ✅ GET: all person data
@app.get("/person/all", response_model=List[dict])
def get_all_persons():
    if df.empty:
        raise HTTPException(status_code=500, detail="Data not available")
    return df.to_dict(orient="records")

# ✅ GET: person by ID
@app.get("/person/{person_id}", response_model=dict)
def get_person_by_id(person_id: str):
    if df.empty:
        raise HTTPException(status_code=500, detail="Data not available")
    person = df[df['id'] == person_id]
    if person.empty:
        raise HTTPException(status_code=404, detail=f"Person ID '{person_id}' not found")
    return person.iloc[0].dropna().to_dict()

# ✅ GET: top N persons
@app.get("/person/top/{n}", response_model=List[dict])
def get_top_n(n: int):
    if df.empty:
        raise HTTPException(status_code=500, detail="Data not available")

    if n < 1:
        raise HTTPException(status_code=400, detail="n must be >= 1")

    top_df = df.sort_values(by="Predicted_UNMEM_Score", ascending=False).head(n)
    return top_df.to_dict(orient="records")
