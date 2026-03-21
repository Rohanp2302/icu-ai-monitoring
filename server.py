from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# load AI output dataset
df = pd.read_csv("models/ui_final_backend_dataset.csv")


# doctor dashboard
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("icu_dashboard.html", {"request": request})


# family communication page
@app.get("/family", response_class=HTMLResponse)
def family(request: Request):
    return templates.TemplateResponse("family_view.html", {"request": request})


# patient AI data API
@app.get("/patient")
def patient():

    row = df.iloc[0]

    return {
        "summary": row["ai_summary"],
        "heart": row["cardiovascular"],
        "lungs": row["respiratory"],
        "kidneys": row["renal"],
        "risk": float(row["predicted_probability"]),
        "alert": row["alert"]
    }