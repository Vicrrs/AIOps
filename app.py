from fastapi import FastAPI

app =  FastAPI()

@app.get("/")
def home():
    return {"message": "AIOps API"}

@app.get("/health")
def health():
    return {"status": "ok"}
