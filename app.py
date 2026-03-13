from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Multi Agent Reasoning Core is active!"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860)
