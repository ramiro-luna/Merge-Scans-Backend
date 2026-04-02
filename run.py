import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="[IP_ADDRESS]", port=8000, reload=True)