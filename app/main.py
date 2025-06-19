from fastapi import FastAPI
from typing import Optional

app = FastAPI(
    title="My FastAPI App",
    description="Ứng dụng FastAPI mẫu",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "Chào mừng đến với FastAPI!"}

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Xin chào {name}!"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

@app.post("/items/")
async def create_item(item: dict):
    return {"item": item, "status": "created"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}