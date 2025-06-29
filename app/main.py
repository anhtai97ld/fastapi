from fastapi import FastAPI, UploadFile, File
import pandas as pd
import io
from app.predict import predict_house_prices

app = FastAPI(
    title="My FastAPI App",
    description="Ứng dụng FastAPI mẫu",
    version="1.0.0"
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Nhận một tệp CSV và trả về dự đoán giá nhà.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        
        predictions = predict_house_prices(df)
        if predictions is not None:
            return {"predictions": predictions.tolist()}
        else:
            return {"error": "Không thể thực hiện dự đoán."}
    except Exception as e:
        return {"error": str(e)}
