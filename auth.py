from fastapi import HTTPException, Depends
from fastapi.security import APIKeyHeader
import os
from dotenv import load_dotenv

load_dotenv()

api_key_to_user = {
    "minha_key_api_321": "user_123"
}

fake_db = {
    "user_123": {"name": "VERE", "models": ["yolo_v5"]}
}

API_KEY = os.getenv("API_KEY")

api_key_header = APIKeyHeader(name="Authorization")

# Validar o api key
async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="API KEY Inválida")
    
    user_id = api_key_to_user[api_key]
    
    # verificar se o usuario existe
    if user_id not in fake_db:
        raise HTTPException(status_code=404, detail="Usuário não encontrado")
    
    return fake_db[user_id]
