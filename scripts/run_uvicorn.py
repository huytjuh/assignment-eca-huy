import os 
import uvicorn

from configs.fastapi_config import get_fastapi_config

def main():
    uvicorn_config = get_fastapi_config()
    uvicorn.run('api.app:app', host=uvicorn_config.host, port=uvicorn_config.port, reload=False)

if __name__ == "__main__":
    main()