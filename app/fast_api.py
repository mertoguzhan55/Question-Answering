from dataclasses import dataclass
import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.logger import Logger
from app.llama_inference import LlamaInference

@dataclass
class FastAPIServer:
    host: str
    port: int
    reload: bool
    log_level: str
    model: LlamaInference
    logger: Logger
    
    def __post_init__(self):
        self.app = FastAPI()
        self.templates = Jinja2Templates(directory="app/templates")
        self.app.mount(f"/static", StaticFiles(directory="app/static"), name="static")
    def run(self):
        self.server()
        self.logger.info("Server Initialized!")
        uvicorn.run(app=self.app, host=self.host, port=self.port, log_level=self.log_level)
    def server(self):
        @self.app.get("/")
        async def base(request: Request):
            return self.templates.TemplateResponse("index.html", {"request": request})
        @self.app.post("/ask-question")
        async def send_text(request: Request):
            body = await request.json()
            question = body.get("question")
            answer = self.model.generate_text(question)
            return {"answer": answer}

if __name__ == "__main__":
    pass
