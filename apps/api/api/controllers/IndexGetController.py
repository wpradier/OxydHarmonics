from .Controller import Controller
from fastapi import Request, Response
from fastapi.responses import JSONResponse


class IndexGetController(Controller):
    def run(self, req: Request, res: Response) -> Response:
        return JSONResponse(content={"message": "Hello World!"})
