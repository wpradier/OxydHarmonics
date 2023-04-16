from fastapi import status, Request, Response
from .Controller import Controller
from fastapi.responses import JSONResponse


class StatusGetController(Controller):

    def run(self, req: Request, res: Response) -> Response:
        res.status_code = status.HTTP_200_OK
        return JSONResponse(content={"status": "OK"})
