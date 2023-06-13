from .Controller import Controller
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from core.useCases.callLinearModel.CallLinearModelUseCase import CallLinearModelUseCase


class CallLinearModelController(Controller):
    def run(self, req: Request, res: Response) -> Response:
        useCase = CallLinearModelUseCase()
        result = useCase.execute()
        return JSONResponse(content={"message": result})
