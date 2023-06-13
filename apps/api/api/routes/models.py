from fastapi import APIRouter, Request, Response
from api.controllers.CallLinearModelController import CallLinearModelController

router = APIRouter()


@router.get("/linear")
def get_linear(req: Request, res: Response):
    controller = CallLinearModelController()
    return controller.run(req, res)
