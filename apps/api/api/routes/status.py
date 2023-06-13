from fastapi import APIRouter, Request, Response
from api.controllers.StatusGetController import StatusGetController

router = APIRouter()


@router.get("/")
def get_status(req: Request, res: Response):
    controller = StatusGetController()
    return controller.run(req, res)
