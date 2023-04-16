from fastapi import APIRouter, Request, Response
from ..controllers.IndexGetController import IndexGetController


router = APIRouter()


@router.get("/")
def get_index(req: Request, res: Response):
    controller = IndexGetController()
    return controller.run(req, res)
