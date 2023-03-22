from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import Union, Optional
from process import ReturnInfoCard, ReturnInforCardBase64
import os
import shutil
from starlette import status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

class Item(BaseModel):
    name: Optional[str] = 'anhCCCD.jpg'
    stringbase64: Union[str, None] = None
class LimitUploadSize(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, max_upload_size: int) -> None:
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method == 'POST':
            if 'content-length' not in request.headers:
                return Response(status_code=status.HTTP_411_LENGTH_REQUIRED)
            content_length = int(request.headers['content-length'])
            if content_length > self.max_upload_size:
                return Response(status_code=status.HTTP_431_REQUEST_HEADER_FIELDS_TOO_LARGE)
        return await call_next(request)

app = FastAPI()
app.add_middleware(LimitUploadSize, max_upload_size=5000000)  # ~3MB
@app.post("/TDReaderIDCard/uploadFile")
async def uploadFile(file: UploadFile = File(...)):
    try:
        pathSave = os.getcwd() +'/'+ 'anhCCCD'
        if (os.path.exists(pathSave)):
            with open(f'anhCCCD/{file.filename}','wb') as buffer:
                shutil.copyfileobj(file.file, buffer)
        else:
            os.mkdir(pathSave)
            with open(f'anhCCCD/{file.filename}','wb') as buffer:
                shutil.copyfileobj(file.file, buffer)
        obj = await ReturnInfoCard(f'anhCCCD/{file.filename}')
        if(obj.errorCode==0):
            if (obj.type == "cccd_chip_front" or obj.type == "cccd_12_front"):
                return {"errorCode": obj.errorCode, "errorMessage": obj.errorMessage,
                        "data":[{"id": obj.id, "name": obj.name.upper(), "dob": obj.dob,"sex": obj.sex,
                        "nationality": obj.nationality,"home": obj.home, "address": obj.address, "doe": obj.doe,"type": obj.type}]}
            elif (obj.type == "cccd_chip_back" or obj.type == "cccd_12_back"):
                return {"errorCode": obj.errorCode, "errorMessage": obj.errorMessage,
                        "data":[{"features": obj.features, "issue_date": obj.issue_date,
                        "type": obj.type}]}
        else:
            shutil.move(f'anhCCCD/{file.filename}', f'invalid-image/{file.filename}')
            return {"errorCode": obj.errorCode, "errorMessage": obj.errorMessage, "data": []}
    except Exception:
        return {"message": "There was an error uploading the file"}
@app.post("/TDReaderIDCard/uploadBase64")
async def uploadBase64(item: Item):        
        obj = await ReturnInforCardBase64(item.name, item.stringbase64)
        if(obj.errorCode==0):
            if (obj.type == "cccd_chip_front" or obj.type == "cccd_12_front"):
                return {"errorCode": obj.errorCode, "errorMessage": obj.errorMessage,
                        "data":[{"id": obj.id, "name": obj.name.upper(), "dob": obj.dob,"sex": obj.sex,
                        "nationality": obj.nationality,"home": obj.home, "address": obj.address, "doe": obj.doe,"type": obj.type}]}
            elif (obj.type == "cccd_chip_back" or obj.type == "cccd_12_back"):
                return {"errorCode": obj.errorCode, "errorMessage": obj.errorMessage,
                        "data":[{"features": obj.features, "issue_date": obj.issue_date,
                        "type": obj.type}]}
        elif(obj.errorCode!=6 and obj.errorCode!=7 ):
            shutil.move(f'anhCCCD/{item.name}', f'invalid-image/{item.name}')
            return {"errorCode": obj.errorCode, "errorMessage": obj.errorMessage, "data": []}
        else:
            return {"errorCode": obj.errorCode, "errorMessage": obj.errorMessage, "data": []}