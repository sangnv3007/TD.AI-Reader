# TD.AI-Reader
### Setup
Download folder [model](https://drive.google.com/drive/folders/1xnM5YZrCt6QQpDV0IkSec43lji4FlE21?usp=sharing), unzip and copy to project run directory:
```
model
└───transformerocr.pth
|
│   
└───det
│   └── [...]
│   
└───rec
    └── [...]
```
## Requirements

Python 3.7+

FastAPI stands on the shoulders of giants:

* <a href="https://www.starlette.io/" class="external-link" target="_blank">Starlette</a> for the web parts.
* <a href="https://pydantic-docs.helpmanual.io/" class="external-link" target="_blank">Pydantic</a> for the data parts.

## Installation
<div class="termy">

```console
$ pip install -r requirements.txt
```

<div class="termy">

### Run it

Run the server with:

<div class="termy">

```console
$ uvicorn main:app --reload

INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [28720]
INFO:     Started server process [28722]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

</div>

### Interactive API docs

Now go to <a href="http://127.0.0.1:8000/docs" class="external-link" target="_blank">http://127.0.0.1:8000/docs</a>.

You will see the automatic interactive API documentation (provided by <a href="https://github.com/swagger-api/swagger-ui" class="external-link" target="_blank">Swagger UI</a>):

### Alternative API docs

And now, go to <a href="http://127.0.0.1:8000/redoc" class="external-link" target="_blank">http://127.0.0.1:8000/redoc</a>.

You will see the alternative automatic documentation (provided by <a href="https://github.com/Rebilly/ReDoc" class="external-link" target="_blank">ReDoc</a>):
### Check it now 👌

Now go to <a href="http://127.0.0.1:8000/docs" class="external-link" target="_blank">http://127.0.0.1:8000/docs</a>.

* The interactive API documentation will be automatically updated, including the new body:

![Step 1](https://github.com/sangnv3007/TD.AI-Reader/blob/main/step1_IDCard.png)

* Click on the button "Try it out", it allows you to upload file and directly interact with the API:

![Step 2](https://github.com/sangnv3007/TD.AI-Reader/blob/main/step2_IDCard.png)

* Then click on the "Execute" button, the user interface will communicate with your API, get the results and show them on the screen:

Image example

![CCCD Test](https://github.com/sangnv3007/TD.AI-Reader/blob/main/CCCDJPEG%20(150)_IDCard.jpg)

Return results

![Step 3](https://github.com/sangnv3007/TD.AI-Reader/blob/main/step3_IDCard.png)
