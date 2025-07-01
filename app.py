from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional
import os
# new 

# Importing constants and pipeline modules from the project
from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import question_Data, QuestionDataClassifier
from src.pipline.prediction_pipeline import duplicate_question_PredictorConfig
from src.pipline.training_pipeline import TrainPipeline

# Set NLTK data path
os.environ["NLTK_DATA"] = "/usr/share/nltk_data"

# Initialize FastAPI application
app = FastAPI()

# Mount the 'static' directory for serving static files (like CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 template engine for rendering HTML templates
templates = Jinja2Templates(directory='templates')

# Allow all origins for Cross-Origin Resource Sharing (CORS)
origins = ["*"]

# Configure middleware to handle CORS, allowing requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    """
    DataForm class to handle and process incoming form data.
    This class defines the question-related attributes expected from the form.
    """
    def __init__(self, request: Request):
        self.request: Request = request
        self.question1: Optional[str] = None
        self.question2: Optional[str] = None
                

    async def get_question_data(self):
        """
        Method to retrieve and assign form data to class attributes.
        This method is asynchronous to handle form data fetching without blocking.
        """
        form = await self.request.form()
        self.question1 = form.get("question1")
        self.question2 = form.get("question2")

# Route to render the main page with the form
@app.get("/", tags=["authentication"])
async def index(request: Request):
    """
    Renders the main HTML form page for question data input.
    """
    return templates.TemplateResponse(
            "index.html",{"request": request, "context": "Rendering"})

# Route to trigger the model training process
@app.get("/train")
async def trainRouteClient():
    """
    Endpoint to initiate the model training pipeline.
    """
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful!!!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")

# Route to handle form submission and make predictions
@app.post("/")
async def predictRouteClient(request: Request):
    """
    Endpoint to receive form data, process it, and make a prediction.
    """
    try:
        form = DataForm(request)
        await form.get_question_data()

        question_data = question_Data(
            question1=form.question1,
            question2=form.question2,
                                )

        # Convert form data into a DataFrame for the model
        question_pairs = question_data.get_question_input_data_frame()

        # Initialize the prediction pipeline
        model_predictor = QuestionDataClassifier()

        # Make a prediction and retrieve the result
        value = model_predictor.predict(dataframe=question_pairs)[0]

        # Interpret the prediction result as 'Response-Yes' or 'Response-No'
        status = "Duplicate" if value == 1 else "Not Duplicate"

        # Render the same HTML page with the prediction result
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "context": status},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}

# Main entry point to start the FastAPI server
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)