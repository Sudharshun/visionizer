from openai import OpenAI
import uvicorn

from fastapi import Body, FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse,JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseSettings
import json

class Settings(BaseSettings):
    OPENAI_API_KEY: str = 'OPENAI_API_KEY'

    class Config:
        env_file = '.env'

settings = Settings()

client = OpenAI(
     api_key=settings.OPENAI_API_KEY,
)

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
def index(request: Request, animal: str= Form(...)):
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=generate_prompt(animal),
        temperature=0.6,
    )
    result = response.choices[0].text
    return templates.TemplateResponse("index.html", {"request": request, "result": result})

from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse
import openai

app = FastAPI()

from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse
import openai

app = FastAPI()

@app.post("/generate_statements", response_class=JSONResponse)
async def generate_statements(request: Request, data: dict = Body(...)):
    vision = data.get("vision", "")
    mission = data.get("mission", "")
    values = data.get("values", "")

    system_prompt = (
        "You are an AI expert of framing great vision, mission statements based on provided values.\n"
        f"Vision: {vision}\n"
        f"Mission: {mission}\n"
        f"Values: {values}\n"
        "Please generate a vision and mission statement while trying to infuse the values provided.\n"
        "Here are some examples of vision and mission statements:\n"
        "Vision: To be the enterprises numero uno Database team solving all the Team Data needs\n"
        "Mission: To provide the best in class Database solutions to the team\n"
        "Please provide a well-crafted vision and mission statement. Your output must be a JSON with below format:\n"
        "{\"vision\": \"vision goes here\", \"mission\": \"mission goes here\"}"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Vision: {vision}\nMission: {mission}\nValues: {values}"}
        ],
        temperature=0.7,
    )
    result = response.choices[0].message.content.strip()
    # Parse the JSON string into a dictionary
    result_dict = json.loads(result)

    # Return the dictionary as a JSON response
    return JSONResponse(content=result_dict)


def generate_prompt(animal):
    return """Suggest three names for an animal that is a superhero.

Animal: Cat
Names: Captain Sharpclaw, Agent Fluffball, The Incredible Feline
Animal: Dog
Names: Ruff the Protector, Wonder Canine, Sir Barks-a-Lot
Animal: {}
Names:""".format(
        animal.capitalize()
    )

if __name__ == "__main__":
    uvicorn.run('app:app', host="localhost", port=5001, reload=True)
