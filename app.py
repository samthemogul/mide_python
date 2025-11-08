from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
import os

# Load environment variables first so os.getenv reads them
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI()

client = genai.Client(api_key=GEMINI_API_KEY)


def generate_response(tweet: str, bank_name: str = "Zenith") -> str:
  """Generate a response from the GenAI client and return text.

  This keeps the logic synchronous; if the model call is slow you can
  run it in a threadpool from the async endpoint.
  """
  response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=(
      f"I am going to pass in a tweet on {bank_name} can you help me assess "
      f"the criticality of the attached tweet to customer service on a scale of "
      f"(0-100%), Tweet: {tweet}"
    ),
  )
  # SDK may provide .text or other representation; coerce to str as fallback
  return response.text



class GetResponseRequest(BaseModel):
  tweet: str
  bank_name: str = "Zenith"


class GetResponseResponse(BaseModel):
  response: str


@app.post("/get_response", response_model=GetResponseResponse)
async def get_response(req: GetResponseRequest):
  """Accept JSON body with tweet and optional bank_name, return generated text."""
  try:
    result = generate_response(req.tweet, req.bank_name)
    return {"response": result}
  except Exception as exc:
    raise HTTPException(status_code=500, detail=str(exc))
