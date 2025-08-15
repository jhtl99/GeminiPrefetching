from google import genai
from google.genai import types
# from google.colab import userdata
import os
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY environment variable")

client = genai.Client(api_key=GOOGLE_API_KEY)


def generate(s):

  #ADJUST HYPERPARAMETERS HERE
  temp = 1
  model_Version = "2.5"



  client = genai.Client(
      vertexai=True,
      project="561083428989",
      location="us-central1",
  )

  model = None
  if model_Version == "2.0":
  ##gemini 2.0
    model = "projects/561083428989/locations/us-central1/endpoints/4812796590752268288"

  elif model_Version == "2.5":
    ##gemini 2.5
    model = "projects/561083428989/locations/us-central1/endpoints/3554040494902214656"

  contents = [
    types.Content(
      role="user",
      parts=[
          types.Part(text=s) # <--- ADD YOUR PROMPT HERE
      ]
    )
  ]

  generate_content_config = types.GenerateContentConfig(
    temperature = temp,
    top_p = 0.95,
    max_output_tokens = 8192,
    # safety_settings = [types.SafetySetting(
    #   category="HARM_CATEGORY_HATE_SPEECH",
    #   threshold="OFF"
    # ),types.SafetySetting(
    #   category="HARM_CATEGORY_DANGEROUS_CONTENT",
    #   threshold="OFF"
    # ),types.SafetySetting(
    #   category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
    #   threshold="OFF"
    # ),types.SafetySetting(
    #   category="HARM_CATEGORY_HARASSMENT",
    #   threshold="OFF"
    # )],
  )
  response = client.models.generate_content(
      model=model,
      contents=contents,
      config=generate_content_config,
  )
  # print(response.text)
  return response
  # for chunk in client.models.generate_content_stream(
  #   model = model,
  #   contents = contents,
  #   config = generate_content_config,
  #   ):
  #   print(chunk.text, end="")

print(generate("hello! what version of gemini ai are you?"))