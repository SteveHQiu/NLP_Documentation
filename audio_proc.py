import os

import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

#%%
def transcribeAudio(audio_path):
    
    # Read the audio file as binary data
    audio_file = open(audio_path, "rb")
    # with open(audio_path, "rb") as audio_file:
    #     audio_data = audio_file.read()
    
    # Make the API call to transcribe the audio using Whisper
    response = openai.Audio.transcribe(
        file=audio_file,
        model="whisper-1",
        response_format="text",
    )
    
    # Extract and return the transcription
    return response


#%%
if __name__ == "__main__":
    response = transcribeAudio("temp_out.wav")
    print(response)
