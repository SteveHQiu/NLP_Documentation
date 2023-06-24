#%%
import openai

#%%
def transcribe_audio_with_whisper(audio_path, api_key):
    openai.api_key = api_key
    
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


# %%
response = transcribe_audio_with_whisper("test.mp3", "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")
print(response)


