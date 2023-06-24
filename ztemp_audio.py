import pyaudio
import wave

def record_audio(file_path, duration=1, sample_rate=44100, channels=1):
    audio = pyaudio.PyAudio()

    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=1024)

    print("Recording audio...")
    frames = []

    for i in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    print("Recording completed.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save audio 
    audio_file = wave.open(file_path, "wb")
    audio_file.setnchannels(1)
    audio_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    audio_file.setframerate(sample_rate)
    audio_file.writeframes(b"".join(frames)) # Join frames together
    audio_file.close()
    
    

# Example usage
record_audio('output.wav', duration=1)
