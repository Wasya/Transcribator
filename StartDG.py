import os

from deepgram import DeepgramClient
from dotenv import load_dotenv

load_dotenv()

DEEPGRAM_API_KEY = os.environ["DEEPGRAM_API_KEY"]

AUDIO_URL = "https://static.deepgram.com/examples/Bueller-Life-moves-pretty-fast.wav"

def main():
    try:
        deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)

        response = deepgram.listen.v1.media.transcribe_url(
            url=AUDIO_URL,
            model="nova-3",
            language="en",
        )

        # Print the full response object
        print(response)

        # Or access the transcript directly
        print("\nTranscript:")
        print(response.results.channels[0].alternatives[0].transcript)

    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    main()
