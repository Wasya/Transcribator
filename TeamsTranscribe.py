import os
import sys

from dotenv import load_dotenv

load_dotenv()

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")

if __name__ == "__main__":
    if not DEEPGRAM_API_KEY:
        print("DEEPGRAM_API_KEY не найден. Создайте .env на основе .env.example.", file=sys.stderr)
        sys.exit(1)

    from teams_transcribe.gui import run

    run(DEEPGRAM_API_KEY)
