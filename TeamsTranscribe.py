from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    from teams_transcribe.config import load_settings
    from teams_transcribe.gui import run

    # DEEPGRAM_API_KEY is optional: without it the app offers only the local WhisperX engine.
    run(load_settings())
