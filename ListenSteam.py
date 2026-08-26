import argparse
import os
import threading

import httpx
from deepgram import DeepgramClient
from deepgram.core.events import EventType
from dotenv import load_dotenv

from naming import build_timestamped_name, NAME_MAX_LENGTH

load_dotenv()

DEEPGRAM_API_KEY = os.environ["DEEPGRAM_API_KEY"]

# Note: This is an English stream, update accordingly for other languages
STREAM_URL = "https://playerservices.streamtheworld.com/api/livestream-redirect/CSPANRADIOAAC.aac"


def build_output_filename(out_file: str | None) -> str:
    return f"{build_timestamped_name(out_file, 'DeepGram')}.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="Потоковая транскрибация аудио через Deepgram")
    parser.add_argument(
        "--OutFile",
        dest="out_file",
        default=None,
        help="Имя выходного файла (очищается от недопустимых символов, "
        f"обрезается до {NAME_MAX_LENGTH} символов и дополняется датой/временем)",
    )
    return parser.parse_args()


args = parse_args()
output_filename = build_output_filename(args.out_file)

client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

with open(output_filename, "w", encoding="utf-8") as out_file, client.listen.v1.connect(
    model="nova-3",
    language="en",
    smart_format=True,
    interim_results=True,
    endpointing=10,
) as connection:
    ready = threading.Event()

    def on_message(result):
        event_type = result.type
        if event_type == "Results":
            channel = result.channel
            alt = channel.alternatives[0]
            transcript = alt.transcript
            prefix = "[ FINAL ]" if result.is_final else "[Interim]"
            if transcript:
                print(f"{prefix} {transcript}")
                if result.is_final:
                    out_file.write(transcript + "\n")
                    out_file.flush()

    connection.on(EventType.OPEN, lambda _: ready.set())
    connection.on(EventType.MESSAGE, on_message)

    def stream():
        ready.wait()
        with httpx.stream("GET", STREAM_URL, follow_redirects=True) as response:
            for chunk in response.iter_bytes():
                connection.send_media(chunk)

    threading.Thread(target=stream, daemon=True).start()

    print(f"Transcribing {STREAM_URL}...")
    print(f"Saving transcript to {output_filename}")
    connection.start_listening()
