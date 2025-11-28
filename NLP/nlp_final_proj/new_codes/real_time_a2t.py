#!/usr/bin/env python3
"""
Real-time transcription from microphone -> AssemblyAI streaming API.

Usage:
    export ASSEMBLYAI_API_KEY="your_key_here"
    python assemblyai_realtime_mic.py

Press Ctrl+C to stop streaming. The script will commit remaining audio before closing.
"""
import os
import queue
import threading
import time
import json
import base64
import sys

import sounddevice as sd
import numpy as np
import websocket   # websocket-client

# config
API_KEY = os.environ.get("ASSEMBLYAI_API_KEY") or "<PASTE_YOUR_KEY_HERE>"
WS_URL = "wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000"  # latest universal streaming endpoint
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_DURATION = 0.2           # seconds per chunk sent (200ms)
BLOCKSIZE = int(SAMPLE_RATE * BLOCK_DURATION)   # samples per block
COMMIT_EVERY_SEC = 1.5         # commit buffer to server this often (seconds)

if API_KEY is None or API_KEY == "<PASTE_YOUR_KEY_HERE>":
    print("Please set ASSEMBLYAI_API_KEY environment variable or edit the script.")
    sys.exit(1)

# a thread-safe queue for audio chunks to send
audio_q = queue.Queue(maxsize=200)

# callback invoked by sounddevice for each audio block
def audio_callback(indata, frames, time_info, status):
    """
    indata: numpy array shape (frames, channels)
    dtype: float32 usually in [-1,1] — we convert to int16 PCM
    """
    if status:
        # print("Audio status:", status, file=sys.stderr)
        pass
    # convert float32 -> int16 PCM
    # clamp to [-1,1]
    arr = np.clip(indata, -1.0, 1.0)
    int16 = (arr * 32767).astype(np.int16)
    # convert to bytes (little-endian)
    chunk_bytes = int16.tobytes()
    try:
        audio_q.put_nowait(chunk_bytes)
    except queue.Full:
        # drop frames if the queue is full to avoid blocking audio callback
        pass

# WebSocket event handlers
def on_open(ws):
    print("[ws] connection opened")
    # start thread that sends audio from queue to server
    sender = threading.Thread(target=send_audio_worker, args=(ws,), daemon=True)
    sender.start()
    # start thread that periodically commits the buffer
    committer = threading.Thread(target=periodic_commit_worker, args=(ws,), daemon=True)
    committer.start()

def on_message(ws, message):
    try:
        obj = json.loads(message)
    except Exception:
        print("[ws] raw:", message)
        return
    # Print the full JSON (useful) but also try to show transcript text
    # AssemblyAI realtime messages commonly include 'type' and transcript fields
    mtype = obj.get("type")
    if mtype:
        # print short summary
        if mtype == "transcript" or mtype == "message":
            # transcript object can be nested
            text = obj.get("text") or obj.get("alternatives", [{}])[0].get("transcript")
            if text:
                print("[transcript]", text)
                return
        # print everything else compact
        print("[ws event]", mtype, "-", json.dumps(obj, ensure_ascii=False))
    else:
        print("[ws message]", json.dumps(obj, ensure_ascii=False))

def on_error(ws, error):
    print("[ws] error:", error, file=sys.stderr)

def on_close(ws, close_status_code, close_msg):
    print("[ws] closed", close_status_code, close_msg)

# Worker: pull chunks from queue, base64-encode, send as input_audio_buffer.append
def send_audio_worker(ws):
    """
    Continuously take chunks from audio_q and send them to the websocket as base64.
    Uses event type 'input_audio_buffer.append'
    """
    while True:
        try:
            chunk = audio_q.get()
        except Exception:
            time.sleep(0.01)
            continue
        if chunk is None:
            # sentinel for shutdown
            break
        # base64 encode
        b64 = base64.b64encode(chunk).decode("ascii")
        msg = {"type": "input_audio_buffer.append", "audio": b64}
        try:
            ws.send(json.dumps(msg))
        except Exception as e:
            print("[ws] send failed:", e)
            break

# Worker: periodically commit buffer so server will create a new item and transcribe
def periodic_commit_worker(ws):
    while True:
        time.sleep(COMMIT_EVERY_SEC)
        try:
            ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        except Exception as e:
            print("[ws] commit failed:", e)
            break

def main():
    # make the websocket app
    headers = [f"Authorization: {API_KEY}"]
    # Some AssemblyAI docs use wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000
    # Another endpoint is wss://streaming.assemblyai.com/v3/ws  — check your account docs if one fails.
    ws = websocket.WebSocketApp(
        WS_URL,
        header=headers,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    # start microphone capture in the main thread (sounddevice needs to be started here)
    try:
        with sd.RawInputStream(samplerate=SAMPLE_RATE,
                               blocksize=BLOCKSIZE,
                               dtype='float32',
                               channels=CHANNELS,
                               callback=audio_callback):
            print("Microphone stream started. Press Ctrl+C to stop.")
            # run the websocket in a dedicated thread so microphone callback remains responsive
            wst = threading.Thread(target=ws.run_forever, kwargs={"ping_interval": 20, "ping_timeout": 10}, daemon=True)
            wst.start()
            # wait while the ws thread runs and audio flows
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Interrupted by user — stopping")
    finally:
        # signal sender to stop by pushing sentinel
        try:
            audio_q.put_nowait(None)
        except Exception:
            pass
        # commit remaining audio and close socket nicely
        try:
            # give some time for queue to flush
            time.sleep(0.5)
            # send a final commit
            # create a new connection (or reuse ws) to ensure commit; here we try sending on existing ws
            ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            time.sleep(0.5)
        except Exception:
            pass
        try:
            ws.close()
        except Exception:
            pass
        print("Done.")

if __name__ == "__main__":
    main()
