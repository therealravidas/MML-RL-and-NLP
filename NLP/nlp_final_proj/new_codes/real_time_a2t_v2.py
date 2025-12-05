#!/usr/bin/env python3
"""
Real-time microphone -> AssemblyAI Universal-Streaming (v3) WebSocket example.

Set environment variable:
  export ASSEMBLYAI_API_KEY="sk_..."

Then run:
  python assemblyai_realtime_universal.py
"""
import os, sys, queue, threading, time, json, base64
import sounddevice as sd
import numpy as np
import websocket

API_KEY = os.environ.get("ASSEMBLYAI_API_KEY") or "<PASTE_KEY>"
# Use the universal streaming v3 endpoint. Add sample_rate and optionally speech_model.
# For EU use: "wss://streaming.eu.assemblyai.com/v3/ws"
SAMPLE_RATE = 16000
# Choose model: default universal (English) or multilingual:
SPEECH_MODEL = "universal-streaming-english"   # or "universal-streaming-multilingual"
WS_URL = f"wss://streaming.assemblyai.com/v3/ws?sample_rate={SAMPLE_RATE}&speech_model={SPEECH_MODEL}"

# small chunking config
BLOCK_DURATION = 0.2                 # seconds per chunk sent
BLOCKSIZE = int(SAMPLE_RATE * BLOCK_DURATION)
CHANNELS = 1
COMMIT_EVERY_SEC = 1.2

if API_KEY is None or API_KEY == "<PASTE_KEY>":
    print("Set ASSEMBLYAI_API_KEY env var")
    sys.exit(1)

audio_q = queue.Queue(maxsize=400)

def audio_callback(indata, frames, time_info, status):
    if status:
        pass
    arr = np.clip(indata, -1.0, 1.0)
    int16 = (arr * 32767).astype(np.int16)
    try:
        audio_q.put_nowait(int16.tobytes())
    except queue.Full:
        pass

def send_audio_worker(ws, stop_event):
    while not stop_event.is_set():
        try:
            chunk = audio_q.get(timeout=0.1)
        except Exception:
            continue
        if chunk is None:
            break
        b64 = base64.b64encode(chunk).decode("ascii")
        msg = {"type":"input_audio_buffer.append","audio": b64}
        try:
            ws.send(json.dumps(msg))
        except Exception:
            break

def periodic_commit_worker(ws, stop_event):
    while not stop_event.is_set():
        time.sleep(COMMIT_EVERY_SEC)
        try:
            ws.send(json.dumps({"type":"input_audio_buffer.commit"}))
        except Exception:
            break

def on_message(ws, message):
    try:
        obj = json.loads(message)
    except Exception:
        print("[ws] raw:", message); return
    mtype = obj.get("type")
    if mtype == "session.connected":
        print("[ws] session started")
    elif mtype == "transcript" or mtype == "turn":
        # print whatever text is present (structure can vary)
        text = obj.get("text") or obj.get("alternatives", [{}])[0].get("transcript")
        if text:
            print("[transcript]", text)
        else:
            print("[ws event]", json.dumps(obj, ensure_ascii=False))
    else:
        # debug other events (endpointing, session, etc.)
        print("[ws event]", mtype, "-", json.dumps(obj, ensure_ascii=False))

def on_open(ws):
    print("[ws] opened")

def on_error(ws, error):
    print("[ws] error:", error)

def on_close(ws, code, reason):
    print("[ws] closed", code, reason)

def run_ws_session():
    # build headers and websocket
    headers = [f"Authorization: {API_KEY}"]
    stop_event = threading.Event()
    while True:
        ws_app = websocket.WebSocketApp(
            WS_URL,
            header=headers,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        # threads to send audio + commit (they need a live ws object)
        sender = threading.Thread(target=lambda: None)
        committer = threading.Thread(target=lambda: None)
        # start ws in thread and then start senders after connection open
        wst = threading.Thread(target=ws_app.run_forever, kwargs={"ping_interval":20, "ping_timeout":10}, daemon=True)
        wst.start()
        # wait briefly for socket to become ready
        time.sleep(0.5)
        if not wst.is_alive():
            print("[ws] run_forever thread died immediately; retrying in 2s")
            time.sleep(2)
            continue
        # start workers
        sender = threading.Thread(target=send_audio_worker, args=(ws_app, stop_event), daemon=True)
        sender.start()
        committer = threading.Thread(target=periodic_commit_worker, args=(ws_app, stop_event), daemon=True)
        committer.start()
        # wait while thread is alive
        try:
            while wst.is_alive():
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("Interrupted by user")
        # cleanup and maybe reconnect
        stop_event.set()
        try:
            ws_app.close()
        except Exception:
            pass
        # decide whether to reconnect
        print("[ws] connection ended — reconnecting in 2s")
        time.sleep(2)
        stop_event.clear()

def main():
    # start microphone capture and websocket session runner
    try:
        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE, dtype='float32', channels=CHANNELS, callback=audio_callback):
            print("Microphone started; connecting to AssemblyAI Universal-Streaming...")
            run_ws_session()
    except KeyboardInterrupt:
        print("Stopping")
    finally:
        try:
            audio_q.put_nowait(None)
        except Exception:
            pass

if __name__ == "__main__":
    main()
