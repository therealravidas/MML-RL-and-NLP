import time
import requests

API_KEY = "f4eabc3180944d7f9f36134b537a0c7c"   # <-- paste your AssemblyAI API key
AUDIO_FILE = "/home/BTECH_7TH_SEM/Downloads/Boarding call.wav"  # <-- replace with your path

# --------------------------
# 1) Upload the audio file
# --------------------------
def upload_file(filename):
    headers = {"authorization": API_KEY}
    with open(filename, "rb") as f:
        response = requests.post(
            "https://api.assemblyai.com/v2/upload",
            headers=headers,
            data=f
        )
    response.raise_for_status()
    return response.json()["upload_url"]


# --------------------------
# 2) Start the transcription
# --------------------------
def start_transcription(upload_url):
    endpoint = "https://api.assemblyai.com/v2/transcript"
    json = {
        "audio_url": upload_url,
        "language_code": "en",   # or "en_in" for Indian English
        "auto_chapters": False,
        "speaker_labels": False
    }
    headers = {"authorization": API_KEY, "content-type": "application/json"}
    response = requests.post(endpoint, json=json, headers=headers)
    response.raise_for_status()
    return response.json()["id"]


# --------------------------
# 3) Poll until transcription is done
# --------------------------
def wait_for_completion(transcript_id):
    endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    headers = {"authorization": API_KEY}

    while True:
        response = requests.get(endpoint, headers=headers)
        response.raise_for_status()
        status = response.json()

        if status["status"] == "completed":
            return status["text"]
        elif status["status"] == "error":
            raise Exception("Transcription failed: " + status["error"])

        print("Status:", status["status"], "- waiting 5 sec...")
        time.sleep(5)


# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
    print("Uploading file...")
    upload_url = upload_file(AUDIO_FILE)
    print("Upload complete:", upload_url)

    print("Submitting for transcription...")
    transcript_id = start_transcription(upload_url)
    print("Transcript ID:", transcript_id)

    print("Waiting for transcription to finish...")
    text = wait_for_completion(transcript_id)

    print("\n====== TRANSCRIPT ======\n")
    print(text)
    print("\n========================\n")
