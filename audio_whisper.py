import torch
import sounddevice as sd
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def record_audio(duration=5, fs=16000):
    print(f"\nMów teraz! (Nagrywam {duration} sekund...)")

    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')

    sd.wait()
    print("Koniec nagrywania. Przetwarzam...\n")

    return recording.flatten()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uruchamianie silnika Whisper na: {device}...")

    model_id = "openai/whisper-tiny"
    processor = WhisperProcessor.from_pretrained(model_id)

    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    print("System nasłuchu gotowy!")

    while True:
        input("Naciśnij [ENTER], aby rozpocząć nagrywanie (lub wpisz 'exit', aby wyjść)... ")
        audio_array = record_audio(duration=5)

        input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        predicted_ids = model.generate(input_features, language="en")
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        print("=" * 50)
        print(f"🗣️ ROZPOZNANY TEKST: {transcription.strip()}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()