from asr import ASR

print("Loading Model...")
prediction = ASR()

audio_path = ""
print("Model Load successfully")
while audio_path != "q":
    audio_path = input("Enter audio path:")
    if audio_path != "q":
        result = prediction.predict(audio_path)

        print(result)


print("thanks...")