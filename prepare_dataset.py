import pandas as pd

data = []

with open("data/train.txt", "r", encoding="utf-8") as f:
    for line in f:
        text, emotion = line.strip().split(";")
        data.append([text, emotion])

df = pd.DataFrame(data, columns=["text", "emotion"])
df.to_csv("data/text_emotion_dataset.csv", index=False)

print("Dataset converted successfully!")


