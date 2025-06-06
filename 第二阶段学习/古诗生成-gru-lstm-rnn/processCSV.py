import pandas as pd
import re

def is_five_character_quatrain(poem):
    lines = re.split(r"[，,。.！!？?；;：:\n]",poem)
    lines = [line.strip() for line in lines if line.strip()]
    return len(lines) == 4 and all( len(line) == 5 for line in lines)

if __name__ == "__main__":
    file = "day19/data/元.csv"
    df = pd.read_csv(file)

    poems = df['内容'].dropna()

    df_five = df[df["内容"].apply(is_five_character_quatrain)]

    df_five.to_csv("D:/Desktop/deepLearnProject/day19/data/5poems3.csv", index = False, encoding="utf-8")
    print(f"已成功提取并保存{len(df_five)}首五言律诗到5poems.csv")

    pass