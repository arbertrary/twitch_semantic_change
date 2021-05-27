import os.path
import requests
import csv
import time
import random

from tqdm import tqdm


def download(csv_path, emoteset):
    with open(csv_path, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",")
        for row in tqdm(reader):
            url = row.get("src")

            filename = row.get("name") + ".png"
            if emoteset == "twitch":
                new_url = url.replace("1.0", "3.0")
                path = os.path.join("images/global", filename)
            elif emoteset == "ffz":
                new_url = url[:-1] + "4"
                path = os.path.join("images/ffz", filename)
            else:
                new_url = url.replace("1x", "3x")
                path = os.path.join("images/bttv", filename)

            print(filename)
            try:
                img = requests.get(url)
            except requests.exceptions.RequestException as e:
                    raise SystemExit(e)

            with open(path, "wb") as file:
                file.write(img.content)
            
            pause = random.randint(2,6)
            time.sleep(pause)


if __name__ == '__main__':
    twitch_csv = "2021/global_emotes.csv"
    ffz_csv = "2021/ffz_emotes.csv"
    bttv_csv = "2021/bttv_global_emotes.csv"
    download(twitch_csv, "twitch")
    download(ffz_csv, "ffz")
    download(bttv_csv, "bttv")
