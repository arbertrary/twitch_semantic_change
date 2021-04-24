import requests
from bs4 import BeautifulSoup
import os
import re
import csv


def get_bttv_globals():
    # url = "https://betterttv.com/emotes/global"
    # page = requests.get(url)
    # soup = BeautifulSoup(page.content, "html.parser")
    # print(soup.prettify())
    with open("ffz/bttv_global.html") as page:
        soup = BeautifulSoup(page.read(), "html.parser")
        print(soup.prettify())
        emotecards = soup.find("div", {"class": "EmoteCards_emoteCards__1lpxg"})
        emotes = emotecards.find_all("a")

        with open("bttv_global_emotes.csv", "w") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["name", "id", "src"])
            for emote in emotes:
                emote_name = emote.find("div").text
                # emote_src = emote.find("img")["src"]
                href = emote.get("href")
                emote_id = re.match(r".*emotes/(.*)", href).group(1)
                emote_src = "https://cdn.betterttv.net/emote/{}/1x".format(emote_id)

                csv_writer.writerow([emote_name, emote_id, emote_src])


def get_ffz_list():
    ffz_dir = "ffz"
    ffz_dict = {}
    for filename in sorted(os.listdir(ffz_dir)):
        # print(filename)
        if filename.endswith(".html"):
            with open(os.path.join(ffz_dir, filename)) as file:
                soup = BeautifulSoup(file.read(), "html.parser")
                emotes = soup.find_all("tr", {"class": "selectable"})
                for emote in emotes:
                    a = emote.find("td", {"class": "emote-name"}).find("a")
                    emote_name = a.text
                    emote_id = re.match(r".*/(\d*)-(.*)", a.get("href")).group(1)
                    emote_source = "https://cdn.frankerfacez.com/emoticon/{}/1".format(emote_id)
                    count = int(emote.find("td", {"class": None}).text.replace(",", ""))

                    emote_dict = {"id": emote_id, "count": count, "src": emote_source}

                    if ffz_dict.get(emote_name):
                        ffz_dict[emote_name]["count"] += count
                    else:
                        ffz_dict[emote_name] = emote_dict

    print(ffz_dict)

    with open("ffz_emotes.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["name", "id", "count", "src"])
        for emote in ffz_dict:
            emote_dict = ffz_dict[emote]
            writer.writerow([emote, emote_dict["id"], emote_dict["count"], emote_dict["src"]])


def get_twitch_global(html_path):
    with open(html_path) as file:
        soup = BeautifulSoup(file.read(), "html.parser")
        emotes = soup.find_all("a", {"class": "emote-name"})
        # < img
        # src = "https://static-cdn.jtvnw.net/emoticons/v1/191762/1.0"
        # data - tooltip = "<strong>Squid1</strong>"
        # data - regex = "Squid1"
        # data - toggle = "popover"
        # data - image - id = "191762"
        #
        # class ="emote expandable-emote" >
        with open("global_emotes.csv", "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["name", "id", "src"])

            for emote in emotes:
                img = emote.find("img")
                emote_source = img["src"]
                emote_name = img["data-regex"]
                emote_id = img["data-image-id"]
                # emote_dict = {"name": emote_name, "id": emote_id, "src": emote_source}

                writer.writerow([emote_name, emote_id, emote_source])

                print(emote_name, emote_id, emote_source)


if __name__ == '__main__':
    get_twitch_global("../testdata/twitch_global_emotes/Twitch Emotes - Bringing a little Kappa to you everyday.html")
    # get_ffz_list()
    # get_bttv_globals()
    # print("asd")
