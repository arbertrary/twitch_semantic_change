import requests
from requests_html import HTMLSession
import re
import csv
import time

BTTV_URL = "https://betterttv.com/emotes/global"
FFZ_URL = lambda p: "https://www.frankerfacez.com/emoticons/?page={}".format(p)
TWITCH_URL = "https://twitchemotes.com/"


def get_bttv_global():
    r = session.get(BTTV_URL)
    r.html.render()

    time.sleep(5)
    emotecards = r.html.find(".EmoteCards_emoteCards__1lpxg", first=True)
    emotes = emotecards.find("a")

    with open("bttv_global_emotes.csv", "w") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["name", "id", "src"])
        print("Writing to file \"bttv_global_emotes.csv\"")
        for emote in emotes:
            emote_name = emote.find("div", first=True).text
            href = emote.attrs.get("href")
            emote_id = re.match(r".*emotes/(.*)", href).group(1)
            emote_src = "https://cdn.betterttv.net/emote/{}/3x".format(emote_id)

            csv_writer.writerow([emote_name, emote_id, emote_src])


def get_ffz_new():
    ffz_dict = {}
    for page in range(1, 11):
        url = FFZ_URL(page)
        r = session.get(url)
        print("Sleeping for 2 seconds")
        time.sleep(2)

        emotes = r.html.find(".selectable")

        for emote in emotes:
            a = emote.find(".emote-name", first=True).find("a", first=True)
            emote_name = a.text
            emote_id = re.match(r".*/(\d*)-(.*)", a.attrs.get("href")).group(1)
            emote_source = "https://cdn.frankerfacez.com/emoticon/{}/4".format(emote_id)

            emote_dict = {"id": emote_id, "src": emote_source}

            ffz_dict[emote_name] = emote_dict

    with open("ffz_emotes.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["name", "id", "src"])
        print("Writing to file \"ffz_emotes.csv\"")

        for emote in ffz_dict:
            emote_dict = ffz_dict[emote]
            writer.writerow([emote, emote_dict["id"], emote_dict["src"]])


def get_twitch_global():
    r = session.get(TWITCH_URL)
    emotes = r.html.find(".emote-name")

    with open("twitch_global_emotes.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["name", "id", "src"])
        print("Writing to file \"twitch_global_emotes.csv\"")

        for emote in emotes:
            img = emote.find("img", first=True).attrs
            emote_source = img["src"]
            emote_source = emote_source.replace("1.0", "3.0")
            emote_name = img["data-regex"]
            emote_id = img["data-image-id"]

            writer.writerow([emote_name, emote_id, emote_source])


if __name__ == '__main__':
    print("# SCRIPT COULD POSSIBLY FAIL ON FIRST RUN\n"
          "Maybe something with the HTMLSession.\n"
          "Just run it again, it should succeed then.")
    session = HTMLSession()
    print("# Getting BTTV global emotes")
    get_bttv_global()
    print("# Getting original Twitch global emotes")
    get_twitch_global()
    print("# Getting top 500 FFZ emotes (contains duplicates. resulting list of emotes might be shorter)")
    get_ffz_new()

## Old versions with beautifulsoup and requests
# def get_twitch_global():
#     page = requests.get(TWITCH_URL)
#     soup = BeautifulSoup(page.content, "html.parser")
#     emotes = soup.find_all("a", {"class": "emote-name"})
#
#     with open("twitch_global_emotes.csv", "w") as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(["name", "id", "src"])
#         print("Writing to file \"twitch_global_emotes.csv\"")
#
#         for emote in emotes:
#             img = emote.find("img")
#             emote_source = img["src"]
#             emote_name = img["data-regex"]
#             emote_id = img["data-image-id"]
#
#             writer.writerow([emote_name, emote_id, emote_source])

# def get_ffz():
#     ffz_dict = {}
#     for page in range(1, 11):
#         url = FFZ_URL(page)
#         page = requests.get(url)
#         print("Sleeping for 2 seconds")
#         time.sleep(2)
#         soup = BeautifulSoup(page.content, "html.parser")
#         emotes = soup.find_all("tr", {"class": "selectable"})
#         for emote in emotes:
#             a = emote.find("td", {"class": "emote-name"}).find("a")
#             emote_name = a.text
#             emote_id = re.match(r".*/(\d*)-(.*)", a.get("href")).group(1)
#             emote_source = "https://cdn.frankerfacez.com/emoticon/{}/1".format(emote_id)
#
#             emote_dict = {"id": emote_id, "src": emote_source}
#
#             ffz_dict[emote_name] = emote_dict
#
#     with open("ffz_emotes.csv", "w") as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(["name", "id", "src"])
#         print("Writing to file \"ffz_emotes.csv\"")
#
#         for emote in ffz_dict:
#             emote_dict = ffz_dict[emote]
#             writer.writerow([emote, emote_dict["id"], emote_dict["src"]])
