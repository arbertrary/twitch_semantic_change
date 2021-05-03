import requests
import argparse
import csv
from bs4 import BeautifulSoup
import time
from tqdm import tqdm

BASE_URL = lambda word: "http://jeseme.org/search?word={}&corpus=dta".format(word)


def find_jeseme(word, changepoint):
    url = BASE_URL(word)
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    rejection = soup.find(lambda tag: tag.name == "h2" and "not covered by" in tag.text)
    if not rejection:
        print("# JeSemE contains " + word + "\n\tChangepoint: " + changepoint)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str)
    parser.add_argument("--word_column", type=int, default=0)
    parser.add_argument("--changepoint_column", type=int, default=1)
    parser.add_argument("--up_to", type=int)
    options = parser.parse_args()

    with open(options.in_file, "r") as infile:
        reader = csv.reader(infile, delimiter="\t")
        for i in tqdm(range(options.up_to)):
            row = next(reader)
            word = row[options.word_column]
            changepoint = row[options.changepoint_column]
            time.sleep(2)
            find_jeseme(word, changepoint)
