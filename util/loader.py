from csv import DictReader
import os


class ChatYielder(object):
    def __init__(self, filepaths):
        self.filepaths = filepaths

    def __iter__(self):
        for filepath in self.filepaths:
            with open(filepath, "r") as file:
                reader = DictReader(file, delimiter=",")
                for row in reader:
                    yield row["msg"]


class EmoteYielder(object):
    def __init__(self, filepaths):
        self.filepaths = filepaths

    def __iter__(self):
        for filepath in self.filepaths:
            with open(filepath, "r") as file:
                reader = DictReader(file, delimiter=",")
                for row in reader:
                    emotes = [x[0] for x in row["emotes"].split("/") if x] + [x[0] for x in row["extemotes"].split("/")
                                                                              if x]
                    yield emotes
