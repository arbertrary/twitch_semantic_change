#!/usr/bin/env python3

import os, sys
import gzip, zipfile
import argparse
import time
import logging
import multiprocessing
import operator
import csv
import json
import re
import datetime


# work_dir = os.path.dirname(__file__)  # takes parent folder of executed .py file as work dir
# os.chdir(os.path.dirname(__file__))  # set working directory
# sys.path.append(work_dir)  # look for modules here
# print("working dir:", os.getcwd())
# print("working dir list:", os.listdir(os.getcwd()))
# print("user id:", os.getegid())


def external_emotes(ffz: str, bttv: str) -> dict[str, str]:
    """

    :return: The dict containing global BTTV emotes and the top ~600 FFZ emotes
    """
    #    ffz = "../emotes/ffz_emotes.csv"
    #   bttv = "../emotes/bttv_global_emotes.csv"

    external_emotes_dict = {}
    with open(ffz) as ffz_emotes:
        reader = csv.reader(ffz_emotes, delimiter=",")
        next(reader, None)
        for row in reader:
            external_emotes_dict[row[0]] = "ffz" + row[1]

    with open(bttv) as bttv_emotes:
        reader = csv.reader(bttv_emotes, delimiter=",")
        next(reader, None)
        for row in reader:
            external_emotes_dict[row[0]] = "bttv" + row[1]

    return external_emotes_dict


def external_emote_ranges(msg_text: str, external_emotes_dict: dict[str, str]) -> str:
    """
     Similar to the uemo field,
     detect FFZ and BTTV emotes in the given message and their location
    """

    msg = msg_text.split()
    index = 0
    range_dict = {}
    for word in msg:
        if external_emotes_dict.get(word):
            emote_range = str(index) + "-" + str(index + len(word)-1)
            emote_id = external_emotes_dict.get(word)
            if range_dict.get(emote_id):
                range_dict[emote_id] += "," + emote_range
            else:
                range_dict[emote_id] = emote_range

        index = index + len(word) + 1

    emote_ranges = str.join("/", list(map(lambda k: k + ":" + range_dict[k], range_dict.keys())))
    return emote_ranges


def clean_old_chatlogs(infilepath: str):
    filename = re.match(r".*_(\d{12}.txt).*", infilepath).group(1)

    outfilepath = os.path.join(out_root, filename)
    try:
        if infilepath.endswith(".gz"):
            infile = gzip.open(infilepath, "rt")
        elif infilepath.endswith(".zip"):
            infile = zipfile.ZipFile(infilepath).open(os.path.basename(infilepath).replace(".zip", ""), "r")
        else:
            infile = open(infilepath, "rt")

        outstrings = []
        for line in infile.readlines():
            if infilepath.endswith(".zip"):
                msg_data = json.loads(line.decode("utf-8"))
            else:
                msg_data = json.loads(line)
            lng = msg_data.get("stream").get("language")
            if lng != "en":
                continue

            viewercount = int(msg_data.get("stream").get("viewer_count"))

            timestamp = msg_data.get("timestamp")
            roomstate = msg_data.get("roomstate")
            emote_only = roomstate.get("emote-only")
            r9k = roomstate.get("r9k")
            chid = roomstate.get("room-id")
            channelname = roomstate.get("channel").removeprefix("#")

            # if chid in channels_dict:
            #     # channels_dict.get(chid).get("vcnt").append(viewercount)
            #     channels_dict[chid]["vcnt_summed"] += viewercount
            #     channels_dict[chid]["n_msgs"] += 1
            # else:
            #     channel = {"ch": channelname, "vcnt_summed": viewercount, "n_msgs": 1}
            #     channels_dict[chid] = channel

            userstate = msg_data.get("userstate")
            usid = userstate.get("user-id")
            # username = userstate.get("username")
            uemo = userstate.get("emotes-raw")
            sub = userstate.get("subscriber")
            mod = userstate.get("mod")

            # users_dict[usid] = username

            msg = msg_data.get("message")

            game = msg_data.get("game").get("name")

            # TODO: clean message string here
            # msg_text = clean_message(msg_text)
            ext_emotes = external_emote_ranges(msg, emotes_dict)
            outList = [timestamp, chid, msg, uemo, ext_emotes, game, usid, sub, mod, emote_only, r9k]
            outstrings.append(outList)

        infile.close()

        with open(outfilepath, "w", encoding="utf-8") as outfile:
            csv_writer = csv.writer(outfile)
            csv_writer.writerow(
                ["ts", "chid", "msg", "emotes", "extemotes", "game", "usid", "sub", "mod", "emonly", "r9k"])
            outstrings.sort(key=operator.itemgetter(1, 0))
            for msg in outstrings:
                csv_writer.writerow(msg)

    except json.decoder.JSONDecodeError as e:
        print(filename)
        logging.error("# JSON ERROR AT: " + filename)
        logging.error(e)
    except UnicodeDecodeError as e:
        print(filename)
        logging.error("# UNICODEDECODE ERROR AT: " + filename)
        logging.error(e)
    except zipfile.BadZipFile as e:
        print(filename)
        logging.error("# zipfile.BadZipFile ERROR AT: " + filename)
        logging.error(e)
    except KeyError as k:
        print(filename)
        logging.error("# KeyError No item named ERROR AT: " + filename)
        logging.error(k)
    logging.info(filename)


def clean_djinn4_chatlogs(infilepath: str):
    filename = re.match(r".*_(\d{12}.txt)", os.path.basename(infilepath)).group(1)

    outfilepath = os.path.join(out_root, "H2_" + filename)
    try:
        with open(infilepath, "r") as infile:
            outstrings = []
            for line in infile.readlines():
                msg_data = json.loads(line)
                # lng = msg_data.get("lng")
                # if lng != "en":
                #    continue

                timestamp = msg_data.get("ts")
                emote_only = msg_data.get("emo")
                r9k = msg_data.get("r9k")
                chid = msg_data.get("chid")

                usid = msg_data.get("usid")
                uemo = msg_data.get("uemo")
                sub = msg_data.get("sub")
                mod = msg_data.get("mod")

                msg = msg_data.get("msg")

                game = msg_data.get("game")

                ext_emotes = external_emote_ranges(msg, emotes_dict)

                outList = [timestamp, chid, msg, uemo, ext_emotes, game, usid, sub, mod, emote_only, r9k]
                outstrings.append(outList)

            infile.close()

            with open(outfilepath, "w") as outfile:

                csv_writer = csv.writer(outfile)
                csv_writer.writerow(
                    ["ts", "chid", "msg", "emotes", "extemotes", "game", "usid", "sub", "mod", "emonly", "r9k"])
                outstrings.sort(key=operator.itemgetter(1, 0))
                for msg in outstrings:
                    csv_writer.writerow(msg)
    except json.decoder.JSONDecodeError as e:
        print(filename)
        logging.error("# JSON ERROR AT: " + filename)
        logging.error(e)
    except UnicodeDecodeError as e:
        print(filename)
        logging.error("# UNICODEDECODE ERROR AT: " + filename)
        logging.error(e)
    except zipfile.BadZipFile as e:
        print(filename)
        logging.error("# zipfile.BadZipFile ERROR AT: " + filename)
        logging.error(e)
    except KeyError as k:
        print(filename)
        logging.error("# KeyError No item named ERROR AT: " + filename)
        logging.error(k)
    logging.info(filename)


def clean_message(message: str):
    """
    shorten repetitions of letters etc.
    See the cleaning for the twitch paper

    :return:
    """
    # TODO: clean the text
    # lowercase 
    print("TODO")


if __name__ == '__main__':
    # us_dict = {}
    # ch_dict = {}
    # emotes_dict = external_emotes("../emotes/ffz_emotes.csv", "../emotes/bttv_global_emotes.csv")
    # print("Started at {}\n".format(datetime.datetime.now()))

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infiles_rootdir", type=str,
                        help="path to a chatlog directory; Under this directory only files should be located")
    parser.add_argument("-o", "--outfiles_dir", type=str)
    parser.add_argument("-f", "--ffz", type=str, help="path to the ffz emotes csv")
    parser.add_argument("-b", "--bttv", type=str, help="path to the bttv emotes csv")
    parser.add_argument("-mp", "--multi", type=int, default=0, help="whether to use multiprocessing")
    options = parser.parse_args()
    log_path = os.path.abspath(os.path.join(options.infiles_rootdir, os.pardir, "prep_fixed.log"))
    print(log_path)
    logging.basicConfig(filename=log_path, encoding="utf-8", level=logging.DEBUG, format="%(message)s")
    logging.info("# Started at {}\n".format(datetime.datetime.now()))
    start_time = time.time()

    us_dict = {}
    ch_dict = {}

    ffz_path = os.path.abspath(options.ffz)
    bttv_path = os.path.abspath(options.bttv)
    in_root = os.path.abspath(options.infiles_rootdir)
    out_root = os.path.abspath(options.outfiles_dir)

    print(ffz_path)
    print(bttv_path)
    print(in_root)
    print(out_root)

    # emotes_dict = external_emotes("/home/stud/bernstetter/ma/initial/emotes/ffz_emotes.csv",
    #                               "/home/stud/bernstetter/ma/initial/emotes/bttv_global_emotes.csv")
    emotes_dict = external_emotes(options.ffz, options.bttv)

    pool = multiprocessing.Pool(10)
    filelist = [os.path.join(in_root, file) for file in os.listdir(in_root)]
    if not os.path.exists(out_root):
        os.mkdir(out_root)

    if options.multi == 1:
        # pool.map(clean_djinn4_chatlogs, filelist)
        pool.map(clean_old_chatlogs, filelist)
    else:
        for file in filelist:
            clean_old_chatlogs(file)
            # clean_djinn4_chatlogs(file)

    # for file in os.listdir(in_root):
    #     chatlog_path = os.path.join(in_root, file)
    #
    #     if not os.path.exists(out_root):
    #         os.mkdir(out_root)
    #
    #     with open(log_path, "r") as log:
    #         last = log.read().splitlines()
    #         if file in last:
    #             logging.info("Skipped " + file)
    #             continue
    #     try:
    #         if options.multi == 1:
    #             pool.apply_async(clean_djinn4_chatlogs, (chatlog_path, out_root, emotes_dict, ch_dict, us_dict))
    #         else:
    #             clean_djinn4_chatlogs(chatlog_path, out_root, emotes_dict, ch_dict, us_dict)
    #     except json.decoder.JSONDecodeError as e:
    #         print(chatlog_path)
    #         logging.error("# JSON ERROR AT: " + chatlog_path)
    #         logging.error(e)
    #     except UnicodeDecodeError as e:
    #         print(chatlog_path)
    #         logging.error("# UNICODEDECODE ERROR AT: " + chatlog_path)
    #         logging.error(e)
    #     except zipfile.BadZipFile as e:
    #         print(chatlog_path)
    #         logging.error("# zipfile.BadZipFile ERROR AT: " + chatlog_path)
    #         logging.error(e)
    #     except KeyError as k:
    #         print(chatlog_path)
    #         logging.error("# KeyError No item named ERROR AT: " + chatlog_path)
    #         logging.error(k)

    # if month in months_old:
    #     for file in os.listdir(data_path):
    #         clean_old_chatlogs(os.path.join(data_path, file), out_root, emotes_dict, ch_dict, us_dict)
    # else:
    #     for file in os.listdir(data_path):
    #         clean_chatlogs(os.path.join(data_path, file), out_root, emotes_dict, ch_dict, us_dict)
    try:
        pool.close()
        pool.join()
    except:
        pass
    # with open(os.path.abspath(os.path.join(options.infiles_rootdir, os.pardir, "users.json")), "w") as users:
    #    json.dump(us_dict, users)
    # with open(os.path.abspath(os.path.join(options.infiles_rootdir, os.pardir, "channels_vcnt_mp.json")),
    #           "w") as channels:
    #     json.dump(ch_dict, channels)

    print("--- %s seconds ---" % (time.time() - start_time))
