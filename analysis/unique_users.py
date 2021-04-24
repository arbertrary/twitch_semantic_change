import os
import logging
import argparse
import gzip
import zipfile
import json
import time, datetime


def unique_users(infilepath: str, users_dict):
    if infilepath.endswith(".gz"):
        infile = gzip.open(infilepath, "rt")
    elif infilepath.endswith(".zip"):
        infile = zipfile.ZipFile(infilepath).open(os.path.basename(infilepath).replace(".zip", ""), "r")
    else:
        infile = open(infilepath, "rt")

    for line in infile.readlines():
        if infilepath.endswith(".zip"):
            msg_data = json.loads(line.decode("utf-8"))
        else:
            msg_data = json.loads(line)
        lng = msg_data.get("stream").get("language")
        if lng != "en" and lng != "unknown":
            continue

        userstate = msg_data.get("userstate")
        usid = userstate.get("user-id")
        username = userstate.get("username")
        users_dict[usid] = username

    infile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infiles_rootdir", type=str,
                        help="path to a chatlog directory; Under this directory only files should be located")
    options = parser.parse_args()
    log_path = os.path.abspath(os.path.join(options.infiles_rootdir, os.pardir, "unique_users.log"))
    print(log_path)

    logging.basicConfig(filename=log_path, encoding="utf-8", level=logging.DEBUG, format="%(message)s")
    logging.info("# Started at {}\n".format(datetime.datetime.now()))
    start_time = time.time()

    us_dict = {}

    in_root = os.path.abspath(options.infiles_rootdir)

    print(in_root)

    i = 0
    for file in os.listdir(in_root):
        chatlog_path = os.path.join(in_root, file)
        i+=1

        if i % 100 == 0:
            with open(os.path.abspath(os.path.join(options.infiles_rootdir, os.pardir, "users.json")), "w") as users:
                json.dump(us_dict, users)

        with open(log_path, "r") as log:
            last = log.read().splitlines()
            if file in last:
                logging.info("Skipped " + file)
                continue
        try:
            unique_users(chatlog_path, us_dict)
            logging.info(file)
        except json.decoder.JSONDecodeError as e:
            print(chatlog_path)
            logging.error("# JSON ERROR AT: " + chatlog_path)
            logging.error(e)
        except UnicodeDecodeError as e:
            print(chatlog_path)
            logging.error("# UNICODEDECODE ERROR AT: " + chatlog_path)
            logging.error(e)
        except zipfile.BadZipFile as e:
            print(chatlog_path)
            logging.error("# zipfile.BadZipFile ERROR AT: " + chatlog_path)
            logging.error(e)
        except KeyError as k:
            print(chatlog_path)
            logging.error("# KeyError No item named ERROR AT: " + chatlog_path)
            logging.error(k)

    with open(os.path.abspath(os.path.join(options.infiles_rootdir, os.pardir, "users.json")), "w") as users:
        json.dump(us_dict, users)

    print("--- %s seconds ---" % (time.time() - start_time))
