# Description

- A dataset containing 12 months of Twitch chat message logs
- Spanning from May 2019 to April 2020 (inclusive)
- 23-31 days per month
- generally 288 files per day 
- 585 Gigabyte, 5.5 Billion messages
- Chatlogs from 556,679 channels
- 

# Format

CSV text files, each including 5 minutes of chat log data (various channels)

```
"ts", "msg", "emotes", "extemotes", "game", "chid", "usid", "sub", "mod", "emonly", "r9k"

``` 

- `ts`: the timestamp
- `msg`: message text
- `emotes`: id and indices of twitch emotes in message
- `extemotes`: id and indices of external (ffz and bttv) emotes
- `game`: which game was played
- `chid`: channel id
- `usid`: user id
- `sub`: is the user subscribed to the channel
- `mod`: is the user mod to the channel
- `emonly`: is the chat in emote-only mode
- `r9k`: is the chat in r9k mode (If enabled, messages with more than 9 characters must be unique. R9K mode makes it so the same line of chat (10 or more characters) can't be repeated within 10 minutes.)

# Location

vingilot

/home/stud/bernstetter/datasets/twitch

# Total size:

- 49G	201905/clean
- 62G	201906/clean
- 53G	201907/clean
- 21G	201908/clean
- 41G	201909/clean
- 11G	201910/clean
- 26G	201911/clean
- 66G	201912/clean
- 74G	202001/clean
- 58G	202002/clean
- 67G	202003/clean
- 57G	202004/clean
- **585G total**

# unique users

28,987,589

# Message counts

## 201905:

- Recorded days: 25
- 201905: 455,837,118
- Msgs per day: 18 Mio
- Channels: 150,620

## 201906:

- Recorded days: 30
- 201906: 576,633,342
- Msgs per day: 19 Mio
- Channels: 161,946

## 201907:

- Recorded days: 31
- 201907: 495,832,931
- Msgs per day: 16 Mio
- Channels: 177,663

## 201908:

- Recorded days: 26
- 201908: 195,981,733
- Msgs per day: 7.5Mio
- Channels: 140,509

## 201909:

- Recorded days: 25
- 201909: 378,937,289
- Msgs per day: 15Mio
- Channels: 88,686

## 201910:

- Recorded days: 23
- 201910: 97,572,107
- Msgs per day: 4.2Mio
- Channels: 52,261

## 201911:

- Recorded days: 28
- 201911: 242,466,896
- Msgs per day: 8.7Mio
- Channels: 127,974

## 201912:

- Recorded days: 31
- 201912: 619,438,751
- Msgs per day: 20Mio
- Channels: 154,648

## 202001:

- Recorded days: 31
- 202001: 692,604,499
- Msgs per day: 22Mio
- Channels: 149,706

## 202002:
- Recorded days: 29
- 202002: 545,737,501
- Msgs per day: 19Mio
- Channels: 134,886


## 202003:

- Recorded days: 31
- 202003: 626,926,340
- Msgs per day: 20Mio
- Channels: 143,944

## 202004:

- Recorded days: 23
- 202004: 534,629,116
- Msgs per day: 23Mio
- Channels: 113,257

## Total

**ca. 5,462,000,000 => ca 5.5 Billion messages**


# ll output

## 201905
49G	201905/clean
-rw-r--r-- 1 studbernstetter stud 10488375 Jan 14 20:22 channels.json
drwxr-sr-x 2 studbernstetter stud     7078 Jan 14 20:22 clean
-rw-r--r-- 1 studbernstetter stud   247771 Jan 14 20:22 prep_run2.log
drwxr-sr-x 2 studbernstetter stud     7078 Jan 13 13:36 raw

## 201906
62G	201906/clean
-rw-r--r-- 1 studbernstetter stud 11286028 Jan 14 21:57 channels.json
drwxr-sr-x 2 studbernstetter stud     8635 Jan 14 21:57 clean
-rw-r--r-- 1 studbernstetter stud   302266 Jan 14 21:57 prep_run2.log
drwxr-sr-x 2 studbernstetter stud     8635 Jan 12 13:16 raw

## 201907
53G	201907/clean
-rw-r--r-- 1 studbernstetter stud 12343088 Jan 14 22:10 channels.json
drwxr-sr-x 2 studbernstetter stud     8848 Jan 14 22:10 clean
-rw-r--r-- 1 studbernstetter stud   309721 Jan 14 22:10 prep_run2.log
drwxr-sr-x 2 studbernstetter stud     8848 Jan 12 13:20 raw

## 201908
21G	201908/clean
-rw-r--r-- 1 studbernstetter stud 9745546 Jan 14 16:53 channels.json
drwxr-sr-x 2 studbernstetter stud    7319 Jan 14 16:53 clean
-rw-r--r-- 1 studbernstetter stud  268445 Jan 14 16:53 prep_run2.log
drwxr-sr-x 2 studbernstetter stud    7404 Jan 12 13:20 raw

## 201909
41G	201909/clean
171G	201909/raw
-rw-r--r-- 1 studbernstetter stud 6171056 Jan 14 20:30 channels.json
drwxr-sr-x 2 studbernstetter stud    7030 Jan 14 20:30 clean
-rw-r--r-- 1 studbernstetter stud  246091 Jan 14 20:30 prep_run2.log
drwxr-sr-x 2 studbernstetter stud    7030 Jan 12 13:27 raw

## 201910
11G	201910/clean
169G	201910/raw
179G	201910
total 3926
-rw-r--r-- 1 studbernstetter stud 3611388 Jan 14 16:58 channels.json
drwxr-sr-x 2 studbernstetter stud    6584 Jan 14 17:24 clean
-rw-r--r-- 1 studbernstetter stud  204145 Jan 14 16:58 prep_run2.log
drwxr-sr-x 2 studbernstetter stud    6584 Jan 12 13:23 raw

## 201911
26G	201911/clean
-rw-r--r-- 1 studbernstetter stud 8874237 Jan 14 18:19 channels.json
drwxr-sr-x 2 studbernstetter stud    7870 Jan 14 18:19 clean
-rw-r--r-- 1 studbernstetter stud  275630 Jan 14 18:19 prep_run2.log
drwxr-sr-x 2 studbernstetter stud    7871 Jan 12 13:20 raw

## 201912
66G	201912/clean
-rw-r--r-- 1 studbernstetter stud 10765911 Jan 15 00:08 channels.json
drwxr-sr-x 2 studbernstetter stud     8898 Jan 15 00:08 clean
-rw-r--r-- 1 studbernstetter stud   311888 Jan 15 00:08 prep_run2.log
drwxr-sr-x 2 studbernstetter stud     8901 Jan 12 13:20 raw

## 202001
74G	202001/clean
-rw-r--r-- 1 studbernstetter stud 10461173 Jan 15 00:58 channels.json
drwxr-sr-x 2 studbernstetter stud     8907 Jan 15 00:58 clean
-rw-r--r-- 1 studbernstetter stud   312203 Jan 15 00:58 prep_run2.log
drwxr-sr-x 2 studbernstetter stud     8910 Jan 12 13:20 raw

## 202002
58G	202002/clean
4.1M	202002/raw
58G	202002
total 9830
-rw-r--r-- 1 studbernstetter stud 9388677 Jan 14 22:47 channels.json
drwxr-sr-x 2 studbernstetter stud    8333 Jan 14 22:47 clean
-rw-r--r-- 1 studbernstetter stud  292113 Jan 14 22:47 prep_run2.log
drwxr-sr-x 2 studbernstetter stud    8336 Jan 12 13:20 raw

## 202003
67G	202003/clean
-rw-r--r-- 1 studbernstetter stud 10057261 Jan 15 00:05 channels.json
drwxr-sr-x 2 studbernstetter stud     8827 Jan 15 00:04 clean
-rw-r--r-- 1 studbernstetter stud   311766 Jan 15 00:04 prep_run2.log
drwxr-sr-x 2 studbernstetter stud     8847 Jan 12 13:20 raw

## 202004
57G	202004/clean
239G	202004/raw
-rw-r--r-- 1 studbernstetter stud 7920571 Jan 15 00:02 channels.json
drwxr-sr-x 2 studbernstetter stud    6399 Jan 15 00:02 clean
-rw-r--r-- 1 studbernstetter stud  218036 Jan 15 00:02 prep_run2.log
drwxr-sr-x 2 studbernstetter stud    6402 Jan 12 14:15 raw

