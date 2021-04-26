import argparse
import csv
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="read one cell")
    parser.add_argument("-f1", "--file1", type=str)
    parser.add_argument("-c1", "--column1", type=int)
    parser.add_argument("-f2", "--file2", type=str)
    parser.add_argument("-c2", "--column2", type=int)
    parser.add_argument("-u", "--up_to", type=int, default=50)

    args = parser.parse_args()

    words1 = []
    words2 = []

    with open(args.file1, "r") as f1:
        reader = csv.reader(f1, delimiter="\t")

        for i in range(args.up_to):
            words1.append(next(reader)[args.column1])

    with open(args.file2, "r") as f2:
        reader = csv.reader(f2, delimiter="\t")

        for i in range(args.up_to):
            words2.append(next(reader)[args.column2])
            
    print("# File 1: {}".format(args.file1))
    print("# File 2: {}".format(args.file2))

    intersection_set = set.intersection(set(words1), set(words2))

    print("# Compared top {} words".format(args.up_to))

    print("# Intersection length: {}".format(len(intersection_set)))

    print("# Intersection words: {}".format(", ".join(intersection_set)))

