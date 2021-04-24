import argparse
import os
import csv

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="read one cell")
        parser.add_argument("-in", "--in_file")

        args = parser.parse_args()
        filepath = args.in_file

        with open(filepath, "r") as csvfile:
            reader = csv.DictReader(csvfile, delimiter="\t")
            for row in reader:
                if row["k_retrieved"] == "50":
                    print(row["average_precision"])

