import argparse
import os
import csv


def read_ap_at_50(filepath):
    with open(filepath, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            if row["k_retrieved"] == "50":
                print("AP @ 50: {:0.3f}".format(float(row["average_precision"])))
                print("Total # of pseudowords @ 50: {}".format(50 - int(row["n_non_pseudowords"])))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="read one cell")
    parser.add_argument("-in", "--in_file")

    args = parser.parse_args()
    in_path = args.in_file

    if os.path.isfile(in_path):
        read_ap_at_50(in_path)
    elif os.path.isdir(in_path):
        for file in sorted(os.listdir(in_path)):
            print("\n")
            print("# " +file)
            in_file = os.path.join(in_path, file)
            read_ap_at_50(in_file)

