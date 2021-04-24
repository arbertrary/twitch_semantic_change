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
    pseudo1 = len([w for w in words1 if "pseudoword" in w])
    pseudo2 = len([w for w in words2 if "pseudoword" in w])
    print("Total:\t{}\t{}".format(pseudo1, pseudo2))

    for scheme in ["C1", "C2", "C3", "D1", "D2", "D3", "D4"]:
        scheme_count1 = len([w for w in words1 if scheme in w])
        scheme_count2 = len([w for w in words2 if scheme in w])

        print("Scheme {}:\t{}\t{}".format(scheme, scheme_count1, scheme_count2))
    
    print("###")
