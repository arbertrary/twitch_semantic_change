import csv
import os

if __name__ == '__main__':
    dir1 = "sg0"
    dir2 = "sg1"

    intersect = []
    cp_intersect = []
    for file in os.listdir(dir1):
        if file.endswith(".tsv"):
            # print(file)
            temp = []
            temp_cp = []
            with open(os.path.join(dir1, file), "r") as csvfile:
                reader = csv.reader(csvfile, delimiter="\t")
                for i, row in enumerate(reader):
                    if i >= 50:
                        continue
                    temp.append(row[0])
                    temp_cp.append((row[0], row[1]))
            intersect.append(temp)
            cp_intersect.append(temp_cp)

    for file in os.listdir(dir2):
        if file.endswith(".tsv"):
            # print(file)
            temp = []
            temp_cp = []
            with open(os.path.join(dir2, file), "r") as csvfile:
                reader = csv.reader(csvfile, delimiter="\t")
                for i, row in enumerate(reader):
                    if i >= 50:
                       continue
                    temp.append(row[0])
                    temp_cp.append((row[0], row[1]))
            intersect.append(temp)
            cp_intersect.append(temp_cp)
    print("\n##  IN ALL FOUR")
    print(set.intersection(set(intersect[0]),set(intersect[1]), set(intersect[2]), set(intersect[3]))) 
    print(set.intersection(set(cp_intersect[0]),set(cp_intersect[1]), set(cp_intersect[2]), set(cp_intersect[3]))) 
    print("\n## COMPARE TO FIRST")
    print(set.intersection(set(intersect[0]),set(intersect[2]))) 
    print(set.intersection(set(cp_intersect[0]),set(cp_intersect[2]))) 

    print("\n## COMPARE TO LAST")
    print(set.intersection(set(intersect[1]),set(intersect[3]))) 
    print(set.intersection(set(cp_intersect[1]),set(cp_intersect[3]))) 
    
    print("\n## CBOW")
    print(set.intersection(set(intersect[0]),set(intersect[1]))) 
    print(set.intersection(set(cp_intersect[0]),set(cp_intersect[1]))) 

    print("\n## SGNS")
    print(set.intersection(set(intersect[2]),set(intersect[3]))) 
    print(set.intersection(set(cp_intersect[2]),set(cp_intersect[3]))) 
