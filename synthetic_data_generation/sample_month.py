import argparse
import os
import pandas as pd
import multiprocessing
import random


def sample_file(filename, in_dir, out_dir, sp):
    out_filename = filename
    out_path = os.path.join(out_dir, out_filename)
    in_path = os.path.join(in_dir, filename)

    df = pd.read_csv(in_path, sep=",")
    #sample = df.sample(frac=sp)
    if len(df.index) >= 5000:
        sample = df.truncate(after=5000)
    else:
        sample = df
    # Added after first run with 201911; If unsuccessful with original texts try again with everything lowercased
    # sample["msg"] = sample["msg"].str.lower()

    sample.to_csv(out_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_rootdir", type=str)
    parser.add_argument("-o", "--output_rootdir", type=str)
    parser.add_argument("-m", "--month", type=str)
    parser.add_argument("-sp", "--sampling_percent", type=int, default=10)
    parser.add_argument("--multi", type=int, default=1)

    options = parser.parse_args()

    data_path = os.path.join(options.input_rootdir, options.month, "clean_en_sorted")


    dir_size = sum(os.path.getsize(os.path.join(data_path, f)) for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)))
    
    print(data_path)
    print(dir_size)
    if os.path.exists(data_path):
        print("IN DIR EXISTS")

    sampling_percent = float(1000000000 / dir_size)
    if sampling_percent > 1:
        raise ValueError("Somehow it's not a percentage?")
    
    print(sampling_percent)
    # sampling_percent = float(0.01*options.sampling_percent)
    out_dir = os.path.join(options.output_rootdir, options.month)
    os.makedirs(out_dir, exist_ok=True)

    pool = multiprocessing.Pool(4)
    filenames = random.sample(os.listdir(data_path), 5000)
    for file in filenames:
        if options.multi == 1:
            pool.apply_async(sample_file, (file, data_path, out_dir, sampling_percent))
        else:
            sample_file(file, data_path, out_dir, sampling_percent)
    
    try:
        pool.close()
        pool.join()
    except:
        pass
