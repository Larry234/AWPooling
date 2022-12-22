import pandas as pd
import glob
import os
import re
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=str, help='ray tune experiment dir')
    parser.add_argument('--file', default='result.csv', required=True, type=str, help='output csv location')

    args = parser.parse_args()

    file_path = os.path.join(args.data, '**', 'progress.csv')
    results = glob.glob(file_path)
    tems = []
    accs = []

    for result in results:
        f = pd.read_csv(result)

        t = result.split('/')[-2]
        tem = re.search(r't0=\S+t1=\S+t2=\S+t3=\S+t4=\S{5}', t).group()
        tems.append(tem)
        accs.append(max(f['accuracy']))

    df = pd.DataFrame({'temperature': tems, 'accuracy': accs})
    df = df.sort_values(by=['accuracy'], ascending=False)
    df.to_csv(os.path.join(args.data, args.file), sep=' ')