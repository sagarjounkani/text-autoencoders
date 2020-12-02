from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split
import unicodedata
# from apUtils import generate_typo
import re
import string

all_letters = string.ascii_lowercase + "1234567890 "


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def prepareDataset(datapath, inputFile, ipCol, samplesPerAddress=100):
    # ipCol = "inputText"
    opCol = "outputText"
    outfileSep = ","
    outputFile = f'{inputFile.split(".")[0]}_out.csv'
    addressNames = pd.read_csv(f'{datapath}/{inputFile}', header=0)
    addressNames[ipCol] = addressNames[ipCol].apply(normalizeString)
    corruptNames = defaultdict(set)
    for address in addressNames[ipCol]:
        while len(corruptNames[address]) <= samplesPerAddress:
            corruptName = generate_typo(address)
            if corruptName != address:
                corruptNames[address].add(corruptName)

    fout = open(f'{datapath}{outputFile}', 'w')
    fout.write(f"{ipCol}{outfileSep}{opCol}\n")
    for k, v in corruptNames.items():
        for name in v:
            fout.write(f"{name}{outfileSep}{k}\n")
    fout.close()

    fullDataset = pd.read_csv(f'{datapath}{outputFile}')
    X_train, X_test_val, y_train, y_test_val = train_test_split(fullDataset[ipCol], fullDataset[opCol], test_size=0.4,
                                                                stratify=fullDataset[opCol], random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, stratify=y_test_val,
                                                    random_state=0)

    pd.concat([X_train, y_train], axis=1).reset_index(drop=True).to_csv(f'{datapath}train.csv', index=False)
    pd.concat([X_test, y_test], axis=1).reset_index(drop=True).to_csv(f'{datapath}test.csv', index=False)
    pd.concat([X_val, y_val], axis=1).reset_index(drop=True).to_csv(f'{datapath}val.csv', index=False)

    print("train/test/val datasets generated")


if __name__ == "__main__":
    prepareDataset('data', 'consolidatedData_v2.csv', 'address', samplesPerAddress=10)
