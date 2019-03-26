import os
from pprint import pprint

i = 0
experiments = ""
for dirname, subdirs, files in os.walk("./"):
    if 'RNN' in dirname and len(subdirs) == 0:
        experiment = ""
        stats = {}
        with open(os.path.join(dirname, "exp_config.txt")) as fp:
            lines = fp.readlines()
            for line in lines:
                tokens = line.split()
                if tokens[0] == "emb_size":
                    stats["emb_size"] = tokens[1]
                elif tokens[0] == "optimizer":
                    stats["optimizer"] = tokens[1]
                elif tokens[0] == "initial_lr":
                    stats["initial_lr"] = tokens[1]
                elif tokens[0] == "dp_keep_prob":
                    stats["dp_keep_prob"] = tokens[1]

        print(stats)

        experiment += "&" + stats["optimizer"]
        experiment += "&" + stats["emb_size"]
        experiment += "&" + stats["initial_lr"]
        experiment += "&" + stats["dp_keep_prob"]

        with open(os.path.join(dirname, "log.txt")) as fp:
            lines = fp.readlines()
            print(lines[-1])

print(i)
