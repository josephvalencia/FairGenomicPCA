import pandas as pd
from src.data_utils import load_1KG_annotations

df = pd.read_csv("data/1000Genomes/id_codes.txt",sep="\t")
df = df.set_index('population_description')

pop_labels = load_1KG_annotations("data/1000Genomes/recoded_1000G.raw.noadmixed.lbls3")

a = df.loc[pop_labels]['super_population_code'].values

with open("data/1000Genomes/recoded_1000G.raw.noadmixed.lbls_super",'w') as outFile:
    for val in a:
        outFile.write(val+"\n")
