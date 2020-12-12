import sys
import gzip
import allel
import numpy as np

def parse_VCF(vcf_file):

    with gzip.open(vcf_file) as fh:
        data = allel.read_vcf(fh)

    return data

def load_1KG_genotype(mat_file):

    genotype = []

    with open(mat_file) as inFile:
        for l in inFile:
            snps = l.strip().split(',')
            snps = ['-9' if x == '' else x for x in snps]
            snp_array = np.asarray(snps,dtype=np.int)
            genotype.append(snp_array)

    return np.stack(genotype,axis=0)

def load_1KG_annotations(annot_file):

    inFile = open(annot_file)
    annotation = [l.strip() for l in inFile]
    inFile.close()
    return annotation

def keep_top_variance(train_data,top_snps):

    variance = train_data.var(axis=0)
    idx = np.argsort(-variance)
    train_data = train_data[:,idx]
    train_data = train_data[:,:top_snps]

    return train_data

def keep_random(train_data,top_snps):

    n,m = train_data.shape
    idx = np.random.choice(np.arange(m),size=top_snps)
    train_data = train_data[:,idx]
    return train_data


