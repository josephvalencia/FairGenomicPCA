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

if __name__ == "__main__":

    mat_file = sys.argv[1]
    id_file = sys.argv[2]
    pop_file = sys.argv[3]

    genotype = load_1KG_genotype(mat_file)
    ids = load_1KG_annotations(id_file)
    pop = load_1KG_annotations(pop_file)

