import sys
import gzip
import allel

def parse_VCF(vcf_file):

    with gzip.open(vcf_file) as fh:
        data = allel.read_vcf(fh)

    return data

