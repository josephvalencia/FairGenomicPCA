import sys
import gzip
import allel

def parse_VCF(vcf_file):

    with gzip.open(vcf_file) as fh:
        data = allel.read_vcf(fh)

    return data

if __name__ == "__main__":

    vcf_file = sys.argv[1]
    vcf_data = parse_VCF(vcf_file)
