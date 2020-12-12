import pandas as pd
import numpy as np
import seaborn as sns
import sys
from src.data_utils import load_1KG_genotype, load_1KG_annotations, keep_top_variance
from sklearn import preprocessing
import matplotlib.pyplot as plt


def plot_PC_scatter(savefile,title,data,PC):

    pc_top = PC[:,:2]
    projected = data @ pc_top

    fig = plt.figure(figsize=(8,8))
    ax = plt.axes()

    a,b = np.split(projected,2,1)
    a = a.ravel()
    b = b.ravel()

    sns.scatterplot(x=a,y=b,hue=train_labels,ax=ax,alpha=0.6)
    plt.ylabel('PC 2')
    plt.xlabel('PC 1')
    plt.title(title)
    plt.savefig(savefile)
    plt.close()

if __name__ == "__main__":
   
    sns.set_style("darkgrid")

    # plot genome PCs
    train_data = load_1KG_genotype("data/1000genomes/recoded_1000g.noadmixed.mat")
    train_ids = load_1KG_annotations("data/1000genomes/recoded_1000g.raw.noadmixed.ids")
    train_labels = load_1KG_annotations("data/1000genomes/recoded_1000g.raw.noadmixed.lbls_super")

    le = preprocessing.LabelEncoder()
    racial_labels = le.fit_transform(train_labels).ravel()

    NUM_TOP_SNP = 1000
    genome_data = keep_top_variance(train_data,NUM_TOP_SNP)
    train_scaled = preprocessing.scale(genome_data)
    
    fw = np.load("1000G_FW_1000_22.npz",allow_pickle=True)['PC']
    vanilla = np.load("1000G_PCA_1000_22.npz",allow_pickle=True)['PC']

    title = "1000 Genomes Frank-White-Nash Top 1K SNPs"
    savefile = "FW_scatter_1K.png"
    plot_PC_scatter(savefile,title,train_scaled,fw)    

    title = "1000 Genomes Vanilla PCA Top 1K SNPs"
    savefile = "PCA_scatter_1K.png"
    plot_PC_scatter(savefile,title,train_scaled,vanilla)    

    # plot 1000G progress 

    fw_results = pd.read_csv("1000G_FW_1000_22.csv",sep="\t")
    fw_results['model'] = ["Frank-Wolfe-Nash" for x in range(len(fw_results))]
    
    pca_results = pd.read_csv("1000G_PCA_1000_22.csv",sep="\t")
    pca_results['model'] = ["Vanilla_PCA" for x in range(len(pca_results))]
    
    all_genome_results = pd.concat([fw_results,pca_results])

    plt.title("1000 Genomes Reconstruction Loss vs # PCs")
    plt.ylabel("Loss")
    plt.xlabel("# Principal Components")
    sns.lineplot(data=all_genome_results,x="n_components",y="reconstruction_loss",hue="model",style="model", markers=["o", "o"])
    plt.savefig("1000G_loss_1K.png")
    plt.close()

    plt.title("1000 Genomes Nash Social Welfare vs # PCs")
    plt.ylabel("NSW")
    plt.xlabel("# Principal Components")
    sns.lineplot(data=all_genome_results,x="n_components",y="nash_social_welfare",hue="model",style="model", markers=["o", "o"])
    plt.savefig("1000G_Nash_1K.png")
    plt.close()

    NUM_TOP_SNP = 5000
    genome_data = keep_top_variance(train_data,NUM_TOP_SNP)
    train_scaled = preprocessing.scale(genome_data)
    
    fw = np.load("1000G_FW_5000_22.npz",allow_pickle=True)['PC']
    vanilla = np.load("1000G_PCA_5000_22.npz",allow_pickle=True)['PC']

    title = "1000 Genomes Frank-White-Nash Top 5K SNPs"
    savefile = "FW_scatter_5K.png"
    plot_PC_scatter(savefile,title,train_scaled,fw)    

    title = "1000 Genomes Vanilla PCA Top 5K SNPs"
    savefile = "PCA_scatter_5K.png"
    plot_PC_scatter(savefile,title,train_scaled,vanilla)    

    # plot 1000G progress 

    fw_results = pd.read_csv("1000G_FW_5000_22.csv",sep="\t")
    fw_results['model'] = ["Frank-Wolfe-Nash" for x in range(len(fw_results))]
    
    pca_results = pd.read_csv("1000G_PCA_5000_22.csv",sep="\t")
    pca_results['model'] = ["Vanilla_PCA" for x in range(len(pca_results))]
    
    all_genome_results = pd.concat([fw_results,pca_results])

    plt.title("1000 Genomes Reconstruction Loss vs # PCs")
    plt.ylabel("Loss")
    plt.xlabel("# Principal Components")
    sns.lineplot(data=all_genome_results,x="n_components",y="reconstruction_loss",hue="model",style="model", markers=["o", "o"])
    plt.savefig("1000G_loss_5K.png")
    plt.close()

    plt.title("1000 Genomes Nash Social Welfare vs # PCs")
    plt.ylabel("NSW")
    plt.xlabel("# Principal Components")
    sns.lineplot(data=all_genome_results,x="n_components",y="nash_social_welfare",hue="model",style="model", markers=["o", "o"])
    plt.savefig("1000G_Nash_5K.png")
    plt.close()

    # plot Adult progress  
    fw_results = pd.read_csv("ADULT_FW_22.csv",sep="\t")
    fw_results['model'] = ["Frank-Wolfe-Nash" for x in range(len(fw_results))]
    
    pca_results = pd.read_csv("ADULT_PCA_22.csv",sep="\t")
    pca_results['model'] = ["Vanilla_PCA" for x in range(len(pca_results))]
    
    all_adult_results = pd.concat([fw_results,pca_results])

    plt.title("Adult Reconstruction Loss vs # PCs")
    plt.ylabel("Loss")
    plt.xlabel("# Principal Components")
    sns.lineplot(data=all_adult_results,x="n_components",y="reconstruction_loss",hue="model",style="model", markers=["o", "o"])
    plt.savefig("ADULT_loss.png")
    plt.close()

    plt.title("Adult Nash Social Welfare vs # PCs")
    plt.ylabel("NSW")
    plt.xlabel("# Principal Components")
    sns.lineplot(data=all_adult_results,x="n_components",y="nash_social_welfare",hue="model",style="model", markers=["o", "o"])
    plt.savefig("ADULT_Nash.png")
    plt.close()
