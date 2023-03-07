from itertools import combinations
import numpy as np
import pandas as pd
import math
import random
import re


class DatasetLoader:
    def __init__(self, data_path):
        self.codes = {}
        self.data_path = data_path
        self.drug_gene = []
        self.path_sim = []
        self.gene_phenotype = []
        self.drug_phenotype = []
        self.baits_prey = []

    def drug_target_load(self):
        # DRUG'S TARGET
        drug_gene = pd.read_csv(self.data_path + "data/CTD_gene_drug_interaction.tsv", sep='\t')

        # print(drug_gene.info())
        self.codes['drugname2mesh'] = {row[0].upper(): row[1] for idx, row in
                                       drug_gene[['Chemical Name', 'Chemical ID']].drop_duplicates().iterrows()}
        self.codes['mesh2drugname'] = {row[0].upper(): row[1] for idx, row in
                                       drug_gene[['Chemical Name', 'Chemical ID']].drop_duplicates().iterrows()}
        self.codes['gene_symbol2id'] = {row[0].upper(): row[1] for idx, row in
                                        drug_gene[['Gene Symbol', 'Gene ID']].drop_duplicates().iterrows()}

        drug_gene.drop_duplicates(subset=['Gene ID', 'Chemical Name'], keep='first',
                                  inplace=True)
        drug_gene = drug_gene[['Gene ID', 'Chemical Name']]
        drug_gene['Chemical Name'] = drug_gene['Chemical Name'].str.upper()
        drug_gene['Chemical Name'] = 'drug_' + drug_gene['Chemical Name'].astype(str)
        drug_gene['Gene ID'] = 'gene_' + drug_gene['Gene ID'].astype(str)

        drug_gene.drop_duplicates(keep='first', inplace=True)
        return drug_gene

    def pathways_load(self):
        # PATHWAYS
        pathways = pd.read_csv(self.data_path + "data/CTD_pathways.tsv",
                               sep='\t')

        # print(pathways.info())
        path_sim = pd.concat([pd.DataFrame(list(combinations(pathway, 2, )), columns=['gene1', 'gene2']) \
                              for pathway in pathways['Association inferred via'].apply(lambda x: x.split('|') \
                if '|' in x else None).dropna().values]).drop_duplicates()

        path_sim_kegg = pd.read_csv(self.data_path + 'data/KegglinkevaluationPPPN_1', header=None, sep='\t')
        path_sim_kegg.columns = ['gene1', 'gene2', 'positive']
        path_sim_kegg.replace('PP', 1, inplace=True)
        path_sim_kegg.replace('PN', 0, inplace=True)

        path_sim_kegg = path_sim_kegg.loc[path_sim_kegg['positive'] == 1, ['gene1', 'gene2']]

        gene_name = pd.read_excel(self.data_path + 'data/All_Human_Protein_Coding_Genes_3_27_2020.xlsx')
        gene_dict = {row['Gene Id']: row['Gene Symbol'] for _, row in gene_name[['Gene Id', 'Gene Symbol']].iterrows()}

        path_sim_kegg['gene1'] = path_sim_kegg['gene1'].apply(lambda x: gene_dict.get(x))
        path_sim_kegg['gene2'] = path_sim_kegg['gene2'].apply(lambda x: gene_dict.get(x))
        path_sim_kegg.dropna(inplace=True)

        # print(path_sim.head())
        # We need other pathways:
        # - one from KEGG
        # - ace-inhibitors since the ace pathways are used from sars-cov-2 to enter host cells
        ace_inh = pd.read_csv(self.data_path + "data/ace-inhibitors.tsv", sep='\t')
        # print(path_ace_inh.head())
        path_ace_inh = list()
        genes_1 = (ace_inh['From'].dropna())
        genes_2 = (ace_inh['To'].dropna())

        for i in range(len(ace_inh)):
            if genes_1[i].isupper():
                path_ace_inh.append(genes_1[i])
            if genes_2[i].isupper():
                path_ace_inh.append(genes_2[i])

        path_ace_inh = list(dict.fromkeys(path_ace_inh))
        # print(len(path_ace_inh))  # il numero non corrisponde a quello riportato sul preprocessing probabilmente è perché sono stati aggiornati proprio i dataset)

        fluv = pd.read_csv(self.data_path + "data/Fluvoxamine_Pathway.tsv", sep='\t')
        pathway_fluv = list(dict.fromkeys(fluv['Genes'].dropna()))

        losartan = pd.read_csv(self.data_path + "data/losartan_pathway.tsv", sep='\t')
        # pathway_losartan=list(set(['AGTR1','CYP2C9',"CYP3A4",'CYP2C9',"CYP3A4",'CYP2C9',"CYP3A4", 'CYP2C9',"CYP3A4", 'UGT1A1',"UGT2B7"]))
        pathway_losartan = []
        for g in losartan['Genes'].dropna():
            gs = g.replace('"', '').split(',')
            pathway_losartan += gs
        pathway_losartan = list(set(pathway_losartan))

        propofol = pd.read_csv(self.data_path + "data/propofol_pathway.tsv", sep='\t')
        pathway_propofol = []
        for g in propofol['Controller'].dropna():
            gs = g.replace('"', '').split(',')
            pathway_propofol += gs
        pathway_propofol = list(set(pathway_propofol))

        remdesivir = pd.read_csv(self.data_path + "data/Remdesivir_pathway.tsv", sep='\t')
        pathway_remdesivir = []
        for g in remdesivir['Controller'].dropna():
            gs = g.replace('"', '').split(',')
            pathway_remdesivir += gs
        pathway_remdesivir = list(set(pathway_remdesivir))

        # losartan = losartan['Genes']
        # .apply(lambda x: x.split(','))

        # pathway_losartan = list(losartan)

        path_sim = pd.concat(
            [path_sim] + [path_sim_kegg] + [pd.DataFrame(list(combinations(pathway, 2, )), columns=['gene1', 'gene2']) \
                                            for pathway in
                                            [path_ace_inh, pathway_fluv, pathway_losartan, pathway_propofol,
                                             pathway_remdesivir]])

        path_sim['gene1'] = path_sim['gene1'].apply(lambda x: self.codes['gene_symbol2id'].get(x))
        path_sim['gene2'] = path_sim['gene2'].apply(lambda x: self.codes['gene_symbol2id'].get(x))

        path_sim.dropna(inplace=True)

        path_sim['gene1'] = path_sim['gene1'].apply(lambda x: 'gene_' + str(x))
        path_sim['gene2'] = path_sim['gene2'].apply(lambda x: 'gene_' + str(x))
        path_sim.drop_duplicates(inplace=True)
        return path_sim

    def phenotypes_load(self):
        # PHENOTYPES
        phenotypes = pd.read_csv(self.data_path + "data/CTD_diseases_phenotypes.tsv", sep='\t')

        self.codes['phenotype_id_to_name'] = {row[0]: row[1] for idx, row in phenotypes[
            ['Phenotype Term ID', 'Phenotype Term Name']].drop_duplicates().iterrows()}
        # Drug & phenotypes
        drug_phenotype = phenotypes['Chemical Inference Network'].dropna().apply(lambda x: x.split('|')).apply(
            pd.Series) \
            .merge(phenotypes['Phenotype Term ID'], left_index=True, right_index=True) \
            .melt(id_vars=['Phenotype Term ID'], value_name='drug').drop('variable', axis=1).dropna()
        # ho capito che fa quindi mi è sembrato inutile scriverlo in altro modo
        drug_phenotype['drug'] = drug_phenotype['drug'].str.upper()
        drug_phenotype['drug'] = 'drug_' + drug_phenotype['drug'].astype(str)
        drug_phenotype['Phenotype Term ID'] = 'phenotype_' + drug_phenotype['Phenotype Term ID'].astype(str)
        drug_phenotype = drug_phenotype[['drug', 'Phenotype Term ID']]

        # Genes and phenotypes
        gene_phenotype = phenotypes['Gene Inference Network'].dropna().apply(lambda x: x.split('|')).apply(pd.Series) \
            .merge(phenotypes['Phenotype Term ID'], left_index=True, right_index=True) \
            .melt(id_vars=['Phenotype Term ID'], value_name='gene').drop('variable', axis=1).dropna()

        gene_phenotype['Phenotype term ID'] = 'phenotype_' + gene_phenotype['Phenotype Term ID'].astype(str)
        gene_phenotype['gene'] = gene_phenotype['gene'].apply(lambda x: self.codes['gene_symbol2id'].get(x))
        gene_phenotype.dropna(inplace=True)

        gene_phenotype['gene'] = 'gene_' + gene_phenotype['gene'].astype(str)
        gene_phenotype = gene_phenotype[['gene', 'Phenotype Term ID']]
        return drug_phenotype, gene_phenotype

    def baits_prey_load(self):
        # BAITS AND PREY
        baits_prey = pd.read_csv(self.data_path + "data/2020-03-18_Krogan_SARSCoV2_27baits.csv")

        # We're interested in PreyGene and Bait only from this dataset, so let's drop the other columns:
        baits_prey = baits_prey[['Bait', 'PreyGene']]
        baits_prey['Bait'] = 'bait_' + baits_prey['Bait'].astype(str)
        baits_prey['PreyGene'] = baits_prey['PreyGene'].apply(lambda x: self.codes['gene_symbol2id'].get(x))
        baits_prey.dropna(inplace=True)
        baits_prey['PreyGene'] = 'gene_' + baits_prey['PreyGene'].astype(str)
        return baits_prey

    def load_data(self):

        self.drug_gene = self.drug_target_load()
        self.path_sim = self.pathways_load()
        self.gene_phenotype, self.drug_phenotype = self.phenotypes_load()
        self.baits_prey = self.baits_prey_load()

        self.drug_gene.columns = ['node1', 'node2']
        self.path_sim.columns = ['node1', 'node2']
        self.baits_prey.columns = ['node1', 'node2']
        self.gene_phenotype.columns = ['node1', 'node2']
        self.drug_phenotype.columns = ['node1', 'node2']

        self.drug_gene['type'] = 'gene-drug'
        self.path_sim['type'] = 'gene-gene'
        self.baits_prey['type'] = 'bait-gene'
        self.gene_phenotype['type'] = 'gene-phenotype'
        self.drug_phenotype['type'] = 'drug-phenotype'
        edge_index = pd.concat(
            [self.drug_gene, self.path_sim, self.baits_prey, self.gene_phenotype, self.drug_phenotype])
        edge_index['node1'] = edge_index['node1'].astype(str)
        edge_index['node2'] = edge_index['node2'].astype(str)

        return edge_index
    