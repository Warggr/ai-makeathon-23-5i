import logging
import sys
import pandas as pd
import numpy as np
import scipy as sp
import neo4j
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from neo4j import GraphDatabase, Session, Query, Record
from neo4j.exceptions import ServiceUnavailable
from typing import Dict, Tuple
from itertools import cycle


class Predictor(nn.Module):
    PHENOTYPE_EMBEDDING_SIZE = 10
    PROTEIN_EMBEDDING_SIZE = 10
    GENE_EMBEDDING_SIZE = 10

    def __init__(self, nb_phenotypes, nb_genes, nb_proteins):
        self.phenotype_encoder = nn.Linear(in_features=nb_phenotypes, out_features=PHENOTYPE_EMBEDDING_SIZE)
        self.gene_encoder = nn.Linear(in_features=nb_genes, out_features=GENE_EMBEDDING_SIZE)
        self.protein_encoder = nn.Linear(in_features=nb_proteins, out_features=PROTEIN_EMBEDDING_SIZE)

        FINAL_EMBEDDING_SIZE=15
        self.fc1 = nn.Linear(in_features=(PHENOTYPE_EMBEDDING_SIZE+PROTEIN_EMBEDDING_SIZE+GENE_EMBEDDING_SIZE), out_features=FINAL_EMBEDDING_SIZE)
        self.fc2 = nn.Linear(in_features=FINAL_EMBEDDING_SIZE, out_features=27)
    def forward(self, phenotypes, genes, proteins):
        phenotype_encoding = self.phenotype_encoder(phenotypes)
        gene_encoding = self.gene_encoder(genes)
        protein_encoding = self.protein_encoder(proteins)
        encoding = torch.concat(phenotype_encoding, gene_encoding, protein_encoding)
        encoding = nn.functional.relu(encoding)
        x = self.fc1(encoding)
        x = nn.functional.relu(x)
        x = self.fc2(encoding)
        x = nn.functional.softmax(x)
        return x

def make_hugematrix_dataset(patient_constaint='') -> Tuple[Dict[str, torch.Tensor], Dict[int, int], pd.Column]:
    connection_names = ['HAS_PHENOTYPE', 'HAS_DAMAGE', 'HAS_PROTEIN']
    # Get all data
    data = dict(
        patients = session.run("MATCH (b:Biological_sample) "+patient_constaint+" RETURN b.subjectid AS subject_id").to_df()['subject_id'],
        genes = session.run("MATCH (g:Gene) WHERE EXISTS { (b:Biological_sample)-[:HAS_DAMAGE]->(g) } RETURN g").to_df()['g'],
        proteins = session.run("MATCH (p:Protein) RETURN p").to_df()['p'],
        phenotypes = session.run("MATCH (p:Phenotype) RETURN p").to_df()['p'],
    )
    # Assign a list index ("code") to each ID
    codes = { key: {} for key in data.keys() }
    for key, data in data.items():
        for index, val in enumerate(data):
            codes[key][val] = index
    patient_codes = codes['patients']
    # Create empty sparse dataset
    sparse_datasets = dict(
        phenotype = torch.sparse_coo_tensor(indices=[], values=[], size=(len(patient_codes), len(phenotype_codes)), dtype=bool),
        gene = torch.sparse_coo_tensor(indices=[], values=[], size=(len(patient_codes), len(gene_codes)), dtype=torch.float32),
        protein = torch.sparse_coo_tensor(indices=[], values=[], size=(len(patient_codes), len(protein_codes)), dtype=torch.float32),
    )
    # Fill sparse datasets
    for key, connection_name in zip(sparse_datasets.keys(), connection_names):
        dataset = sparse_datasets[key]
        code = codes[key]
        ## Retrieve data from neo4j
        if connection_name == 'HAS_PHENOTYPE':
            edges = session.run(f"MATCH (b:Biological_sample)-[c:{connection_name}]->(p:{key.title()}) RETURN b.subjectid AS subject_id, id(p) AS values").to_df()
            subject_ids, values, weights = edges['subject_id'], edges['values'], cycle(True)
        else:
            edges = session.run(f"MATCH (b:Biological_sample)-[c:{connection_name}]->(p:{key.title()}) RETURN b.subjectid AS subject_id, c.score AS weight, id(p) AS values").to_df()
            subject_ids, values, weights = edges['subject_id'], edges['values'], edges['weight']

        ## Put data into sparse dataset
        for subject_id, value, weight in zip(subject_ids, values, weights):
            subject_id, value = patient_codes[subject_id], code[value]
            dataset[subject_id, value] = weight

    return sparse_datasets, patient_codes, data['patients']

def main(session : Session, task_1_file=sys.stdout, task_2_file=sys.stdout):
    diseases = session.run("MATCH (d:Disease) RETURN d").to_df()['d']
    sparse_datasets, patient_codes, _ = make_hugematrix_dataset(patient_constaint='WHERE EXISTS {MATCH (b)-[:HAS_DISEASE]->[d:Disease]}')

    disease_per_patient = session.run(f"MATCH (b:Biological_sample)-[:HAS_DISEASE]->(d:Disease) RETURN b.subjectid AS subject_id, d.synonyms").to_df()

    disease = np.array(size=(len(patient_codes),), dtype=np.int8)
    for patient_id, disease in disease_per_patient['subject_id'], disease_per_patient['synonyms']:
        if disease.name == 'control':
            icdm = 0
        else:
            icdm = next(filter(lambda name: name.startswith('ICD10CM:'), disease))
            icdm = icdm[len('ICD10CM:')]
            icdm = ord(icdm) - ord('a') + 1
        disease[patient_codes[patient_id]] = icdm

    dataset = TensorDataset(*sparse_datasets.values(), disease)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader, val_loader = DataLoader(train_dataset), DataLoader(val_dataset)

    predictor = Predictor(**{ 'nb_' + key + 's' : len(value) for key, value in codes.items() })
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train(True)
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            *inputs, disease = data
            disease = nn.functional.one_hot(disease,num_classes=27)
            optimizer.zero_grad()
            prediction = predictor(*inputs)
            loss = loss_fn(prediction, disease)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = running_loss / (i+1)
        model.eval()

        with torch.no_grad():
            running_loss = 0.0
            for i, data in enumerate(val_loader):
                *inputs, disease = data
                disease = nn.functional.one_hot(disease,num_classes=27)
                prediction = predictor(*inputs)
                loss = loss_fn(prediction, disease)
                running_loss += loss
            val_loss = running_loss / (i+1)
        print(f'Epoch {epoch}: LOSS train {train_loss} val {val_loss}')

    sparse_datasets, patient_codes, all_patients = make_hugematrix_dataset(patient_constaint='')
    predictions = np.array(size=(len(all_patients),), dtype=np.int8)
    for i in range(len(all_patients)):
        predictions[i] = predictor(*[ series[i] for series in sparse_datasets.values() ])

    datasetA = pd.DataFrame({ 'subject_id': all_patients, 'disease': predictions })
    datasetA.to_csv(task_1_file, index=False)

    datasetB = datasetA[datasetA['disease'] != 0]
    datasetB['disease'].apply(lambda charcode: chr(charcode-1))
    datasetB.to_csv(task_2_file, index=False)

def run(config, task_1_file=sys.stdout, task_2_file=sys.stdout):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Get Neo4j credentials from config
    neo4j_credentials = config.get("neo4j_credentials", {})
    NEO4J_URI = neo4j_credentials.get("NEO4J_URI", "")
    NEO4J_USERNAME = neo4j_credentials.get("NEO4J_USERNAME", "")
    NEO4J_PASSWORD = neo4j_credentials.get("NEO4J_PASSWORD", "")
    NEO4J_DB = neo4j_credentials.get("NEO4J_DB", "")
    logger.info(f"Neo4j Connect to {NEO4J_URI} using {NEO4J_USERNAME}")

    # Driver instantiation
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    # Create a driver session with defined DB
    with driver.session(database=NEO4J_DB) as session:
        main(session, task_1_file, task_2_file)

    # Close the driver connection
    driver.close()

if __name__ == "__main__":
    import yaml
    file_path = './config.yml' # TODO
    with open(file_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    run(config)
