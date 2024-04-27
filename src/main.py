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
        self.phenotype_encoder = nn.Linear(in_features=nb_phenotypes, out_features=Predictor.PHENOTYPE_EMBEDDING_SIZE)
        self.gene_encoder = nn.Linear(in_features=nb_genes, out_features=Predictor.GENE_EMBEDDING_SIZE)
        self.protein_encoder = nn.Linear(in_features=nb_proteins, out_features=Predictor.PROTEIN_EMBEDDING_SIZE)

        FINAL_EMBEDDING_SIZE=15
        self.fc1 = nn.Linear(in_features=(Predictor.PHENOTYPE_EMBEDDING_SIZE+Predictor.PROTEIN_EMBEDDING_SIZE+Predictor.GENE_EMBEDDING_SIZE), out_features=FINAL_EMBEDDING_SIZE)
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

def make_hugematrix_dataset(session, patient_constraint='') -> Tuple[Dict[str, torch.Tensor], Dict[int, int], pd.Series]:
    # Get all data
    data = dict(
        patient = session.run("MATCH (b:Biological_sample) "+patient_constraint+" RETURN b.subjectid AS subject_id").to_df()['subject_id'],
        gene = session.run("MATCH (g:Gene) WHERE EXISTS { (b:Biological_sample)-[:HAS_DAMAGE]->(g) } RETURN ID(g) AS id").to_df()['id'],
        protein = session.run("MATCH (p:Protein) WHERE EXISTS { (b:Biological_sample)-[:HAS_PROTEIN]->(p) } RETURN ID(p) AS id").to_df()['id'],
        phenotype = session.run("MATCH (p:Phenotype) WHERE EXISTS { (b:Biological_sample)-[:HAS_PHENOTYPE]->(p) } RETURN ID(p) AS id").to_df()['id'],
    )
    # Assign a list index ("code") to each ID
    codes = { key: {} for key in data.keys() }
    for key, series in data.items():
        for index, val in enumerate(series):
            codes[key][int(val)] = index
    patient_codes = codes['patient']
    # Create sparse datasets
    sparse_datasets = {}
    for key, connection_name, dtype in zip(['gene', 'protein', 'phenotype'], ['HAS_DAMAGE', 'HAS_PROTEIN', 'HAS_PHENOTYPE'], [bool, torch.float32, torch.float32]):
        code = codes[key]
        ## Retrieve data from neo4j
        if connection_name == 'HAS_PHENOTYPE':
            edges = session.run(f"MATCH (b:Biological_sample)-[c:{connection_name}]->(p:{key.title()}) {patient_constraint} RETURN b.subjectid AS subject_id, id(p) AS values").to_df()
            subject_ids, property_ids, weights = edges['subject_id'], edges['values'], cycle([True])
        else:
            edges = session.run(f"MATCH (b:Biological_sample)-[c:{connection_name}]->(p:{key.title()}) {patient_constraint} RETURN b.subjectid AS subject_id, c.score AS weight, id(p) AS values").to_df()
            subject_ids, property_ids, weights = edges['subject_id'], edges['values'], edges['weight']

        indices, values = [], []
        ## Put data into sparse dataset
        for subject_id, property_id, weight in zip(subject_ids, property_ids, weights):
            subject_id, property_id = patient_codes[int(subject_id)], code[int(property_id)]
            indices.append([ subject_id, property_id ])
            values.append(weight)
        indices = torch.tensor(indices).transpose(0, 1)
        sparse_datasets[key] = torch.sparse_coo_tensor(indices, values, size=(len(patient_codes), len(code)), dtype=dtype)

    return sparse_datasets, patient_codes, data['patient']

def main(session : Session, task_1_file=sys.stdout, task_2_file=sys.stdout):
    diseases = session.run("MATCH (d:Disease) RETURN d").to_df()['d']
    sparse_datasets, patient_codes, _ = make_hugematrix_dataset(session, patient_constraint='WHERE EXISTS {MATCH (b)-[:HAS_DISEASE]->(d:Disease)}')

    disease_per_patient = session.run(f"MATCH (b:Biological_sample)-[:HAS_DISEASE]->(d:Disease) RETURN b.subjectid AS subject_id, d").to_df()

    disease_labels = np.zeros((len(patient_codes),), dtype=np.int8)
    for patient_id, disease in zip(disease_per_patient['subject_id'], disease_per_patient['d']):
        if disease['name'] == 'control':
            icdm = 0
        else:
            icdm = next(filter(lambda name: name.startswith('ICD10CM:'), disease['synonyms']), None)
            if icdm is None:
                icdm = 'A' # TODO
            else:
                icdm = icdm[len('ICD10CM:')]
            icdm = ord(icdm) - ord('a') + 1
        disease_labels[patient_codes[int(patient_id)]] = icdm

    dataset = TensorDataset(*sparse_datasets.values(), torch.tensor(disease_labels))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader, val_loader = DataLoader(train_dataset), DataLoader(val_dataset)

    predictor = Predictor(**{ 'nb_' + key + 's' : value.size(1) for key, value in sparse_datasets.items() })
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

    sparse_datasets, patient_codes, all_patients = make_hugematrix_dataset(session, patient_constraint='')
    predictions = np.zeros((len(all_patients),), dtype=np.int8)
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
