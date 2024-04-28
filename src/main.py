import logging
import sys
import os
import pandas as pd
import numpy as np
import neo4j
from neo4j import GraphDatabase, Session, Query, Record
from neo4j.exceptions import ServiceUnavailable
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.multioutput import MultiOutputClassifier

import featurecloud

mlb = MultiLabelBinarizer()
all_ids = None

def get_data(session: Session):
    phenotype_df = session.run("MATCH (b:Biological_sample)-[r:HAS_PHENOTYPE]-(c)  RETURN b.subjectid AS id, c.name AS phenotype_name").to_df()
    damaged_genes_df = session.run("MATCH (b:Biological_sample)-[r:HAS_DAMAGE]-(c)  RETURN c.name AS gene_name, b.subjectid AS id").to_df()
    protein_df = session.run("MATCH (b:Biological_sample)-[r:HAS_PROTEIN]-(c)  RETURN c.name AS protein_name, b.subjectid AS id").to_df()
    disease_df = session.run("MATCH (b:Biological_sample)-[r:HAS_DISEASE]-(c) RETURN b.subjectid AS id,c.name AS disease_name").to_df()

    data_df = pd.merge(phenotype_df, damaged_genes_df, on='id', how='outer')
    data_df = pd.merge(data_df, protein_df, on='id', how='outer')

    if os.path.exists('./data.csv'):
        os.remove('data.csv')
    data_df.to_csv('data.csv', index=False)

    if os.path.exists('./labels.csv'):
        os.remove('labels.csv')
    disease_df.to_csv('labels.csv', index=False)

    return data_df, disease_df

def process_data_training(data_df):
    # data_df = pd.read_csv('data.csv')
    grouped_df = data_df.groupby('id').agg({
        'gene_name': lambda x: list(x),
        'protein_name': lambda x: list(x),
        'phenotype_name': lambda x:list(x)
    }).reset_index()

    # Encode genes
    encoded_genes = pd.DataFrame(mlb.fit_transform(grouped_df['gene_name']), columns=mlb.classes_)
    encoded_genes.columns = ['gene_' + col for col in encoded_genes.columns]

    # Encode proteins
    encoded_proteins = pd.DataFrame(mlb.transform(grouped_df['protein_name']), columns=mlb.classes_)
    encoded_proteins.columns = ['protein_' + col for col in encoded_proteins.columns]

    # Encode phenotypes
    encoded_phenotypes = pd.DataFrame(mlb.transform(grouped_df['phenotype_name']), columns=mlb.classes_)
    encoded_phenotypes.columns = ['phenotype_' + col for col in encoded_phenotypes.columns]

    # Concatenate the 'id' column with the encoded gene, protein, and phenotype data
    result = pd.concat([grouped_df['id'], encoded_genes, encoded_proteins, encoded_phenotypes], axis=1)

    global all_ids
    all_ids = pd.unique(result['id'])
    return result

def process_data_validation(disease_df):
    global all_ids
    # Process labels
    # disease_df = pd.read_csv('labels.csv')
    grouped_df = disease_df.groupby('id').agg({'disease_name': lambda x: list(x)})
    disease_encoded = pd.DataFrame(mlb.fit_transform(
        grouped_df['disease_name']), columns=mlb.classes_, index=grouped_df.index)

    # Concatenate the 'id' column with the encoded diseases
    encoded_df = pd.concat([grouped_df.index.to_frame(), disease_encoded], axis=1)
    # Creating negative_examples
    neg_ids = [i for i in all_ids if i not in pd.unique(encoded_df['id'])]
    neg_df = pd.DataFrame(0, index=neg_ids, columns=mlb.classes_)
    for index, row in neg_df.iterrows():
        neg_df['id'] = index
    result = pd.concat([encoded_df, neg_df], axis=0)
    return result

from sklearn.ensemble import RandomForestClassifier
def random_forest(features_df, labels_df):
    features = process_data_training(features_df)
    labels = process_data_validation(labels_df)
    X = features.drop('id', axis=1) # Features
    y = labels.drop('id', axis=1) # labels

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=27)
    rf_classifier = RandomForestClassifier(class_weight='balanced',
        n_estimators=500, min_samples_split=4,
        max_depth=20, random_state=27)
    classifier = MultiOutputClassifier(rf_classifier)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    return y_pred

from sklearn.ensemble import GradientBoostingClassifier
def gradient_boosting(features_df, labels_df):
    features = process_data_training(features_df)
    labels = process_data_validation(labels_df)
    X = features.drop('id', axis=1) # Features
    y = labels.drop('id', axis=1) # labels

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=27)
    gb_classifier = GradientBoostingClassifier(learning_rate=0.1, max_depth=5, n_estimators=150)
    classifier = MultiOutputClassifier(gb_classifier)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    return y_pred


def main(session: Session):
    features, labels = get_data(session)
    random_forest(features, labels)

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
        main(session)

    # Close the driver connection
    driver.close()

if __name__ == "__main__":
    import yaml
    file_path = './config.yml' # TODO
    with open(file_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    run(config)
