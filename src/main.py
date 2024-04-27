import logging
import sys
import pandas as pd
import numpy as np
from neo4j import GraphDatabase, Session, Query, Record
from neo4j.exceptions import ServiceUnavailable

def main(session : Session, out_file=sys.stdout):
    edges = session.run("MATCH (b:Biological_sample)-[e]->(property) WHERE EXISTS {MATCH (d:Disease)<-[:HAS_DISEASE]-(b)} RETURN id(b) AS b, type(e), id(property) AS property").data()
    edges = pd.DataFrame(edges)
    patients = edges['b'].unique()
    properties = edges['type(e)'].unique()
    dataset = pd.DataFrame(index=pd.Index(patients), columns=properties)
    dataset = dataset.assign(healthy=False)
    for _, line in edges.iterrows():
        dataset.loc[line['b'], line['type(e)']] = line['property']

    healthy = session.run("MATCH (b:Biological_sample)-[:HAS_DISEASE]->(d:Disease {name: 'control'}) RETURN id(b) as b").data()
    for line in healthy:
        dataset.loc[line['b'], 'healthy'] = True
    dataset.to_csv(out_file)

def run(config, out_file=sys.stdout):
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
        main(session, out_file)

    # Close the driver connection
    driver.close()

if __name__ == "__main__":
    import yaml
    file_path = './config.yml' # TODO
    with open(file_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    run(config)
