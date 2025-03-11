from neo4j import GraphDatabase

class GDB:
    '''
    This class connects to a graph database on Neo4j
    '''
    def __init__(self):

        #uri = "bolt://localhost:7687"
        self.uri = "neo4j+s://cf2eabb5.databases.neo4j.io"
        self.username = "neo4j"
        self.password = "3y-Emid1PcxxYna6XpIfovSatHVYlZdbYJNRfsa30Ss"
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

        # Metadata for db, information for the llm.
        with self.driver.session() as session:
            node_schema_query = ("CALL db.schema.nodeTypeProperties")
            rel_schema_query = ("CALL db.schema.relTypeProperties")
            viz_schema_query = ("CALL db.schema.visualization")
            full_rel_query = ("MATCH (n)-[r]->(m) "
                              "RETURN labels(n) AS StartNodeLabels, "
                              "properties(n) AS StartNodeProperties, type(r) AS RelationshipType, "
                              "properties(r) AS RelationshipProperties, labels(m) AS EndNodeLabels, "
                              "properties(m) AS EndNodeProperties;")
            self.node_schema = session.execute_read(self.run_query, node_schema_query)
            self.rel_schema = session.execute_read(self.run_query,rel_schema_query)
            self.viz_schema = session.execute_read(self.run_query,viz_schema_query)
            self.full_rel_query = session.execute_read(self.run_query,full_rel_query)

    def run_query(self, tx, query, parameters=None):
        result = tx.run(query, parameters)
        return [record.data() for record in result]

    def connect_driver(self):
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

    def close_driver(self):
        self.driver.close()

'''
gdb = GDB()

with gdb.driver.session() as session:
    query = ("MATCH(n:PLAYER {name:\"LeBron James\"}) -[:TEAMMATES]-> (teammates:PLAYER) return teammates")
    #result = session.execute_read(gdb.run_query, query)
    result = session.run(query)
    for record in result:
        print(record)

gdb.driver.close()
'''