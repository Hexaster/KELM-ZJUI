from openai import OpenAI
from GDB import GDB

# Use deepseek as an encoder to transform from natural language to Neo4j queries
# Can use a local model to do the same job as well
class Neo4jEncoder:
    def __init__(self):
        self.client = OpenAI(api_key="sk-1a1f87fcaa23438d9f630f822d729d63", base_url="https://api.deepseek.com")
        self.gdb = GDB()
        self.messages = [
            {"role": "system", "content": "You are a Neo4j encoder to translate users' natural language to Neo4j queries."
                                          "Here's the metadata for the database:"
                                          f"node schema: {self.gdb.node_schema}"
                                          f"relation schema: {self.gdb.rel_schema}"
                                          f"viz schema: {self.gdb.viz_schema}"
                                          f"The full database: {self.gdb.full_rel_query}"
                                          f"Please be aware of users' typos, you should correct them."
                                          f"Please return the cypher code only, do not include any explanations or additional text, do not include ```cypher```."}
        ]
        pass

    def encode(self, input):
        '''
        Encoding the input to a Neo4j query.
        :param input: The input user gave
        :return: The coresponding Neo4j query
        '''
        self.messages.append({"role": "user", "content": input})
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=self.messages,
            stream = False
        )
        return response.choices[0].message.content

'''
encoder = Neo4jEncoder()

with encoder.gdb.driver.session() as session:
    query = encoder.encode("Who's the teammate of Labron James?")
    result = session.execute_read(encoder.gdb.run_query, query)
    for record in result:
        print(record)
'''
