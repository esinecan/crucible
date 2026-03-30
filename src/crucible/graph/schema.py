SCHEMA_STATEMENTS = [
    # Uniqueness constraints
    "CREATE CONSTRAINT corpus_id IF NOT EXISTS FOR (c:Corpus) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (ch:Chunk) REQUIRE ch.id IS UNIQUE",
    "CREATE CONSTRAINT insight_id IF NOT EXISTS FOR (i:Insight) REQUIRE i.id IS UNIQUE",
    # Vector indexes — separate per layer
    "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS "
    "FOR (ch:Chunk) ON (ch.embedding) "
    "OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}",
    "CREATE VECTOR INDEX insight_embedding IF NOT EXISTS "
    "FOR (i:Insight) ON (i.embedding) "
    "OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}",
    # Fulltext indexes
    "CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS FOR (ch:Chunk) ON EACH [ch.text]",
    "CREATE FULLTEXT INDEX insight_text IF NOT EXISTS FOR (i:Insight) ON EACH [i.text]",
    # Property indexes
    "CREATE INDEX document_corpus IF NOT EXISTS FOR (d:Document) ON (d.corpus_id)",
    "CREATE INDEX document_path IF NOT EXISTS FOR (d:Document) ON (d.path)",
    "CREATE INDEX chunk_document IF NOT EXISTS FOR (ch:Chunk) ON (ch.document_id)",
    "CREATE INDEX insight_corpus IF NOT EXISTS FOR (i:Insight) ON (i.corpus_id)",
    "CREATE INDEX insight_strategy IF NOT EXISTS FOR (i:Insight) ON (i.strategy)",
    "CREATE INDEX insight_layer IF NOT EXISTS FOR (i:Insight) ON (i.layer)",
]


def ensure_schema(session) -> None:
    for stmt in SCHEMA_STATEMENTS:
        session.run(stmt)
