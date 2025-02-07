import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

# Neo4j connection settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

def init_neo4j():
    """Initialize Neo4j with indexes and constraints."""
    try:
        # Connect to Neo4j
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
        
        with driver.session() as session:
            # Create constraints
            logger.info("Creating constraints...")
            session.run("""
                CREATE CONSTRAINT document_id IF NOT EXISTS
                FOR (d:Document) REQUIRE d.id IS UNIQUE
            """)
            
            # Create indexes
            logger.info("Creating indexes...")
            session.run("""
                CREATE INDEX document_content IF NOT EXISTS
                FOR (d:Document)
                ON (d.content)
            """)
            
            session.run("""
                CREATE INDEX document_embedding IF NOT EXISTS
                FOR (d:Document)
                ON (d.embedding)
            """)
            
            logger.info("Neo4j initialization completed successfully")
            
    except Exception as e:
        logger.error(f"Error initializing Neo4j: {str(e)}")
        raise
    finally:
        driver.close()

if __name__ == "__main__":
    init_neo4j() 