from neo4j import GraphDatabase
from rich.console import Console
from rich.table import Table
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Rich console
console = Console()

# Neo4j connection details
uri = "bolt://localhost:7687"
username = "neo4j"
password = "password123"

def inspect_graph():
    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        with driver.session() as session:
            # Get node counts by label
            node_counts = session.run("""
                MATCH (n)
                RETURN labels(n) as labels, count(n) as count
            """)
            
            # Create table for node counts
            table = Table(title="Neo4j Graph Statistics")
            table.add_column("Label", style="cyan")
            table.add_column("Count", style="green")
            
            for record in node_counts:
                table.add_row(str(record["labels"]), str(record["count"]))
            
            console.print(table)
            
            # Get sample documents
            console.print("\n[bold]Sample Documents:[/bold]")
            documents = session.run("""
                MATCH (d:Document)
                RETURN d.content as content, d.metadata_str as metadata
                LIMIT 5
            """)
            
            for doc in documents:
                console.print(f"\n[cyan]Content:[/cyan] {doc['content']}")
                console.print(f"[cyan]Metadata:[/cyan] {doc['metadata']}")
            
            # Check indexes
            console.print("\n[bold]Indexes:[/bold]")
            indexes = session.run("SHOW INDEXES")
            
            index_table = Table(title="Neo4j Indexes")
            index_table.add_column("Name", style="cyan")
            index_table.add_column("Type", style="green")
            index_table.add_column("Properties", style="yellow")
            
            for index in indexes:
                index_table.add_row(
                    str(index.get("name", "N/A")),
                    str(index.get("type", "N/A")),
                    str(index.get("properties", []))
                )
            
            console.print(index_table)

if __name__ == "__main__":
    try:
        inspect_graph()
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}") 