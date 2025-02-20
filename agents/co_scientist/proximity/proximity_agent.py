"""Proximity agent for hypothesis clustering and similarity analysis."""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from ..base_agent import BaseAgent, AgentState
from ..generation.generation_agent import Hypothesis

class SimilarityScore(BaseModel):
    """Similarity assessment between two hypotheses."""
    score: float = Field(description="Overall similarity score (0-1)", ge=0.0, le=1.0)
    aspects: Dict[str, float] = Field(description="Similarity scores by aspect")
    shared_elements: List[str] = Field(description="Common elements between hypotheses")
    key_differences: List[str] = Field(description="Important distinguishing features")

class HypothesisCluster(BaseModel):
    """Cluster of related hypotheses."""
    cluster_id: str = Field(description="Unique identifier for this cluster")
    hypotheses: List[str] = Field(description="IDs of hypotheses in this cluster")
    centroid: Optional[str] = Field(description="ID of central/representative hypothesis")
    theme: str = Field(description="Main theme of the cluster")
    key_features: List[str] = Field(description="Defining features of the cluster")
    intra_cluster_similarity: float = Field(description="Average similarity within cluster", ge=0.0, le=1.0)

class ProximityState(AgentState):
    """Proximity agent state."""
    similarity_cache: Dict[str, SimilarityScore] = Field(default_factory=dict)
    clusters: Dict[str, HypothesisCluster] = Field(default_factory=dict)
    embeddings: Dict[str, List[float]] = Field(default_factory=dict)
    proximity_metrics: Dict[str, Any] = Field(default_factory=dict)

class ProximityAgent(BaseAgent):
    """Agent responsible for hypothesis clustering and similarity analysis."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        agent_id: str = "proximity",
        system_prompt: Optional[str] = None,
        embedding_dim: int = 768,
        min_cluster_size: int = 3,
        similarity_threshold: float = 0.7
    ):
        """Initialize the proximity agent."""
        if system_prompt is None:
            system_prompt = """You are the proximity agent responsible for analyzing hypothesis relationships.
Your role is to:
1. Assess hypothesis similarities
2. Identify related hypotheses
3. Form meaningful clusters
4. Find representative examples
5. Track relationship patterns
6. Guide hypothesis organization

Follow these guidelines:
- Consider multiple similarity aspects
- Identify meaningful relationships
- Form coherent clusters
- Find representative examples
- Track emerging patterns
- Maintain clear organization
- Support hypothesis navigation"""

        super().__init__(
            llm=llm,
            agent_id=agent_id,
            agent_type="proximity",
            system_prompt=system_prompt,
            output_parser=PydanticOutputParser(pydantic_object=SimilarityScore)
        )
        
        # Initialize proximity-specific state
        self.state = ProximityState(
            agent_id=agent_id,
            agent_type="proximity",
            similarity_cache={},
            clusters={},
            embeddings={},
            proximity_metrics={}
        )
        
        self.embedding_dim = embedding_dim
        self.min_cluster_size = min_cluster_size
        self.similarity_threshold = similarity_threshold
        
    async def compute_similarity(
        self,
        hypothesis_a: Hypothesis,
        hypothesis_b: Hypothesis,
        aspects: Optional[List[str]] = None
    ) -> SimilarityScore:
        """Compute similarity between two hypotheses."""
        # Check cache
        cache_key = f"{hypothesis_a.id}_{hypothesis_b.id}"
        if cache_key in self.state.similarity_cache:
            return self.state.similarity_cache[cache_key]
            
        # Generate similarity assessment using LLM
        result = await self.arun({
            "hypothesis_a": hypothesis_a.dict(),
            "hypothesis_b": hypothesis_b.dict(),
            "aspects": aspects or ["methodology", "concepts", "evidence", "implications"],
            "previous_scores": [
                score.dict() for score in self.state.similarity_cache.values()
                if hypothesis_a.id in [score.hypothesis_a, score.hypothesis_b]
                or hypothesis_b.id in [score.hypothesis_a, score.hypothesis_b]
            ]
        })
        
        # Create similarity score
        similarity = SimilarityScore(**result)
        
        # Update cache
        self.state.similarity_cache[cache_key] = similarity
        self.state.similarity_cache[f"{hypothesis_b.id}_{hypothesis_a.id}"] = similarity
        
        return similarity
        
    async def update_embeddings(self, hypotheses: List[Hypothesis]) -> None:
        """Update hypothesis embeddings."""
        for hypothesis in hypotheses:
            if hypothesis.id not in self.state.embeddings:
                # This would use a real embedding model in practice
                # For now, we'll generate random embeddings
                self.state.embeddings[hypothesis.id] = list(np.random.rand(self.embedding_dim))
                
    async def cluster_hypotheses(self, hypotheses: List[Hypothesis]) -> List[HypothesisCluster]:
        """Cluster hypotheses based on similarity."""
        # Ensure we have embeddings
        await self.update_embeddings(hypotheses)
        
        # Create embedding matrix
        embedding_matrix = np.array([
            self.state.embeddings[h.id] for h in hypotheses
        ])
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embedding_matrix)
        
        # Perform clustering
        clustering = DBSCAN(
            eps=1 - self.similarity_threshold,
            min_samples=self.min_cluster_size,
            metric="precomputed"
        ).fit(1 - similarity_matrix)
        
        # Create clusters
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label == -1:  # Noise points
                continue
                
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(hypotheses[i].id)
            
        # Create cluster objects
        cluster_objects = []
        for label, hypothesis_ids in clusters.items():
            # Find centroid (most central hypothesis)
            centroid = None
            if hypothesis_ids:
                # Use hypothesis with highest average similarity to others
                similarities = []
                for h_id in hypothesis_ids:
                    avg_sim = np.mean([
                        self.get_cached_similarity(h_id, other_id).score
                        for other_id in hypothesis_ids
                        if other_id != h_id
                    ])
                    similarities.append((h_id, avg_sim))
                centroid = max(similarities, key=lambda x: x[1])[0]
            
            cluster = HypothesisCluster(
                cluster_id=f"cluster_{label}",
                hypotheses=hypothesis_ids,
                centroid=centroid,
                theme=f"Theme for cluster {label}",  # This would be generated by LLM
                key_features=[],  # This would be generated by LLM
                intra_cluster_similarity=np.mean([
                    self.get_cached_similarity(h1, h2).score
                    for i, h1 in enumerate(hypothesis_ids)
                    for h2 in hypothesis_ids[i+1:]
                ])
            )
            cluster_objects.append(cluster)
            self.state.clusters[cluster.cluster_id] = cluster
            
        return cluster_objects
        
    def get_cached_similarity(self, hypothesis_a_id: str, hypothesis_b_id: str) -> SimilarityScore:
        """Get cached similarity score."""
        cache_key = f"{hypothesis_a_id}_{hypothesis_b_id}"
        reverse_key = f"{hypothesis_b_id}_{hypothesis_a_id}"
        
        if cache_key in self.state.similarity_cache:
            return self.state.similarity_cache[cache_key]
        if reverse_key in self.state.similarity_cache:
            return self.state.similarity_cache[reverse_key]
            
        # Return default score if not found
        return SimilarityScore(
            score=0.0,
            aspects={},
            shared_elements=[],
            key_differences=[]
        )
        
    def get_similar_hypotheses(
        self,
        hypothesis: Hypothesis,
        min_similarity: float = 0.7,
        max_results: int = 5
    ) -> List[Tuple[str, float]]:
        """Get most similar hypotheses to the given one."""
        similarities = []
        for other_id in self.state.embeddings.keys():
            if other_id != hypothesis.id:
                score = self.get_cached_similarity(hypothesis.id, other_id)
                similarities.append((other_id, score.score))
                
        return sorted(
            [s for s in similarities if s[1] >= min_similarity],
            key=lambda x: x[1],
            reverse=True
        )[:max_results]
        
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get statistics about hypothesis clusters."""
        if not self.state.clusters:
            return {}
            
        stats = {
            "total_clusters": len(self.state.clusters),
            "average_cluster_size": np.mean([
                len(c.hypotheses) for c in self.state.clusters.values()
            ]),
            "largest_cluster": max(
                len(c.hypotheses) for c in self.state.clusters.values()
            ),
            "average_intra_cluster_similarity": np.mean([
                c.intra_cluster_similarity for c in self.state.clusters.values()
            ]),
            "cluster_sizes": {
                c.cluster_id: len(c.hypotheses)
                for c in self.state.clusters.values()
            }
        }
        
        return stats
        
    def analyze_proximity_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in hypothesis relationships."""
        if not self.state.similarity_cache:
            return {}
            
        # Calculate overall similarity statistics
        similarities = [s.score for s in self.state.similarity_cache.values()]
        
        # Analyze aspect-specific patterns
        aspect_stats = {}
        for score in self.state.similarity_cache.values():
            for aspect, value in score.aspects.items():
                if aspect not in aspect_stats:
                    aspect_stats[aspect] = []
                aspect_stats[aspect].append(value)
                
        return {
            "total_comparisons": len(self.state.similarity_cache) // 2,  # Divide by 2 due to symmetry
            "average_similarity": np.mean(similarities),
            "similarity_distribution": {
                "min": min(similarities),
                "max": max(similarities),
                "std": np.std(similarities)
            },
            "aspect_averages": {
                aspect: np.mean(values)
                for aspect, values in aspect_stats.items()
            },
            "cluster_stats": self.get_cluster_stats()
        } 