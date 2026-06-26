"""
vectorless_rag/tree_builder.py
Builds a knowledge graph from document chunks using spaCy NER.
Enables entity-centric traversal: "Who approved doc X?" → graph walk.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

import networkx as nx
from loguru import logger

from src.models import Chunk, RetrievedChunk, RouteType


class KnowledgeGraphBuilder:
    """
    Extracts (entity, relation, entity) triples from text
    and stores them in a NetworkX directed graph.
    """

    def __init__(self, graph_path: str = "data/indexes/knowledge_graph.json"):
        self.graph_path = Path(graph_path)
        self.graph: nx.DiGraph = nx.DiGraph()
        self._nlp = None
        # chunk lookup: entity -> list of chunk_ids mentioning it
        self._entity_chunk_map: dict[str, list[str]] = defaultdict(list)
        self._chunk_store: dict[str, Chunk] = {}
        if self.graph_path.exists():
            self._load()

    def _get_nlp(self):
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_trf")
            except OSError:
                logger.warning("en_core_web_trf not found, falling back to en_core_web_sm")
                self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def build(self, chunks: list[Chunk]):
        nlp = self._get_nlp()
        self._chunk_store = {c.chunk_id: c for c in chunks}

        logger.info(f"Building knowledge graph from {len(chunks)} chunks...")
        for chunk in chunks:
            doc = nlp(chunk.content[:1000])  # cap at 1000 chars for speed
            entities = [(ent.text.strip(), ent.label_) for ent in doc.ents if len(ent.text.strip()) > 1]

            for ent_text, ent_label in entities:
                node_id = f"{ent_label}:{ent_text}"
                self.graph.add_node(node_id, label=ent_label, name=ent_text)
                self._entity_chunk_map[node_id].append(chunk.chunk_id)

            # simple co-occurrence edges (entities appearing in same sentence)
            for sent in doc.sents:
                sent_ents = [
                    f"{e.label_}:{e.text.strip()}"
                    for e in sent.ents
                    if len(e.text.strip()) > 1
                ]
                for i, e1 in enumerate(sent_ents):
                    for e2 in sent_ents[i + 1 :]:
                        if self.graph.has_edge(e1, e2):
                            self.graph[e1][e2]["weight"] += 1
                        else:
                            self.graph.add_edge(e1, e2, weight=1, relation="co-occurs")

        logger.info(f"Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        self._save()

    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        """Find chunks containing entities most related to the query."""
        nlp = self._get_nlp()
        doc = nlp(query)
        query_entities = [f"{e.label_}:{e.text.strip()}" for e in doc.ents]

        if not query_entities:
            logger.debug("No entities found in query for graph retrieval")
            return []

        # collect related nodes via BFS (1 hop)
        relevant_nodes = set(query_entities)
        for entity in query_entities:
            if entity in self.graph:
                neighbours = list(self.graph.neighbors(entity)) + list(self.graph.predecessors(entity))
                relevant_nodes.update(neighbours[:20])

        # gather chunk_ids from relevant nodes
        chunk_scores: dict[str, float] = defaultdict(float)
        for node in relevant_nodes:
            for chunk_id in self._entity_chunk_map.get(node, []):
                chunk_scores[chunk_id] += 1.0

        top_chunk_ids = sorted(chunk_scores, key=chunk_scores.get, reverse=True)[:top_k]

        results = []
        for cid in top_chunk_ids:
            chunk = self._chunk_store.get(cid)
            if chunk:
                results.append(
                    RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        content=chunk.content,
                        score=chunk_scores[cid],
                        source=RouteType.VECTORLESS,
                        metadata={**chunk.metadata, "retrieval_method": "graph"},
                    )
                )
        return results

    def _save(self):
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": [{"id": n, **self.graph.nodes[n]} for n in self.graph.nodes],
            "edges": [{"src": u, "dst": v, **self.graph[u][v]} for u, v in self.graph.edges],
            "entity_chunk_map": dict(self._entity_chunk_map),
            "chunk_store": {cid: c.model_dump() for cid, c in self._chunk_store.items()},
        }
        with open(self.graph_path, "w") as f:
            json.dump(data, f)

    def _load(self):
        from src.models import Chunk
        with open(self.graph_path) as f:
            data = json.load(f)
        self.graph = nx.DiGraph()
        for node in data["nodes"]:
            nid = node.pop("id")
            self.graph.add_node(nid, **node)
        for edge in data["edges"]:
            self.graph.add_edge(edge["src"], edge["dst"], **{k: v for k, v in edge.items() if k not in ("src", "dst")})
        self._entity_chunk_map = defaultdict(list, data.get("entity_chunk_map", {}))
        self._chunk_store = {cid: Chunk(**c) for cid, c in data.get("chunk_store", {}).items()}
        logger.info(f"Loaded knowledge graph: {self.graph.number_of_nodes()} nodes")