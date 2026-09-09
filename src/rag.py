
        return sentences or [text.strip()]

    def _fixed_chunks(
        self,
        text: str,
        chunk_size: int = 60,
        overlap: int = 10,
    ) -> List[str]:

        words = text.split()

        if not words:
            return []

        chunks = []
        start = 0

        while start < len(words):
            chunk = " ".join(
                words[start:start + chunk_size]
            )

            if chunk:
                chunks.append(chunk)

            next_start = (
                start + chunk_size - overlap
            )

            if next_start <= start:
                break

            start = next_start

        return chunks

    def _chunk_text(self, text: str) -> List[str]:
        if self.strategy == "fixed":
            return self._fixed_chunks(text)

        return self._sentence_chunks(text)

    def _build_index(self):
        ids = []
        documents = []
        metadatas = []

        for topic, text in self.docs.items():
            chunks = self._chunk_text(text)

            for index, chunk in enumerate(chunks):
                ids.append(
                    f"{self.strategy}-{topic}-{index}"
                )

                documents.append(chunk)

                metadatas.append(
                    {
                        "topic": topic,
                        "chunk_id": index,
                        "strategy": self.strategy,
                    }
                )

        if not documents:
            return

        embeddings = self.model.encode(
            documents,
            normalize_embeddings=True,
        ).tolist()

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def retrieve(
        self,
        query: str,
        k: int = 3,
    ) -> List[Dict]:

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )[0].tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = results.get(
            "documents",
            [[]],
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]

        distances = results.get(
            "distances",
            [[]],
        )[0]

        hits = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            # Chroma cosine distance:
            # similarity = 1 - distance
            score = 1.0 - float(distance)

            hits.append(
                {
                    "topic": metadata["topic"],
                    "text": document,
                    "score": round(score, 4),
                    "strategy": metadata.get(
                        "strategy",
                        self.strategy,
                    ),
                }
            )

        hits.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return hits

    def answer(self, query: str):
        """
        Generate an answer only when:
        1. The request is inside the Cred support domain.
        2. A relevant local document passes the threshold.

        No external LLM or internet is used.
        """

        if not in_scope(query):
            return None, []

        hits = self.retrieve(
            query,
            k=3,
        )

        if not hits:
            return None, []

        if hits[0]["score"] < self.threshold:
            return None, []

        grounded_text = hits[0]["text"]

        answer = (
            "Based only on the local Cred Banking & "
            "FinTech support knowledge base: "
            + grounded_text
        )

        sources = [
            hit["topic"]
            for hit in hits
            if hit["score"] >= self.threshold
        ]

        return answer, sources

    def collection_name(self) -> str:
        return self.collection.name


def compare_strategies(
    query: str,
    k: int = 3,
) -> Dict:

    sentence_rag = LocalRAG(
        strategy="sentence"
    )

    fixed_rag = LocalRAG(
        strategy="fixed"
    )

    return {
        "query": query,
        "sentence": {
            "collection": (
                sentence_rag.collection_name()
            ),
            "results": sentence_rag.retrieve(
                query,
                k=k,
            ),
        },
        "fixed_overlap": {
            "collection": (
                fixed_rag.collection_name()
            ),
            "results": fixed_rag.retrieve(
                query,
                k=k,
            ),
        },
    }


if __name__ == "__main__":
    query = "What documents are required for KYC?"

    result = compare_strategies(query)

    print("RAG STRATEGY COMPARISON")
    print("=" * 60)

    print("\nSentence-based collection:")
    print(
        result["sentence"]["collection"]
    )

    for hit in result["sentence"]["results"]:
        print(
            hit["topic"],
            hit["score"],
        )

    print("\nFixed-size-overlap collection:")
    print(
        result["fixed_overlap"]["collection"]
    )

    for hit in result["fixed_overlap"]["results"]:
        print(
            hit["topic"],
            hit["score"],
        )
