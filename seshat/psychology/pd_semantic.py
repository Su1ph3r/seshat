"""
Semantic analysis layer for personality disorder detection.

Provides embedding-based similarity to disorder prototypes and
topic modeling for context-aware analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Check for optional dependencies
try:
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class SemanticSimilarityResult:
    """Result of semantic similarity analysis."""
    disorder: str
    similarity_score: float  # 0.0 to 1.0
    matched_prototype: str
    confidence: float


@dataclass
class Topic:
    """Represents an extracted topic."""
    id: int
    keywords: List[str]
    weight: float
    interpretation: str = ""


@dataclass
class TopicAnalysisResult:
    """Result of topic modeling analysis."""
    topics: List[Topic]
    dominant_topic: int
    topic_distribution: List[float]
    disorder_relevance: Dict[str, float]


# Disorder prototype texts for semantic comparison
DISORDER_PROTOTYPES: Dict[str, List[str]] = {
    "paranoid": [
        "I know they are all talking about me behind my back. Everyone is plotting against me and I can't trust anyone. They are watching my every move and waiting for me to fail. People pretend to be friendly but they have hidden agendas. I have to be constantly on guard because the world is full of liars and manipulators.",
        "Someone is definitely spying on me. I catch people looking at me suspiciously all the time. My coworkers are conspiring to get me fired. Even my friends are not who they seem to be. Everyone has ulterior motives.",
        "I don't believe what anyone tells me because people always lie. They act nice to your face but stab you in the back. I've been betrayed too many times to trust again. People are fundamentally dishonest and untrustworthy.",
    ],
    "schizoid": [
        "I prefer to be alone and don't really need other people. Social interactions feel meaningless and draining. I'm perfectly content with my own company and don't understand why others need constant social contact. Emotional connections don't interest me.",
        "I don't feel much emotion about anything. Things that excite others leave me indifferent. I observe life from a distance rather than participating in it. Relationships seem like too much effort for too little reward.",
        "I have no desire for close relationships or intimacy. Being alone is not lonely, it's peaceful. I don't care what others think of me. My inner world is more interesting than the external social world.",
    ],
    "schizotypal": [
        "I sense things that others can't perceive. Sometimes I feel a presence in the room or hear whispers. I believe in telepathy and have had experiences that can't be explained by science. The universe sends me signs and messages.",
        "I often feel like things happen for a reason beyond coincidence. Numbers and patterns have special meanings. I can sometimes tell what people are thinking. Reality feels strange and dreamlike sometimes.",
        "I have unusual experiences that set me apart from others. My thoughts often go off on tangents and I lose track of what I was saying. People find me eccentric but I see connections that others miss.",
    ],
    "antisocial": [
        "Rules are made to be broken. I do what I want and don't care about consequences. Other people are tools to be used for my benefit. I've never felt guilty about anything I've done. Getting caught is the only real crime.",
        "I'm smarter than most people and can manipulate them easily. Laws don't apply to people like me. I take what I want because I deserve it. Other people's feelings are not my problem.",
        "I lie all the time because it gets me what I want. I've conned people out of money and don't feel bad about it. Suckers deserve to be taken advantage of. I never apologize because I'm never wrong.",
    ],
    "borderline": [
        "My emotions are like a rollercoaster that I can't control. One minute I love someone, the next minute I hate them. I feel empty inside and don't know who I really am. Please don't leave me, I can't survive alone.",
        "Everything is either perfect or terrible, there's no in-between. I go from feeling amazing to wanting to die within hours. My relationships are intense and chaotic. I do impulsive things I later regret.",
        "I'm terrified of being abandoned. When someone pulls away, I fall apart completely. I don't have a stable sense of self. My moods swing wildly and I can't regulate my emotions.",
    ],
    "histrionic": [
        "I love being the center of attention! Everyone should notice how fabulous I am! My emotions are so intense and dramatic! Look at me, watch me, admire me! Everything in my life is amazing or catastrophic!",
        "I need constant attention and validation. I dress provocatively to get noticed. My stories are never boring because I embellish them for effect. I cry easily and dramatically. People say I'm theatrical but I'm just passionate!",
        "I feel uncomfortable when I'm not the center of attention. I flirt with everyone to feel validated. My emotions are bigger than life! Every experience is THE BEST or THE WORST! I need everyone to like me!",
    ],
    "narcissistic": [
        "I am exceptional and superior to most people. I deserve special treatment because of my unique talents and achievements. Other people should recognize my greatness and defer to my judgment. I'm destined for extraordinary success.",
        "I'm the smartest person in any room I enter. My ideas are brilliant and others should feel privileged to hear them. People who don't appreciate me are jealous or stupid. I expect to be treated as the special person I am.",
        "I have accomplished more than anyone else I know. My needs should come first because I'm more important. Lesser people's problems are trivial compared to my concerns. I'm naturally superior and everyone should acknowledge it.",
    ],
    "avoidant": [
        "I'm too inadequate to face other people. They will judge me and find me lacking. I avoid social situations because I'll embarrass myself. I know I'm inferior to others and they can see it too.",
        "Criticism devastates me. I stay away from people to avoid rejection. I'm not good enough for relationships or success. Social situations fill me with anxiety and dread. I'd rather be alone than face humiliation.",
        "I'm terrified of being judged negatively. I know I'm awkward and people notice. I avoid new situations because I'll fail. My inadequacy is obvious to everyone. I'm too flawed to deserve acceptance.",
    ],
    "dependent": [
        "I can't make decisions without help from others. I need someone to take care of me and tell me what to do. Being alone terrifies me. I'll do anything to keep people from leaving me, even things I don't want to do.",
        "I feel helpless when I'm alone. I need constant reassurance and guidance. I can't function without someone stronger to rely on. Please don't leave me, I can't survive on my own.",
        "I agree with whatever others say because I'm afraid they'll leave me. I can't trust my own judgment. I need others to make decisions for me. I feel weak and incapable without support.",
    ],
    "obsessive_compulsive": [
        "Everything must be perfect and in order. I follow strict rules and expect others to do the same. Details matter more than the big picture. I can't delegate because no one else does things correctly.",
        "I have very high standards and won't compromise. Rules and procedures must be followed exactly. I spend so much time on details that I sometimes miss deadlines. Things have to be done the right way or not at all.",
        "Order and control are essential. I make lists and schedules for everything. Flexibility is weakness. I'm rigid about ethics and morals. Work is more important than leisure.",
    ],
}

# Topic-to-disorder mapping weights
TOPIC_DISORDER_WEIGHTS: Dict[str, Dict[str, float]] = {
    "trust_betrayal": {"paranoid": 0.8, "antisocial": 0.3, "borderline": 0.2},
    "social_isolation": {"schizoid": 0.9, "avoidant": 0.6, "schizotypal": 0.3},
    "supernatural_unusual": {"schizotypal": 0.9, "borderline": 0.1},
    "rule_breaking": {"antisocial": 0.9, "narcissistic": 0.3},
    "emotional_intensity": {"borderline": 0.8, "histrionic": 0.7, "dependent": 0.3},
    "attention_drama": {"histrionic": 0.9, "narcissistic": 0.4, "borderline": 0.2},
    "superiority_achievement": {"narcissistic": 0.9, "antisocial": 0.2},
    "inadequacy_rejection": {"avoidant": 0.9, "dependent": 0.4, "borderline": 0.3},
    "helplessness_support": {"dependent": 0.9, "avoidant": 0.3, "borderline": 0.2},
    "control_perfectionism": {"obsessive_compulsive": 0.9, "narcissistic": 0.2},
}


class PDSemanticLayer:
    """Semantic analysis using embeddings and topic modeling."""

    def __init__(
        self,
        use_embeddings: bool = True,
        use_topics: bool = True,
        model_name: Optional[str] = None,
    ):
        """
        Initialize the semantic layer.

        Args:
            use_embeddings: Whether to use embedding-based similarity
            use_topics: Whether to use topic modeling
            model_name: Sentence transformer model name (default: all-MiniLM-L6-v2)
        """
        self.use_embeddings = use_embeddings
        self.use_topics = use_topics
        self.model_name = model_name

        # Lazy-loaded components
        self._embedder = None
        self._prototype_embeddings: Dict[str, np.ndarray] = {}
        self._topic_model = None
        self._vectorizer = None

        # Check dependencies
        self._embeddings_available = False
        self._topics_available = SKLEARN_AVAILABLE

    @property
    def embedder(self):
        """Lazy load the text embedder."""
        if self._embedder is None and self.use_embeddings:
            try:
                from seshat.ml.embeddings import TextEmbedder
                self._embedder = TextEmbedder(model_name=self.model_name)
                self._embeddings_available = True
                self._embed_prototypes()
            except ImportError:
                self._embeddings_available = False
                self._embedder = False
        return self._embedder if self._embedder else None

    def _embed_prototypes(self):
        """Pre-compute embeddings for disorder prototype texts."""
        if not self.embedder:
            return

        for disorder, prototypes in DISORDER_PROTOTYPES.items():
            # Embed all prototypes and take the mean
            embeddings = self.embedder.embed(prototypes)
            self._prototype_embeddings[disorder] = np.mean(embeddings, axis=0)

    def compute_semantic_similarity(
        self,
        text: str,
    ) -> Dict[str, SemanticSimilarityResult]:
        """
        Compute semantic similarity between text and disorder prototypes.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary mapping disorder names to similarity results
        """
        if not self.embedder or not self._prototype_embeddings:
            return {}

        # Embed the input text
        text_embedding = self.embedder.embed(text)[0]

        results = {}
        for disorder, prototype_embedding in self._prototype_embeddings.items():
            # Compute cosine similarity
            similarity = float(np.dot(text_embedding, prototype_embedding) /
                             (np.linalg.norm(text_embedding) * np.linalg.norm(prototype_embedding)))

            # Normalize to 0-1 range (cosine similarity is -1 to 1)
            normalized_similarity = (similarity + 1) / 2

            # Find most similar prototype for reference
            prototypes = DISORDER_PROTOTYPES[disorder]
            prototype_sims = []
            for proto in prototypes:
                proto_emb = self.embedder.embed(proto)[0]
                proto_sim = float(np.dot(text_embedding, proto_emb) /
                                (np.linalg.norm(text_embedding) * np.linalg.norm(proto_emb)))
                prototype_sims.append(proto_sim)

            best_idx = np.argmax(prototype_sims)

            results[disorder] = SemanticSimilarityResult(
                disorder=disorder,
                similarity_score=normalized_similarity,
                matched_prototype=prototypes[best_idx][:100] + "...",
                confidence=max(0, normalized_similarity - 0.3) / 0.7,  # Scale confidence
            )

        return results

    def extract_topics(
        self,
        text: str,
        n_topics: int = 5,
        method: str = "lda",
    ) -> TopicAnalysisResult:
        """
        Extract topics from text using topic modeling.

        Args:
            text: Input text to analyze
            n_topics: Number of topics to extract
            method: Topic modeling method ("lda" or "nmf")

        Returns:
            TopicAnalysisResult with extracted topics and disorder relevance
        """
        if not self._topics_available:
            return self._empty_topic_result()

        # Split text into sentences/chunks for topic modeling
        sentences = self._split_into_chunks(text, min_length=50)

        if len(sentences) < 3:
            # Not enough text for meaningful topic modeling
            return self._empty_topic_result()

        # Vectorize the text
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            return self._empty_topic_result()

        # Adjust n_topics based on available documents
        n_topics = min(n_topics, len(sentences) - 1, tfidf_matrix.shape[1])
        if n_topics < 2:
            return self._empty_topic_result()

        # Apply topic modeling
        if method == "lda":
            model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10,
            )
        else:
            model = NMF(
                n_components=n_topics,
                random_state=42,
                max_iter=200,
            )

        try:
            doc_topics = model.fit_transform(tfidf_matrix)
        except Exception:
            return self._empty_topic_result()

        # Extract topic keywords
        feature_names = vectorizer.get_feature_names_out()
        topics = []

        for topic_idx, topic in enumerate(model.components_):
            top_word_indices = topic.argsort()[:-11:-1]
            keywords = [feature_names[i] for i in top_word_indices]
            weight = float(np.mean(doc_topics[:, topic_idx]))

            topics.append(Topic(
                id=topic_idx,
                keywords=keywords,
                weight=weight,
                interpretation=self._interpret_topic(keywords),
            ))

        # Get overall topic distribution
        topic_distribution = list(np.mean(doc_topics, axis=0))
        dominant_topic = int(np.argmax(topic_distribution))

        # Calculate disorder relevance based on topic keywords
        disorder_relevance = self._calculate_disorder_relevance(topics)

        return TopicAnalysisResult(
            topics=topics,
            dominant_topic=dominant_topic,
            topic_distribution=topic_distribution,
            disorder_relevance=disorder_relevance,
        )

    def get_topic_disorder_weights(
        self,
        topic_result: TopicAnalysisResult,
    ) -> Dict[str, float]:
        """
        Convert topic analysis to disorder weight adjustments.

        Args:
            topic_result: Result from extract_topics()

        Returns:
            Dictionary of disorder -> weight adjustment
        """
        if not topic_result.topics:
            return {}

        # Use the pre-calculated disorder relevance
        return topic_result.disorder_relevance

    def analyze(
        self,
        text: str,
    ) -> Dict[str, Any]:
        """
        Perform full semantic analysis.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with semantic similarity and topic analysis results
        """
        results = {
            "semantic_similarity": {},
            "topics": None,
            "topic_disorder_weights": {},
        }

        # Embedding-based similarity
        if self.use_embeddings and self.embedder:
            results["semantic_similarity"] = {
                d: {
                    "similarity_score": r.similarity_score,
                    "confidence": r.confidence,
                }
                for d, r in self.compute_semantic_similarity(text).items()
            }

        # Topic modeling
        if self.use_topics and self._topics_available:
            topic_result = self.extract_topics(text)
            results["topics"] = {
                "topics": [
                    {
                        "id": t.id,
                        "keywords": t.keywords[:5],
                        "weight": t.weight,
                        "interpretation": t.interpretation,
                    }
                    for t in topic_result.topics
                ],
                "dominant_topic": topic_result.dominant_topic,
                "topic_distribution": topic_result.topic_distribution,
            }
            results["topic_disorder_weights"] = topic_result.disorder_relevance

        return results

    def _split_into_chunks(self, text: str, min_length: int = 50) -> List[str]:
        """Split text into sentence-like chunks."""
        import re
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Filter short sentences
        return [s.strip() for s in sentences if len(s.strip()) >= min_length]

    def _interpret_topic(self, keywords: List[str]) -> str:
        """Generate interpretation for topic keywords."""
        # Map common keyword patterns to interpretations
        keyword_set = set(kw.lower() for kw in keywords)

        if keyword_set & {"trust", "betray", "lie", "deceive", "watch", "spy"}:
            return "trust_betrayal"
        elif keyword_set & {"alone", "isolat", "social", "people", "friend"}:
            return "social_isolation"
        elif keyword_set & {"feel", "sense", "energy", "spirit", "psychic"}:
            return "supernatural_unusual"
        elif keyword_set & {"rule", "law", "break", "don't care", "want"}:
            return "rule_breaking"
        elif keyword_set & {"emotion", "feel", "love", "hate", "intense"}:
            return "emotional_intensity"
        elif keyword_set & {"attention", "look", "notice", "amazing", "dramatic"}:
            return "attention_drama"
        elif keyword_set & {"best", "superior", "deserve", "special", "great"}:
            return "superiority_achievement"
        elif keyword_set & {"afraid", "reject", "inadequate", "inferior", "fail"}:
            return "inadequacy_rejection"
        elif keyword_set & {"help", "need", "alone", "can't", "support"}:
            return "helplessness_support"
        elif keyword_set & {"perfect", "control", "order", "rule", "must"}:
            return "control_perfectionism"
        else:
            return "general"

    def _calculate_disorder_relevance(
        self,
        topics: List[Topic],
    ) -> Dict[str, float]:
        """Calculate disorder relevance from topics."""
        relevance = {
            "paranoid": 0.0,
            "schizoid": 0.0,
            "schizotypal": 0.0,
            "antisocial": 0.0,
            "borderline": 0.0,
            "histrionic": 0.0,
            "narcissistic": 0.0,
            "avoidant": 0.0,
            "dependent": 0.0,
            "obsessive_compulsive": 0.0,
        }

        for topic in topics:
            interpretation = topic.interpretation
            if interpretation in TOPIC_DISORDER_WEIGHTS:
                weights = TOPIC_DISORDER_WEIGHTS[interpretation]
                for disorder, weight in weights.items():
                    relevance[disorder] += topic.weight * weight

        # Normalize to 0-1 range
        max_relevance = max(relevance.values()) if relevance.values() else 1
        if max_relevance > 0:
            relevance = {k: min(1.0, v / max_relevance) for k, v in relevance.items()}

        return relevance

    def _empty_topic_result(self) -> TopicAnalysisResult:
        """Return empty topic result."""
        return TopicAnalysisResult(
            topics=[],
            dominant_topic=-1,
            topic_distribution=[],
            disorder_relevance={},
        )

    def is_available(self) -> Dict[str, bool]:
        """Check which features are available."""
        # Trigger lazy loading to check availability
        _ = self.embedder

        return {
            "embeddings": self._embeddings_available,
            "topics": self._topics_available,
        }
