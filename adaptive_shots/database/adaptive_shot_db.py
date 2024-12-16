import sqlite3
import sqlite_vec
import numpy as np
from typing import List, Optional, Tuple
from ..models.schemas import RegisterPromptModel, ShotPromptsList, ShotPrompt
from ..services.embedding_service import SentenceTransformerEmbedding
from .base import BaseDatabase
from ..config import DEFAULT_MODEL_NAME, DEFAULT_UCB_CONSTANT, DEFAULT_SHOTS_LIMIT

class AdaptiveShotDatabase(BaseDatabase):
    """Implementation of shot database using SQLite with vector extension."""

    def __init__(self, db_location: str, model_name: str = DEFAULT_MODEL_NAME):
        """Initialize the database connection and embedding service."""
        self._db_location = db_location
        self._conn = self._initialize_connection()
        self._embedding_service = SentenceTransformerEmbedding(model_name)

    def _initialize_connection(self) -> sqlite3.Connection:
        """Initialize SQLite connection with vector extension."""
        conn = sqlite3.connect(self._db_location)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def register_prompt(self, prompt: str, answer: str, rating: float, 
                       domain: str, used_shots: Optional[ShotPromptsList] = None) -> None:
        """Register a new prompt and update ratings of used prompts."""
        validated_data = RegisterPromptModel(
            prompt=prompt,
            answer=answer,
            rating=rating,
            domain=domain
        )

        embedding = self._embedding_service.encode(validated_data.prompt).tobytes()

        with self._conn:  # Use context manager for automatic transaction handling
            self._conn.execute('''
                INSERT INTO qa_table (
                    question, answer, domain, 
                    ratings_sum, ratings_squared_sum, ratings_count, vector
                )
                VALUES (?, ?, ?, ?, ?, ?, ?);
            ''', (
                validated_data.prompt, validated_data.answer, validated_data.domain,
                validated_data.rating, validated_data.rating**2, 1, embedding
            ))

            if used_shots:
                self._update_used_shots_ratings(used_shots, validated_data.rating)

    def _update_used_shots_ratings(self, used_shots: ShotPromptsList, rating: float) -> None:
        """Update ratings for previously used shots."""
        for shot in used_shots.prompts:
            self._conn.execute('''
                UPDATE qa_table 
                SET ratings_sum = ratings_sum + ?,
                    ratings_squared_sum = ratings_squared_sum + ?,
                    ratings_count = ratings_count + 1 
                WHERE id = ?;
            ''', (rating, rating**2, shot.id))

    def get_best_shots(self, prompt: str, domain: str, 
                      c: float = DEFAULT_UCB_CONSTANT, 
                      limit: int = DEFAULT_SHOTS_LIMIT) -> ShotPromptsList:
        """Retrieve the best matching prompts using UCB algorithm."""
        query_vector = self._embedding_service.encode(prompt).tobytes()
        
        cursor = self._conn.execute(
            'SELECT SUM(ratings_count) FROM qa_table WHERE domain = ?', 
            (domain,)
        )
        total_ratings_count = cursor.fetchone()[0] or 0

        if not total_ratings_count:
            return ShotPromptsList(prompts=[])

        results = self._fetch_best_shots(query_vector, domain, total_ratings_count, limit)
        return ShotPromptsList(prompts=[
            ShotPrompt(
                id=row[0],
                prompt=row[1],
                answer=row[2],
                domain=row[3],
                predicted_rating=row[4],
                ucb=row[5]
            ) for row in results
        ])

    def _fetch_best_shots(self, query_vector: bytes, domain: str,
                         total_ratings_count: int, limit: int) -> List[Tuple]:
        """Fetch best shots using UCB-1 Tuned formula."""
        return self._conn.execute('''
            SELECT id, question, answer, domain,
                (1 + ((ratings_sum / ratings_count - 1) * 
                    ((2 - vec_distance_cosine(vector, ?)) / 2))) AS predicted_rating,
                (1 + ((ratings_sum / ratings_count - 1) * 
                    ((2 - vec_distance_cosine(vector, ?)) / 2))) + 
                SQRT(MIN(0.25,
                    (ratings_squared_sum / ratings_count - 
                    POW(ratings_sum / ratings_count, 2) + 
                    SQRT(2 * LN(?) / ratings_count))
                ) * LN(?) / ratings_count) AS ucb
            FROM qa_table
            WHERE domain = ?
            ORDER BY ucb DESC
            LIMIT ?;
        ''', (query_vector, query_vector, total_ratings_count, 
              total_ratings_count, domain, limit)).fetchall()

    def create_one_shot_prompt(self, prompt: str, domain: str) -> Tuple[str, Optional[ShotPromptsList]]:
        """Create a one-shot prompt using the best matching record."""
        shot_list = self.get_best_shots(prompt, domain, limit=1)
        if not shot_list.prompts:
            return prompt, None

        best_shot = shot_list.prompts[0]
        return (
            f"{best_shot.prompt}\n\n{best_shot.answer}\n\nPrompt: {prompt}\n\nAnswer: ",
            shot_list
        )

    def create_few_shots_prompt(self, prompt: str, domain: str, 
                              limit: int = DEFAULT_SHOTS_LIMIT) -> Tuple[str, ShotPromptsList]:
        """Create a few-shot prompt by combining multiple relevant records."""
        shot_list = self.get_best_shots(prompt, domain, limit=limit)
        if not shot_list.prompts:
            return prompt, ShotPromptsList(prompts=[])

        few_shot_prompt = "\n\n".join(str(shot) for shot in shot_list.prompts)
        return f"{few_shot_prompt}\n\nPrompt: {prompt}\n\nAnswer: ", shot_list

def initialize_adaptive_shot_db(db_location: str, model_name: str = DEFAULT_MODEL_NAME) -> AdaptiveShotDatabase:
    """Initialize the database with required tables and return a database instance."""
    conn = sqlite3.connect(db_location)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute('''
        CREATE TABLE IF NOT EXISTS qa_table (
            id INTEGER PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            domain TEXT NOT NULL,
            ratings_sum FLOAT NOT NULL,
            ratings_count INTEGER NOT NULL,
            ratings_squared_sum FLOAT NOT NULL,
            vector BLOB NOT NULL
        );
    ''')
    conn.close()

    return AdaptiveShotDatabase(db_location, model_name)