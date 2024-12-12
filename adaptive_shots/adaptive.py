import sqlite3
import sqlite_vec
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional

model = SentenceTransformer('all-MiniLM-L6-v2')
VECTOR_DIMENSION = 384

class ShotPrompt:
    def __init__(self, id: int, prompt: str, answer: str, domain: str, distance: float):
        self.id = id
        self.prompt = prompt
        self.answer = answer
        self.domain = domain
        self.distance = distance

    def __str__(self):
        return f"{self.prompt}\n\n{self.answer}"

class ShotPromptsList:
    def __init__(self, prompts: List[ShotPrompt]):
        self.prompts = prompts

    def to_messages(self) -> List[dict]:
        messages = []
        for shot in self.prompts:
            messages.append({"role": "user", "content": shot.prompt})
            messages.append({"role": "assistant", "content": shot.answer})
        return messages

class AdaptiveShotDatabase:
    def __init__(self, db_location: str):
        self.conn = sqlite3.connect(db_location)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        # vec_version, = self.conn.execute("select vec_version()").fetchone()
        # print(f"vec_version={vec_version}")

    def register_prompt(self, prompt: str, answer: str, rating: float, domain: str, used_prompt_ids: Optional[List[int]] = None):
        embedding = model.encode(prompt).astype(np.float32).tobytes()

        self.conn.execute('''
        INSERT INTO qa_table (question, answer, domain, rating, vector)
        VALUES (?, ?, ?, ?, ?);
        ''', (prompt, answer, domain, rating, embedding))

        if used_prompt_ids:
            for used_id in used_prompt_ids:
                current_rating = self.conn.execute(
                    'SELECT rating FROM qa_table WHERE id = ?;', (used_id,)
                ).fetchone()[0]
                new_rating = current_rating + 0.1 * (rating - current_rating)
                self.conn.execute(
                    'UPDATE qa_table SET rating = ? WHERE id = ?;', (new_rating, used_id)
                )

        self.conn.commit()

    def get_best_shots(self, prompt: str, domain: str, limit: int = 5) -> Tuple[ShotPromptsList, List[int]]:
        """Retrieves the best matching prompts based on a vector search."""
        query_vector = model.encode(prompt).astype(np.float32).tobytes()

        cursor = self.conn.execute('''
        SELECT id, question, answer, domain, vec_distance_L2(vector, ?) AS distance
        FROM qa_table
        WHERE domain = ?
        ORDER BY distance ASC
        LIMIT ?;
        ''', (query_vector, domain, limit))

        results = cursor.fetchall()
        if not results:
            return ShotPromptsList([]), []

        prompts = [ShotPrompt(*row) for row in results]
        ids = [row[0] for row in results]
        return ShotPromptsList(prompts), ids

    def create_one_shot_prompt(self, prompt: str, domain: str) -> Tuple[str, Optional[int]]:
        """Creates a one-shot prompt by retrieving the most relevant record."""
        shot_list, ids = self.get_best_shots(prompt, domain, limit=1)
        if not shot_list.prompts:
            return prompt, None

        best_shot = shot_list.prompts[0]
        one_shot_prompt = f"{best_shot.prompt}\n\n{best_shot.answer}\n\n{prompt}"
        return one_shot_prompt, best_shot.id

    def create_few_shots_prompt(self, prompt: str, domain: str, limit: int = 3) -> Tuple[str, List[int]]:
        """Creates a few-shot prompt by combining multiple relevant records."""
        shot_list, ids = self.get_best_shots(prompt, domain, limit)
        if not shot_list.prompts:
            return prompt, []

        few_shot_prompt = "\n\n".join(str(shot) for shot in shot_list.prompts) + f"\n\n{prompt}"
        return few_shot_prompt, ids
