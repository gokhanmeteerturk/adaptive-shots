# tests/conftest.py
import numpy as np
from pydantic import ValidationError
import pytest
import sqlite3
import tempfile
import os
from adaptive_shots import (
    initialize_adaptive_shot_db,
    RegisterPromptModel,
    ShotPrompt,
    ShotPromptsList
)
from adaptive_shots.services.embedding_service import SentenceTransformerEmbedding


@pytest.fixture
def temp_db_path(tmp_path):
    return str(tmp_path / 'test.db')


@pytest.fixture
def db(temp_db_path):
    return initialize_adaptive_shot_db(temp_db_path)


def test_register_prompt_model_validation():
    # Valid data
    valid_data = {
        'prompt': 'Test prompt',
        'answer': 'Test answer',
        'rating': 7.5,
        'domain': 'test'
    }
    model = RegisterPromptModel(**valid_data)
    assert model.prompt == valid_data['prompt']

    # Invalid rating
    with pytest.raises(ValidationError):
        RegisterPromptModel(**{**valid_data, 'rating': 11.0})


def test_shot_prompts_list_to_messages():
    shots = ShotPromptsList(
        prompts=[
            ShotPrompt(
                id=1,
                prompt='Test prompt',
                answer='Test answer',
                domain='test',
                predicted_rating=7.5,
                ucb=8.0
            )
        ]
    )

    messages = shots.to_messages()
    assert len(messages) == 2
    assert messages[0]['role'] == 'user'
    assert messages[1]['role'] == 'assistant'


def test_sentence_transformer_embedding():
    # Because vec distance methods require float32
    service = SentenceTransformerEmbedding('all-MiniLM-L6-v2')
    text = 'Test text'
    embedding = service.encode(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.dtype == np.float32


def test_register_prompt(db):
    db.register_prompt(
        prompt='Test prompt', answer='Test answer', rating=7.5, domain='test'
    )

    shots = db.get_best_shots('Test prompt', 'test', limit=1)
    assert len(shots.prompts) == 1
    assert shots.prompts[0].prompt == 'Test prompt'


def test_get_best_shots_empty_domain(db):
    shots = db.get_best_shots('Test prompt', 'nonexistent', limit=1)
    assert len(shots.prompts) == 0


def test_create_one_shot_prompt(db):
    # Register a prompt first
    db.register_prompt(
        prompt='Very test prompt', answer='Test answer', rating=8.0, domain='test'
    )

    prompt, shot_list = db.create_one_shot_prompt('Very similar prompt', 'test')

    assert shot_list is not None
    assert len(shot_list.prompts) == 1
    assert 'Similar prompt' in prompt
    assert 'Test answer' in prompt


def test_create_few_shots_prompt(db):
    # Register multiple prompts
    for i in range(3):
        db.register_prompt(
            prompt=f"Test prompt {i}",
            answer=f"Test answer {i}",
            rating=7.5,
            domain='test'
        )
    prompt, shot_list = db.create_few_shots_prompt('New prompt', 'test', limit=3)

    assert len(shot_list.prompts) == 3
    assert all(f"Test prompt {i}" in prompt for i in range(3))


def test_update_used_shots_ratings(db):
    # Register initial prompt
    db.register_prompt(
        prompt='Initial prompt', answer='Test answer', rating=7.0, domain='test'
    )

    # Get the shot
    shots = db.get_best_shots('Initial prompt', 'test', limit=1)

    # Register new prompt with used shots
    db.register_prompt(
        prompt='New prompt',
        answer='New answer',
        rating=8.0,
        domain='test',
        used_shots=shots
    )

    # Verify ratings were updated
    updated_shots = db.get_best_shots('Initial prompt', 'test', limit=1)
    assert updated_shots.prompts[0].predicted_rating != 7.0


def test_full_workflow(db):
    # 1. Register some initial prompts
    prompts = [
        ('How to cook pasta?', 'Boil water and add pasta', 8.0, 'cooking'),
        ('How to make coffee?', 'Use coffee beans and hot water', 7.5, 'cooking'),
        ('Basic pasta recipe', 'Use pasta and sauce', 9.0, 'cooking')
    ]

    for prompt, answer, rating, domain in prompts:
        db.register_prompt(prompt, answer, rating, domain)
    # 2. Get similar prompts for a cooking question
    shots = db.get_best_shots('How to prepare spaghetti?', 'cooking', limit=2)
    assert len(shots.prompts) == 2

    # 3. Create a few-shot prompt
    prompt, shot_list = db.create_few_shots_prompt(
        'How to prepare spaghetti?', 'cooking', limit=2
    )
    assert len(shot_list.prompts) == 2

    # 4. Register new prompt with used shots
    db.register_prompt(
        'How to prepare spaghetti?',
        'Cook spaghetti in boiling water',
        9.0,
        'cooking',
        used_shots=shot_list
    )

    # 5. Verify the workflow affected the ratings
    final_shots = db.get_best_shots('How to cook pasta?', 'cooking', limit=1)
    assert len(final_shots.prompts) == 1
