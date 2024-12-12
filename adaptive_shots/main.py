from adaptive import initialize_database

if __name__ == "__main__":
    db = initialize_database("./db/test.db")

    db.register_prompt("What is the capital of France?", "Paris", 9.5, "geography")
    db.register_prompt("Who wrote Hamlet?", "William Shakespeare", 9.0, "literature")

    user_query = "Who is the author of Romeo and Juliet?"
    one_shot, used_id = db.create_one_shot_prompt(user_query, domain="literature")

    print("Generated One-Shot Prompt:")
    print(one_shot)
