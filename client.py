import banana_dev as banana
import os

api_key = "489940d7-bb23-4f9a-862b-26d166f97d34"#os.environ.get("API_KEY")
model_key = "524c2d04-b2d0-467d-9127-7a4fe4b2a0f9"#os.environ.get("MODEL_KEY")

model_inputs = {'prompt': 'Je suis un jeune étudiant en'}

model_outputs = banana.run(api_key, model_key, model_inputs)

print(model_outputs)
