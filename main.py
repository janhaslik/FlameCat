from model import predict
from model import model
from pathlib import Path

model_file = Path('model/saved_model/flamecat-model.pt')

if model_file.is_file():
    res = predict.predict("""sells company company
 """)
    print(res)
else:
    trainer = model.ModelTrainer()
