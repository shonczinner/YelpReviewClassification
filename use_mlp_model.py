import pandas as pd
from utils.encoding_utils import Text_Classification_Data, Text_Encoder
from models.model_handler import Text_Classification_Handler
from models.models import MLP_1D
k = 1000
vocab_size = k+2 # + 2 for padding token and unknown token
n_classes = 5 # reviews are {1,...,5} stars
n_epochs = 200
batch_size = 32
training_percent = 0.8
review = "The food was delicious."

data_path ="data/filtered_yelp_academic_dataset_review.json"
reviews = pd.read_json(data_path,lines=True)

dat1 = Text_Classification_Data(list(reviews['text']),list(reviews['stars']-1)) # subtract 1 to labels range from {0,...,4}
enc1 = Text_Encoder()

enc1.fit_top_k(dat1.get_text(),k)

mlp_model = MLP_1D(vocab_size,32,3,400,n_classes)

mlp_handler = Text_Classification_Handler(mlp_model)

mlp_handler.load_model("mlp_model.pt")

prediction = mlp_handler.use_model(review, enc1)
print(f"Prediction for the review '{review}': {prediction}")

