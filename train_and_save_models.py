import pandas as pd
from utils.encoding_utils import Text_Classification_Data, Text_Encoder, tensorize_dataset
from models.model_handler import Text_Classification_Handler
from models.models import Linear_Network, MLP_1D
import matplotlib.pyplot as plt

k = 1000
vocab_size = k+2 # + 2 for padding token and unknown token
n_classes = 5 # reviews are {1,...,5} stars
n_epochs = 200
batch_size = 32
training_percent = 0.8

data_path ="data/filtered_yelp_academic_dataset_review.json"
reviews = pd.read_json(data_path,lines=True)

dat1 = Text_Classification_Data(list(reviews['text']),list(reviews['stars']-1)) # subtract 1 to labels range from {0,...,4}
enc1 = Text_Encoder()

enc1.fit_top_k(dat1.get_text(),k)
x,y = tensorize_dataset(enc1.pad_encodings([enc1.encode(s) for s in dat1.texts]),dat1.labels)

mlp_model = MLP_1D(vocab_size,32,3,400,n_classes)
bow_model = Linear_Network(vocab_size,n_classes)

mlp_handler = Text_Classification_Handler(mlp_model,x,y,training_percent,batch_size)
bow_handler = Text_Classification_Handler(bow_model,x,y,training_percent,batch_size)


mlp_handler.train(n_epochs)
bow_handler.train(n_epochs)

mlp_handler.save_model("mlp_model.pt")
bow_handler.save_model("bow_handler.pt")

plt.plot(mlp_handler.get_epoch_losses(),label="1D Conv_MLP")
plt.plot(bow_handler.get_epoch_losses(),label="Bag of words linear model")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.savefig("images/losses.jpg")
plt.close()


plt.plot(mlp_handler.get_evaluation_accuracies(),label="1D Conv_MLP")
plt.plot(bow_handler.get_evaluation_accuracies(),label="Bag of words linear model")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("images/accuracies.jpg")
plt.close()