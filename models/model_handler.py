import torch 
import numpy as np
import matplotlib.pyplot as plt

# Model training/evaluating/using class
class Text_Classification_Handler:
    def __init__(self, model, x=None, y=None, training_percent=None,batch_size=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = "cpu"

        self.model = model.to(self.device)
        
        if x is not None:
            self.x = x.to(self.device)
            self.y = y.to(self.device)

            self.training_percent = training_percent
            self.batch_size = batch_size

            # Generate random indices for training data
            training_indices = np.random.choice(x.shape[0], size=int(x.shape[0] * training_percent), replace=False)

            # Create a mask that is True for test indices
            all_indices = np.arange(x.shape[0])
            test_indices = np.setdiff1d(all_indices, training_indices)

            # Select training data
            self.train_x = self.x[training_indices, :]
            self.train_y = self.y[training_indices]

            # Select test data
            self.test_x = self.x[test_indices, :]
            self.test_y = self.y[test_indices]


            self.train_dataset = torch.utils.data.TensorDataset(self.train_x,self.train_y)
            self.test_dataset = torch.utils.data.TensorDataset(self.test_x,self.test_y)

            self.train_loader = torch.utils.data.DataLoader(self.train_dataset,batch_size=batch_size,shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset,batch_size=batch_size)

            self.criterion = torch.nn.CrossEntropyLoss()
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

            self.epoch_losses = []
            self.evaluation_accuracies = []

    def train(self,epochs,evaluation_rate=1,print_loss_rate=100):
        assert self.train_x is not None, "No training data"
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            if epoch==0 or (evaluation_rate>0 and epoch%evaluation_rate==(evaluation_rate-1)):
                self.evaluate()
            self.train_epoch(print_loss_rate)
        self.evaluate()

    def train_epoch(self,print_loss_rate):
        assert self.train_x is not None, "No training data"
        epoch_loss = 0.0
        for i,(inputs,labels) in enumerate(self.train_loader):       
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            if print_loss_rate>0 and i % print_loss_rate == (print_loss_rate-1): 
                print(f'Batch {i} loss: {loss.item()}')
        self.epoch_losses.append(epoch_loss)

    def evaluate(self):
        assert self.test_x is not None, "No testing data"
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs,labels in self.test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy of the network on the test reviews: {accuracy:.2f}%')
        self.evaluation_accuracies.append(accuracy)
        self.model.train()
    
    def get_epoch_losses(self):
        return self.epoch_losses
    
    def get_evaluation_accuracies(self):
        return self.evaluation_accuracies

    def plot_training_results(self):
        plt.plot(self.epoch_losses)
        plt.xlabel("epoch")
        plt.ylabel("training loss")
        plt.show()

        plt.plot(self.evaluation_accuracies)
        plt.xlabel("evaluation #")
        plt.ylabel('accuracy')
        plt.show()

    # given an encoder and a string, encodes the string, runs the model on it and returns the output
    def use_model(self,s,encoder):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(encoder.encode(s)).to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
        self.model.train()
        prediction =  predicted.cpu().numpy()[0]
        #print(f"The string '{s}' has output: {prediction}")
        return predicted.cpu().numpy()[0]
    
    def save_model(self, path):
            torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)