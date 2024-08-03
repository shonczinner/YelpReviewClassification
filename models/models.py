import torch

class Linear_Network(torch.nn.Module):
    def __init__(self, vocab_size, n_classes):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_classes = n_classes # C

        self.embedding =  torch.nn.EmbeddingBag(vocab_size, n_classes,padding_idx =0,mode='sum')
        self.embedding.weight.data.mul_(0.01)
     
        
    def forward(self, x):
        if len(x.shape) < 2:  # Single batch
            x = x.unsqueeze(0)  # Add batch dimension

        # Apply the embeddings
        x = self.embedding(x)  # B x C

        return x

class MLP_1D(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, filter_size, hidden_size, n_classes):
        super().__init__()
        self.vocab_size = vocab_size  # V
        self.embedding_size = embedding_size
        self.filter_size = filter_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes

        self.embedding = torch.nn.Embedding(vocab_size, embedding_size,padding_idx = 0)
        self.conv = torch.nn.Conv1d(in_channels=embedding_size, out_channels=hidden_size, kernel_size=filter_size, stride=1,padding=filter_size-1)
        self.hidden = torch.nn.Linear(hidden_size, n_classes)
        self.hidden.weight.data.mul_(0.01)
        self.hidden.bias.data.mul_(0.0)

    def forward(self, x):
        if len(x.shape) < 2:  # Single batch
            x = x.unsqueeze(0)  # Add batch dimension

        # Apply the embeddings
        x = self.embedding(x)  # B x T x E

        # Permute to match Conv1d expected input shape: B x E x T
        x = x.permute(0, 2, 1)  # B x E x T

        # Apply the conv layer
        x = self.conv(x)  # B x H x T

        # Apply the hidden layer
        x = torch.nn.functional.tanh(x)
        x = x.permute(0, 2, 1)  # B x T x H
        x = self.hidden(x)  # B x T x C

        # Global pooling
        #x,_= x.max(dim=1)  # B x C
        #x= x.mean(dim=1)  # B x C
        x= x.sum(dim=1)  # B x C

        return x