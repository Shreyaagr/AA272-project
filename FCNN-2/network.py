import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNet(nn.Module):
    def __init__(self, ninp, nhid1, nhid2, nout):
        super(NeuralNet, self).__init__()
#        self.flatten = nn.Flatten()
#        self.lin1 = nn.Linear(int(ninp/2), int(nhid1/2))
#        self.lin2 = nn.Linear(int(nhid1/2), int(nhid1/2))
#        self.relu = nn.ReLU()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(ninp, nhid1),
            nn.ReLU(),
	    nn.Linear(nhid1, nhid1),
            nn.ReLU(),
	    nn.Linear(nhid1, nhid1),
            nn.ReLU(),	
            nn.Linear(nhid1, nhid2),
            nn.ReLU(),
            nn.Linear(nhid2, nhid2),
            nn.ReLU(),
            nn.Linear(nhid2, nout),
        )

    def forward(self, x):
#        x = self.flatten(x)
#        out_pos = self.lin2(self.relu(self.lin2(self.relu(self.lin1(x[:,:3])))))
#        out_dop = self.lin2(self.relu(self.lin2(self.relu(self.lin1(x[:,3:])))))
#        out_comb = torch.cat((x[:,0], x[:,3]), 1)
        output = self.linear_relu_stack(x)
        return output
