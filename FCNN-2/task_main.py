from torch import nn, optim
import utils
import network
import torch
from pathlib import Path
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='training parameters')

#parser.add_argument('--names', default=['Aeronauticsdata'],
#                    help='which datasets (.csv filenames) to include')
parser.add_argument('--nhid1', type=int, default=80,
                    help='hidden size of layer 1 of the neural network')
parser.add_argument('--nhid2', type=int, default=40,
                    help='hidden size of layer 2 of the neural network')
parser.add_argument('--epochs', type=int, default=5000,
                    help='number of epochs')
parser.add_argument('--batch', type=int, default=50,
                    help='training batch size')
parser.add_argument('--lr', type=float, default=0.05,
                    help='learning rate')
parser.add_argument('--calc_no', default=1000,
                    help='Calculation iteration number of UnICORNN')


args = parser.parse_args()
#print(args)
a_lr = np.logspace(-4,-2,num=1000, base=10)
args.lr = round(float(a_lr[int(np.random.randint(1000))]),4)

b_n1 = np.linspace(40,80,5)
args.nhid1 = int(b_n1[int(np.random.randint(5))])

c_n2 = np.linspace(10,40,7)
args.nhid2 = int(c_n2[int(np.random.randint(7))])




ninp = 2                       # 3 input features being used (as of now)
nout = 1
batch_test = 10

trainloader, train_dataset, testloader, test_dataset = utils.get_data(args.batch,batch_test)
model = network.NeuralNet(ninp, args.nhid1, args.nhid2, nout)

objective = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
objective_test = nn.MSELoss(reduction='sum')

def test(dataloader,dataset):
    model.eval()
    loss = 0.
    with torch.no_grad():
        for i, (data, label) in enumerate(dataloader):
            output = model(data)
            loss += objective_test(output, label)
        loss /= len(dataset)
        loss = torch.sqrt(loss)
    return loss.item()

for epoch in range(args.epochs):
    model.train()
    for i, (data, label) in enumerate(trainloader):
        optimizer.zero_grad()
        output = model(data)
        loss = objective(output, label)
        loss.backward()
        optimizer.step()

    train_loss = test(trainloader,train_dataset)
    test_loss = test(testloader,test_dataset)

    Path('results').mkdir(parents=True, exist_ok=True)
    if (epoch == 0):
        f = open('results/train_log_'+ str(args.calc_no) + '.txt', 'a')
        f.write('## learning rate = ' + str(args.lr) + ', nhid1 = ' + str(args.nhid1) + ', nhid2 = ' + str(
            args.nhid2) + '\n')
        f.close()
        f = open('results/test_log_'+ str(args.calc_no) + '.txt', 'a')
        f.write('## learning rate = ' + str(args.lr) + ', nhid1 = ' + str(args.nhid1) + ', nhid2 = ' + str(
            args.nhid2) + '\n')
        f.close()

    if (epoch % 100 == 0):
        f = open('results/train_log_'+ str(args.calc_no) + '.txt', 'a')
    #    if (epoch == 0):
    #        f.write('## learning rate = ' + str(args.lr) + '\n')
        f.write(str(round(train_loss, 2)) + '\n')
        f.close()
        
        f = open('results/test_log_'+ str(args.calc_no) + '.txt', 'a')
    #    if (epoch == 0):
    #        f.write('## learning rate = ' + str(args.lr) + '\n')
        f.write(str(round(test_loss, 2)) + '\n')
        f.close()

    if ((epoch + 1)%100) == 0.:
        args.lr /= 1.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
