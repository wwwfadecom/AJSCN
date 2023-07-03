from __future__ import print_function, division
import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.backends import cudnn
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from utils import load_data, load_graph
from GNN import GNNLayer
from evaluation import eva
import os
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
seed = 211
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)  # cpu
torch.cuda.manual_seed_all(seed)  # gpu
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(
    description='train',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', type=str, default='usps')
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_clusters', default=3, type=int)
parser.add_argument('--n_z', default=10, type=int)
parser.add_argument('--pretrain_path', type=str, default='pkl')
parser.add_argument('--adlr', type=float, default=1e-4)
parser.add_argument('--hidden1', type=int, default=64)
parser.add_argument('--hidden2', type=int, default=16)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print("use cuda: {}".format(args.cuda))
device = torch.device("cuda" if args.cuda else "cpu")

args.pretrain_path = 'data/{}.pkl'.format(args.name)
dataset = load_data(args.name)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(10, args.hidden1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(args.hidden1, args.hidden2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(args.hidden2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity

class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z


class AJSCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1):
        super(AJSCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_z)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.XX = nn.Parameter(torch.ones([5]))
        # degree
        self.v = v
        # self.bt=nn.BatchNorm1d(n_input)

    def dot_product_decode(Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred
    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)
        # GCN Module
        h1 = self.gnn_1(x, adj)
        h2 = self.gnn_2(h1+tra1, adj)
        h3 = self.gnn_3(h2+tra2 , adj)
        h4 = self.gnn_4(h3+tra3 , adj)
        h = self.gnn_5(h4+z, adj,active=False)

        # A_pred = self.dot_product_decode()
        A_pred = torch.sigmoid(torch.matmul(h, h.t()))
        predict = F.softmax(h, dim=1)

        # Joint-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)

        q = q.pow((self.v + 1.0) / 2.0)

        q = (q.t() / torch.sum(q, 1)).t()
        s = 1.0 / (1.0 + torch.sum(torch.pow(h.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        s = s.pow((self.v + 1.0) / 2.0)
        s = (s.t() / torch.sum(s, 1)).t()
        return x_bar, q,s, predict, z, h,A_pred


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_AJSCN(dataset):
    model = AJSCN(500, 500, 2000, 2000, 500, 500,
                 n_input=args.n_input,
                 n_z=args.n_z,
                 n_clusters=args.n_clusters,
                 v=1.0).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # KNN Graph
    adj = load_graph(args.name, args.k)
    adj = adj.cuda()
    adversarial_loss = torch.nn.BCELoss()
    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    with torch.no_grad():
        _, _, _, _, z= model.ae(data)
    discriminator = Discriminator()
    discriminator = discriminator.cuda()
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.adlr, betas=(0.5, 0.999))

    Tensor = torch.cuda.FloatTensor
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'pretrain')
    for epoch in range(200):
        valid = Variable(Tensor(adj.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(adj.shape[0], 1).fill_(0.0), requires_grad=False)

        if epoch % 1 == 0:
            # update_interval
            _, tmp_q,tmp_s, pred, s, h,_ = model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
            res1 = tmp_q.cpu().numpy().argmax(1)  # Q
            eva(y, res1, str(epoch) + 'Q')
            # if(epoch==199):
            #  # max=ac
            #  tsne_plot(s.cpu().data,y)
            #  plt.savefig("/home/y/AJSCN/graph_tsn/"+args.name+"/"+"train1.pdf")

        x_bar, q,tmp_s, pred, zz, h,A_pred = model(data, adj)
        if args.name == 'wiki' or args.name == 'acm' or args.name == 'dblp':
          cels=F.binary_cross_entropy(A_pred.view(-1), adj.to_dense().view(-1))
          norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - torch._sparse_sum(adj)) * 2)
        s = target_distribution(tmp_s)
        kl_c=F.kl_div(tmp_s.log(), s, reduction='batchmean')
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(tmp_s.log(), p, reduction='batchmean')

        re_loss = F.mse_loss(x_bar, data)
        ad_loss = 0.5*adversarial_loss(discriminator(h), valid)
        if args.name=='usps' or args.name=='reut' :
         loss =kl_loss + 0.01 * ce_loss + re_loss+ad_loss*0.1+kl_c*100
        elif args.name == 'hhar':
         loss =kl_loss + 0.01 * ce_loss + re_loss+ad_loss*0.1+kl_c
        elif args.name == 'acm':
         loss = kl_loss + 0.1 * ce_loss + re_loss+ad_loss+cels*norm+kl_c*0.1
        elif args.name == 'wiki':
         loss =kl_loss + 0.01 * ce_loss + re_loss+  ad_loss + cels*norm*0.1+kl_c*0.1
        elif args.name == 'dblp':
         loss =kl_loss + 0.01 * ce_loss + re_loss + ad_loss + cels*norm +  0.1*kl_c
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truthR
        z = Variable(Tensor(np.random.normal(0, 1, (adj.shape[0], 10))))

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(h.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()
def tsne_plot(xs, xs_label,subset=True, title=None):
    xs_label = np.reshape(xs_label,(-1,1))

    num_test = 3000
    if subset:
        combined_imgs = xs[0:num_test]
        combined_labels = xs_label[0:num_test]
        combined_labels = combined_labels.astype('int')

    from sklearn.manifold import TSNE
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    source_only_tsne = tsne.fit_transform(combined_imgs)
    plt.figure(figsize=(15, 15))
    plt.scatter(source_only_tsne[:num_test, 0], source_only_tsne[:num_test, 1],#np.squeeze(combined_labels[:num_test])''''np.squeeze(combined_labels[num_test:])'''
                c=np.squeeze(combined_labels[:num_test]), s=80, marker='o', alpha=0.8, label="")
    plt.legend(loc='best')
    plt.title(title)

if __name__ == "__main__":


    if args.name == 'wiki':
        args.lr = 1e-4
        args.k = None
        args.n_clusters = 17
        args.n_input = 4973
        args.hidden1=32
        args.hidden2=64
        args.adlr=2e-5

    if args.name == 'usps':
        args.n_clusters = 10
        args.n_input = 256

    if args.name == 'hhar':
        args.k = 5
        args.n_clusters = 6
        args.n_input = 561

    if args.name == 'reut':
        args.lr = 1e-4
        args.n_clusters = 4
        args.n_input = 2000

    if args.name == 'acm':
        args.k = None
        args.n_clusters = 3
        args.n_input = 1870
        args.hidden1=32
        args.hidden2=64
        args.adlr = 2e-3


    if args.name == 'dblp':
        args.k = None
        args.n_clusters = 4
        args.n_input = 334
        args.hidden1=32
        args.hidden2=64
        args.adlr = 2e-3

    print(args)
    train_AJSCN(dataset)