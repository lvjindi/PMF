import torch
from torch import nn
import time
import torch.utils.data
from tqdm import tqdm
import utils
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser("Lara")
parser.add_argument('--alpha', type=float, default=0, help='location of the datasets corpus')
parser.add_argument('--attr_num', type=int, default=18, help='location of the datasets corpus')
parser.add_argument('--attr_dim', type=int, default=5, help='batch size')
parser.add_argument('--batch_size', type=int, default=1024, help='init learning rate')
parser.add_argument('--hidden_dim', type=int, default=100, help='momentum')
parser.add_argument('--user_emb_dim', type=int, default=18, help='weight decay')
parser.add_argument('--lr', type=float, default=0.0001, help='weight decay')
parser.add_argument('--epochs', type=int, default=200, help='weight decay')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LaraDataset(torch.utils.data.Dataset):
    # 生成Lara数据集
    def __init__(self, train_csv, user_emb_matrix):
        self.train_csv = pd.read_csv(train_csv, header=None)
        self.user = self.train_csv.loc[:, 0]
        self.item = self.train_csv.loc[:, 1]
        self.attr = self.train_csv.loc[:, 2]
        self.user_emb_matrix = pd.read_csv(user_emb_matrix, header=None)
        self.user_emb_values = np.array(self.user_emb_matrix[:])

    def __len__(self):
        return len(self.train_csv)

    def __getitem__(self, idx):
        user = self.user[idx]
        item = self.item[idx]
        user_emb = self.user_emb_values[user]
        # 处理属性，将字符串类型转换为整数
        attr = self.attr[idx][1:-1].split()
        attr = torch.tensor(list([int(item) for item in attr]), dtype=torch.long)
        attr = np.array(attr)
        return user, item, attr, user_emb


def param_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.xavier_normal_(m.bias.unsqueeze(0))
    else:
        pass


# 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # each attribute ai of the conditional item is assigned to a special generator gi
        # g为生成器第一阶段
        self.g = nn.Embedding(2 * args.attr_num, args.attr_dim)
        # l1, l2, l3组成神经网络，为生成器第二阶段
        self.l1 = nn.Sequential(nn.Linear(args.attr_num * args.attr_dim, args.hidden_dim, bias=True), nn.Tanh())
        self.l2 = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim, bias=True), nn.Tanh())
        self.l3 = nn.Sequential(nn.Linear(args.hidden_dim, args.user_emb_dim, bias=True), nn.Tanh())
        self.__init_param__()

    def __init_param__(self):
        # 初始化参数
        for md in self.g.modules():
            torch.nn.init.xavier_normal_(md.weight)
        for md in self.modules():
            param_init(md)

    def forward(self, attr_id):
        # 通过生成器g获取属性表达
        attr_present = self.g(attr_id)
        attr_feature = torch.reshape(attr_present, [-1, args.attr_num * args.attr_dim])
        # 扔进神经网络中
        output1 = self.l1(attr_feature)
        output2 = self.l2(output1)
        output3 = self.l3(output2)
        return output3


# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.d = nn.Embedding(2 * args.attr_num, args.attr_dim)
        # l1,l2,l3构成判别器部分神经网络
        self.l1 = nn.Sequential(
            nn.Linear(args.attr_num * args.attr_dim + args.user_emb_dim, args.hidden_dim, bias=True), nn.Tanh())
        self.l2 = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim, bias=True), nn.Tanh())
        self.l3 = nn.Linear(args.hidden_dim, args.user_emb_dim, bias=True)
        self.__init_param__()

    def __init_param__(self):
        for md in self.d.modules():
            torch.nn.init.xavier_normal_(md.weight)
        for md in self.modules():
            param_init(md)

    def forward(self, attribute_id, user_emb):
        # 获取embedding
        attribute_id = attribute_id.long()
        attr_present = self.d(attribute_id)
        attr_feature = torch.reshape(attr_present, [-1, args.attr_num * args.attr_dim])
        emb = torch.cat((attr_feature, user_emb), 1)
        emb = emb.float()
        # 把获取到的embbding扔到神经网络里
        output1 = self.l1(emb)
        output2 = self.l2(output1)
        d_logit = self.l3(output2)
        # 返回判别器评分
        score = torch.sigmoid(d_logit)
        return score, d_logit


def train(g, d, train_loader, neg_loader, epochs, g_optim, d_optim, datasetLen):
    g = g.to(device)
    d = d.to(device)
    # loss 函数，BCELoss是二分类的交叉熵，需要的是经过sigmoid层的数据, 范围要求的是[0,1]之间的数
    loss = torch.nn.BCELoss()
    print('-------------------------------Begin train------------------------------')
    for i_epo in range(args.epochs):
        i = 0
        neg_iter = neg_loader.__iter__()
        # 固定生成器参数，对判别器开始训练
        d_loss_sum = 0.0
        for user, item, attr, user_emb in train_loader:
            if i * args.batch_size >= datasetLen:
                break
            # 一共有三对样本，u+，u-，uc
            # 取出负采样的样本
            _, _, neg_attr, neg_user_emb = neg_iter.next()
            # u-(negative user)
            neg_attr = neg_attr.to(device)
            neg_user_emb = neg_user_emb.to(device)
            attr = attr.to(device)
            # u+（true user）, 真实存在的用户
            user_emb = user_emb.to(device)
            # 生成器生成的用户
            fake_user_emb = g(attr)  # 根据item的属性生成用户表达
            d_real, d_logit_real = d(attr, user_emb)
            d_fake, d_logit_fake = d(attr, fake_user_emb)
            d_neg, d_logit_neg = d(neg_attr, neg_user_emb)
            # d_loss分成三部分, 正样本，生成的样本，负样本
            d_optim.zero_grad()
            # where the label of the positive pair (u+, Ic ) is set to 1, otherwise, we set it 0.
            d_loss_real = loss(d_real, torch.ones_like(d_real))
            d_loss_fake = loss(d_fake, torch.zeros_like(d_fake))
            d_loss_neg = loss(d_neg, torch.zeros_like(d_neg))
            d_loss_sum = torch.mean(d_loss_real + d_loss_fake + d_loss_neg)
            d_loss_sum.backward()
            d_optim.step()
            i += 1
        # 固定判别器参数，对生成器进行训练
        g_loss = 0.0
        for user, item, attr, user_emb in train_loader:
            g_optim.zero_grad()
            attr = attr.long()
            attr = attr.to(device)
            fake_user_emb = g(attr)
            fake_user_emb.to(device)
            d_fake, _ = d(attr, fake_user_emb)
            # 此时只用到了(Uc,Ic)样本对
            g_loss = loss(d_fake, torch.ones_like(d_fake))
            g_loss.backward()
            g_optim.step()
        print("train step%03d, d_loss:%.4f, g_loss:%.4f " % (i_epo, d_loss_sum, g_loss))
        item, attr = utils.get_test_data()
        item = item.to(device)
        attr = attr.to(device)
        item_user = g(attr)
        # 通过测试集进行测试
        infer(item, item_user)
        g_optim.zero_grad()


def infer(item, item_user):
    # 评价指标分别有，p@10,p@20,ndcg@10,ndcg@20
    p10 = utils.p_at_k(item, item_user, 10)
    p20 = utils.p_at_k(item, item_user, 20)
    ndcg_10 = utils.ndcg_k(item, item_user, 10)
    ndcg_20 = utils.ndcg_k(item, item_user, 20)
    print("test p@10:%.4f, p@20:%.4f, ndcg@10:%.4f,ndcg@20:%.4f" % (p10, p20, ndcg_10, ndcg_20))
    columns = [p10, p20, ndcg_10, ndcg_20]
    df = pd.DataFrame(columns=columns)
    df.to_csv('data/result/test_result.csv', line_terminator='\n', index=False, mode='a', encoding='utf8')


def main():
    # 加载数据集
    train_dataset = LaraDataset('data/train/train_data.csv', 'data/train/user_emb.csv')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    neg_dataset = LaraDataset('data/train/neg_data.csv', 'data/train/user_emb.csv')
    neg_loader = torch.utils.data.DataLoader(neg_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # 定义生成器
    generator = Generator()
    # 定义判别器
    discriminator = Discriminator()
    # 定义优化器
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, weight_decay=args.alpha)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, weight_decay=args.alpha)
    # 因为负样本的数据量要小一些，为了训练方便，使用负样本的长度来训练
    train(generator, discriminator, train_loader, neg_loader, args.epochs, g_optimizer, d_optimizer,
          neg_dataset.__len__())

if __name__ == '__main__':
    main()