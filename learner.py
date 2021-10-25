import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms

from tqdm import tqdm
from PIL import Image
import os
import wandb
from data.data_pipe import get_train_loader
from model import *
from utils import *

class Face_learner(object):
    def __init__(self, args):

        self.args = args
        self.train = args.train
        self.model = Backbone(args.net_depth, args.drop_ratio, args.net_mode).to(args.device)
        if args.model_dir != '':
            state_dict = torch.load(args.model_dir)
            self.model.load_state_dict(state_dict=state_dict)
            print(f'loading model from {args.model_dir}')
        print('{}_{} model generated'.format(args.net_mode, args.net_depth))

        if not args.train:        
            self.fgnet20_best_acc, self.fgnet30_best_acc, self.agedb20_best_acc, self.agedb30_best_acc, self.lag_best_acc = 0.0, 0.0, 0.0, 0.0, 0.0
            self.data_dir2best_acc = {
                'fgnet20_child': self.fgnet20_best_acc,
                'fgnet30_child': self.fgnet30_best_acc,
                'agedb20_child': self.agedb20_best_acc,
                'agedb30_child': self.agedb30_best_acc,
                'lag': self.lag_best_acc}
            return

        if args.wandb:
            import wandb
            wandb.init(project=f"Inter-Prototype (BMVC2021)")
            wandb.run.name = args.exp

        if args.tensorboard:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(f'result/summary/{args.exp}')

        if self.args.data_mode == 'ms1m':
            self.milestones = [14, 19, 23]
            self.epoch= 25
        else:
            self.milestones = [28, 38, 46] # same milestone used in CurricularFace
            self.epoch= 50

        self.step = 0
        self.loader, self.class_num, self.child_identity = get_train_loader(args)
        self.head = Arcface(embedding_size=args.embedding_size, classnum=self.class_num, args=self.args).to(args.device)

        paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)
        self.optimizer1 = optim.SGD([
                            {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
                        ], lr = args.lr, momentum = args.momentum)

        self.optimizer2 = optim.SGD([
                            {'params': paras_only_bn}
                        ], lr = args.lr, momentum = args.momentum)


        self.board_loss_every = args.loss_freq
        self.evaluate_every = len(self.loader)
        self.save_every = args.save_freq

        print(f'total epochs: {self.epoch} || curr milestones: {self.milestones} || class num: {self.class_num}')
        print(args)

        self.fgnet20_best_acc, self.fgnet30_best_acc, self.agedb20_best_acc, self.agedb30_best_acc, self.lag_best_acc = 0.0, 0.0, 0.0, 0.0, 0.0
        self.data_dir2best_acc = {
            'fgnet20_child': self.fgnet20_best_acc,
            'fgnet30_child': self.fgnet30_best_acc,
            'agedb20_child': self.agedb20_best_acc,
            'agedb30_child': self.agedb30_best_acc,
            'lag': self.lag_best_acc}

    def preprocess_text(self, txt_root, txt_dir, data_dir=None):
        text_path = os.path.join(txt_root, txt_dir)
        lines = sorted(fixed_img_list(text_path))
        if data_dir is None:
            pairs = [' '.join(line.split(' ')[1:]) for line in lines]
            labels = [int(line.split(' ')[0]) for line in lines]
        elif data_dir == 'cacd_vs' or data_dir == 'morph':
            pairs = [' '.join(line.split(' ')[:2]) for line in lines]
            labels = [int(line.split(' ')[-1][0]) for line in lines]
        return pairs, labels


    def verification(self, net, label_list, pair_list, transform, data_dir=None):
        similarities = []
        labels = []
        assert len(label_list) == len(pair_list)

        trans_list = []
        trans_list += [transforms.ToTensor()]
        trans_list += [transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
        t = transforms.Compose(trans_list)

        if len(label_list) == 0:
            return 0, 0, 0
        net.eval()
        with torch.no_grad():
            for idx, pair in enumerate(tqdm(pair_list)):
                if data_dir is None:
                    if 'png' in pair:
                        path_1, path_2 = pair.split('.png ')
                        path_1 = path_1 + '.png'
                        path_2 = path_2[:-1]
                    elif 'jpg' in pair:
                        path_1, path_2 = pair.split('.jpg ')
                        path_1 = path_1 + '.jpg'
                        path_2 = path_2[:-1]
                    elif 'JPG' in pair:
                        path_1, path_2 = pair.split('.JPG ')
                        path_1 = path_1 + '.JPG'
                        path_2 = path_2[:-1]

                img_1 = t(Image.open(path_1)).unsqueeze(dim=0).to(self.args.device)
                img_2 = t(Image.open(path_2)).unsqueeze(dim=0).to(self.args.device)
                imgs = torch.cat((img_1, img_2), dim=0)

                # Extract feature and save
                features = net(imgs)
                similarities.append(cos_dist(features[0], features[1]).cpu())
                label = int(label_list[idx])
                labels.append(label)


        best_accr = 0.0
        best_th = 0.0

        list_th = similarities
        similarities = torch.stack(similarities, dim=0)
        labels = torch.ByteTensor(label_list)
        for i, th in enumerate(list_th):
            pred = (similarities >= th)
            correct = (pred == labels)
            accr = torch.sum(correct).item() / correct.size(0)

            if accr > best_accr:
                best_accr = accr
                best_th = th.item()

        return best_accr, best_th

    def evaluate(self):
        trans_list = []
        trans_list += [transforms.ToTensor()]
        trans_list += [transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
        t = transforms.Compose(trans_list)

        txt_root = './dataset/test/txt_files'
        # data_dir_list = ['fgnet20_child', 'fgnet30_child', 'agedb20_child', 'agedb30_child', 'lag']
        data_dir_list = ['lag', 'fgnet20_child', 'fgnet30_child', 'agedb20_child', 'agedb30_child']

        for txt_dir in data_dir_list:
            print(f'working on : {txt_dir}')
            pair_list, label_list = self.preprocess_text(txt_root, txt_dir + '.txt')
            curr_acc, curr_best_threshold = self.verification(self.model, label_list, pair_list, transform=t)
            print(f'txt_dir: {txt_dir}, curr_accr: {curr_acc}')
            if self.train:
                self.save_model(self.args, txt_dir, curr_acc)

                if curr_acc > self.data_dir2best_acc[txt_dir]:
                    self.data_dir2best_acc[txt_dir] = curr_acc
                    print(f'saving best {txt_dir} model....')
                    self.save_best_model(self.args, txt_dir, self.data_dir2best_acc[txt_dir])

                if self.args.wandb:
                    wandb.log({txt_dir + '_acc': curr_acc}, step=self.step)
                    wandb.log({txt_dir + '_best_acc': self.data_dir2best_acc[txt_dir]}, step=self.step)

                if self.args.tensorboard:
                    self.writer.add_scalar(f"acc/{txt_dir}_acc'", curr_acc, self.step)
                    self.writer.add_scalar(f"acc/{txt_dir}_best_acc'", self.data_dir2best_acc[txt_dir], self.step)

            print(f'{txt_dir} => curr acc: {curr_acc} || curr best_acc: {self.data_dir2best_acc[txt_dir]}')

    # training with inter-prototype loss
    def train_prototype(self, args):
        self.model.train()
        running_loss = 0.
        running_arcface_loss, running_child_loss, running_child_total_loss = 0.0, 0.0, 0.0
        ce_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()

        print(f'total epoch: {self.epoch}')
        for e in range(self.epoch):
            print('epoch {} started'.format(e))
            if e in self.milestones:
                self.schedule_lr()
            for imgs, labels, ages in tqdm(iter(self.loader)):
                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()
                imgs = imgs.to(args.device)
                labels = labels.to(args.device)

                embeddings = self.model(imgs)
                thetas = self.head(embeddings, labels)
                arcface_loss = ce_loss(thetas, labels)

                kernel = self.head.kernel[:, self.child_identity]
                prototype_matrix = torch.mm(l2_norm(kernel, axis=0).T, l2_norm(kernel, axis=0))

                prototype_label = torch.eye(prototype_matrix.shape[0]).to(args.device)
                child_loss = mse_loss(prototype_matrix, prototype_label)
                child_total_loss = self.args.lambda_child * child_loss

                loss = arcface_loss + child_total_loss
                loss.backward()

                running_loss += loss.item()
                running_arcface_loss += arcface_loss.item()
                running_child_loss += child_loss.item()
                running_child_total_loss += child_total_loss.item()

                self.optimizer1.step()
                self.optimizer2.step()

                del embeddings, ages
                del imgs, labels, thetas, arcface_loss

                if self.step % self.board_loss_every == 0:
                    loss_board = running_loss / self.board_loss_every
                    arcface_loss_board = running_arcface_loss / self.board_loss_every

                    child_loss_board = running_child_loss / self.board_loss_every
                    child_total_loss_board = running_child_total_loss / self.board_loss_every

                    if self.args.wandb:
                        wandb.log({
                            "train_loss": loss_board,
                            "arcface_total_loss": arcface_loss_board,
                            "child_loss": child_loss_board,
                            "child_total_loss": child_total_loss_board,
                        }, step=self.step)

                    if self.args.tensorboard:
                        self.writer.add_scalar('train_loss', loss_board, self.step)
                        self.writer.add_scalar('arcface_total_loss', arcface_loss_board, self.step)
                        self.writer.add_scalar('child_loss', child_loss_board, self.step)
                        self.writer.add_scalar('child_total_loss', child_total_loss_board, self.step)

                    running_loss = 0.
                    running_arcface_loss = 0.0
                    running_child_loss = 0.0
                    running_child_total_loss = 0.0

                if self.step % self.evaluate_every == 0  and self.step != 0:
                    self.model.eval()
                    print('evaluating....')
                    self.evaluate()
                    self.model.train()

                self.step += 1

    def schedule_lr(self):
        for params in self.optimizer1.param_groups:
            params['lr'] /= 10
        for params in self.optimizer2.param_groups:
            params['lr'] /= 10
        print(self.optimizer1)
        print(self.optimizer2)

    def save_best_model(self, args, test_dir, accuracy):
        save_path = os.path.join(args.model_path, f'best/{args.exp}/{test_dir}')
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_path, f'{test_dir}_best_model_acc_{accuracy:.3f}.pth'))
        torch.save(self.head.state_dict(), os.path.join(save_path, f'{test_dir}_best_head_acc_{accuracy:.3f}.pth'))

    def save_model(self, args, test_dir, accuracy):
        save_path = os.path.join(args.model_path, f'total/{args.exp}/{test_dir}')
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_path, f'{test_dir}_{self.step}_model_acc_{accuracy:.3f}.pth'))
        torch.save(self.head.state_dict(), os.path.join(save_path, f'{test_dir}_{self.step}_head_acc_{accuracy:.3f}.pth'))
