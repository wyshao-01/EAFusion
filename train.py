from Models import EGNet
import torch.optim as optim
from loss import g_content_loss
import numpy as np
import scipy.io as scio
from utils import make_floor
import utils
import random
import torch
import os
from tqdm import trange
from torch.autograd import Variable
import time
from args import Args
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alpha = torch.tensor([0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.7, 0.8, 0.8]).to(device)
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if alpha is None:
            # 如果没有提供alpha，初始化为全部为1的张量，长度为类别数
            self.alpha = torch.ones(9)  # 假设有10个类别
        else:
            self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 确保所有张量都在同一个设备上
        inputs, targets, self.alpha = inputs.to(self.alpha.device), targets.to(self.alpha.device), self.alpha.to(
            inputs.device)
        #print(self.alpha)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).cuda()
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        x = x.cuda()
        target = target.cuda()
        index = torch.zeros_like(x, dtype=torch.uint8).cuda()
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.FloatTensor).cuda()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)


args = Args()
args_dict = args.get_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def reset_grad(g_optimizer):

    g_optimizer.zero_grad()

def train(train_data_ir, train_data_vi):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    models_save_path = make_floor(os.getcwd(), args_dict['save_model_dir'])
    print(models_save_path)
    loss_save_path = make_floor(models_save_path,args_dict['save_loss_dir'])
    print(loss_save_path)

    G = EGNet().cuda()

    g_content_criterion = g_content_loss().cuda()

    optimizerG = optim.Adam(G.parameters(), args_dict['g_lr'])

    print("\nG_model : \n", G)


    tbar = trange(args_dict['epochs'])


    content_loss_lst = []
    all_intensity_loss_lst = []
    all_texture_loss_lst = []
    g_loss_lst = []
    all_class_loss_lst = []


    all_content_loss = 0.
    all_intensity_loss = 0.
    all_texture_loss = 0.
    all_class_loss = 0.


    for epoch in tbar:
        print('Epoch %d.....' % epoch)

        G.train()
        # scheduler.step()
        '1000张'
        # entity_features_path = './clip picture last/train_entity_oral_msrs_filtered.pkl'
        # train_label_path = './clip picture last/encoded_train_classes_oral.pkl'

        entity_features_path = 'D:/text_fusion\CrossFuse-main/train_dataset_448//train_crop_entity_orig.pkl'
        train_label_path = 'D:/text_fusion\CrossFuse-main/train_dataset_448//train_classes_label.pkl'


        batch_size=args_dict['batch_size']
        image_set_ir, image_set_vi, entity_features,train_label,batches = utils.load_dataset(
            train_data_ir, train_data_vi, entity_features_path,train_label_path, batch_size,num_imgs=None
        )

        count = 0


        for batch in range(batches):
            count +=1
            reset_grad(optimizerG)
            img_model = 'L'

            image_paths_ir = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
            image_paths_vi = image_set_vi[batch * batch_size:(batch * batch_size + batch_size)]



            img_vi = utils.get_train_images_auto_vi(image_paths_vi, height=args_dict['height'], width=args_dict['width'], mode=img_model)
            img_ir = utils.get_train_images_auto_vi(image_paths_ir, height=args_dict['height'], width=args_dict['width'], mode=img_model)

            img_ir = Variable(img_ir, requires_grad=False)
            img_vi = Variable(img_vi, requires_grad=False)

            img_ir = img_ir.cuda()
            img_vi = img_vi.cuda()

            image_paths_ir = [os.path.splitext(os.path.basename(path))[0] for path in image_paths_ir]
            image_paths_vi = [os.path.splitext(os.path.basename(path))[0] for path in image_paths_vi]

            labels_batch = torch.stack([train_label[name] for name in image_paths_ir])

            # 将标签移动到GPU
            labels_batch = labels_batch.cuda()



            text_features_dict1 = entity_features  # 第一个文本特征字典

            # 从字典中获取对应的特征并转换为PyTorch张量
            text_features_batch1 = [torch.from_numpy(text_features_dict1[os.path.basename(path)]) for path in
                                    image_paths_ir]

            # 找到最长的张量长度
            max_length = max(tensor.size(0) for tensor in text_features_batch1)


            # 对每个张量进行填充，以确保它们具有相同的长度
            padded_batch1 = [torch.nn.functional.pad(tensor, (0, 0, 0, max_length - tensor.size(0))) for tensor in
                             text_features_batch1]

            # 使用 torch.stack() 来堆叠张量
            text_features_batch1 = torch.stack(padded_batch1)

            # 将文本特征向量移动到GPU
            text_features_batch = text_features_batch1.cuda(non_blocking=True)

            img_fusion,class_output = G(img_ir, img_vi, text_features_batch)

            criterion_bce= nn.BCEWithLogitsLoss()

            criterion_focal = FocalLoss(gamma=2, alpha=alpha, reduction='mean').to(device)



            class_loss =criterion_focal(class_output,labels_batch.float())



            content_loss,  intensity_loss , texture_loss = g_content_criterion(img_ir, img_vi,img_fusion)  # models_4

            g_loss = content_loss + 1*class_loss

            all_intensity_loss += intensity_loss.item()
            all_texture_loss +=texture_loss.item()
            all_class_loss += class_loss.item()

            all_content_loss += content_loss.item()

            reset_grad(optimizerG)

            g_loss.backward()
            optimizerG.step()

            if (batch + 1) % args_dict['log_interval'] == 0:
                mesg = "{}\tepoch {}:[{}/{}]\n " \
                       "\t content_loss:{:.6}\t g_loss:{:.6}"  \
                       "\t intensity_loss:{:.6}\t  texture_loss:{:.6}\t  class_loss:{:.6}".format(
                    time.ctime(), epoch+1, count, batches,
                    all_content_loss /  args_dict['log_interval'],(all_content_loss) /  args_dict['log_interval'],
                    all_intensity_loss /  args_dict['log_interval'],  all_texture_loss / args_dict['log_interval'], all_class_loss / args_dict['log_interval']
                )
                tbar.set_description(mesg)


                content_loss_lst.append(all_content_loss /  args_dict['log_interval'])
                all_intensity_loss_lst.append(all_intensity_loss /  args_dict['log_interval'])
                all_texture_loss_lst.append(all_texture_loss /  args_dict['log_interval'])
                g_loss_lst.append((all_content_loss ) / args_dict['log_interval'])
                all_class_loss_lst.append(all_class_loss / args_dict['log_interval'])

                all_content_loss = 0.
                all_intensity_loss = 0.
                all_texture_loss = 0.
                all_class_loss = 0.

        if (epoch+1) % 1 == 0:
            # SAVE MODELS
            G.eval()
            G.cuda()
            G_save_model_filename = "G_Epoch_" + str(epoch) + ".model"
            G_model_path = os.path.join(models_save_path,G_save_model_filename)
            torch.save(G.state_dict(), G_model_path)

            # SAVE LOSS DATA

            # content_loss
            content_loss_part = np.array(content_loss_lst)
            loss_filename_path = "content_loss_epoch_" + str(epoch) + ".mat"
            save_loss_path = os.path.join(loss_save_path, loss_filename_path)
            scio.savemat(save_loss_path, {'content_loss_part': content_loss_part})

            all_intensity_loss_part = np.array(all_intensity_loss_lst)
            loss_filename_path = "all_intensity_loss_epoch_" + str(epoch) + ".mat"
            save_loss_path = os.path.join(loss_save_path, loss_filename_path)
            scio.savemat(save_loss_path, {'all_intensity_loss_part': all_intensity_loss_part})

            all_class_loss_part = np.array(all_class_loss_lst)
            loss_filename_path = "all_class_loss_epoch_" + str(epoch) + ".mat"
            save_loss_path = os.path.join(loss_save_path, loss_filename_path)
            scio.savemat(save_loss_path, {'all_class_loss_part': all_class_loss_part})

            all_texture_loss_part = np.array(all_texture_loss_lst)
            loss_filename_path = "all_texture_loss_epoch_" + str(epoch) + ".mat"
            save_loss_path = os.path.join(loss_save_path, loss_filename_path)
            scio.savemat(save_loss_path, {'all_texture_loss_part': all_texture_loss_part})

            # g_loss
            g_loss_part = np.array(g_loss_lst)
            loss_filename_path = "g_loss_epoch_" + str(epoch) + ".mat"
            save_loss_path = os.path.join(loss_save_path, loss_filename_path)
            scio.savemat(save_loss_path, {'g_loss_part': g_loss_part})

    # SAVE LOSS DATA

    # content_loss
    content_loss_total = np.array(content_loss_lst)
    loss_filename_path = "content_loss_total_epoch_" + str(epoch) + ".mat"
    save_loss_path = os.path.join(loss_save_path, loss_filename_path)
    scio.savemat(save_loss_path, {'content_loss_total': content_loss_total})

    # all_intensity_loss
    all_intensity_loss_total = np.array(all_intensity_loss_lst)
    loss_filename_path = "all_intensity_loss_total_epoch_" + str(epoch) + ".mat"
    save_loss_path = os.path.join(loss_save_path, loss_filename_path)
    scio.savemat(save_loss_path, {'all_intensity_loss_total': all_intensity_loss_total})

    # all_texture_loss
    all_texture_loss_total = np.array(all_texture_loss_lst)
    loss_filename_path = "all_texture_loss_total_epoch_" + str(epoch) + ".mat"
    save_loss_path = os.path.join(loss_save_path, loss_filename_path)
    scio.savemat(save_loss_path, {'all_texture_loss_total': all_texture_loss_total})

    #class loss
    all_class_loss_total = np.array(all_class_loss_lst)
    loss_filename_path = "all_class_loss_total_epoch_" + str(epoch) + ".mat"
    save_loss_path = os.path.join(loss_save_path, loss_filename_path)
    scio.savemat(save_loss_path, {'all_class_loss_total': all_class_loss_total})

    # g_loss
    g_loss_total = np.array(g_loss_lst)
    loss_filename_path = "g_loss_total_epoch_" + str(epoch) + ".mat"
    save_loss_path = os.path.join(loss_save_path, loss_filename_path)
    scio.savemat(save_loss_path, {'g_loss_total': g_loss_total})

    # SAVE MODELS
    G.eval()
    G.cuda()

    G_save_model_filename = "Final_G_Epoch_" + str(epoch) + ".model"
    G_model_path = os.path.join(models_save_path, G_save_model_filename)
    torch.save(G.state_dict(), G_model_path)

    print("\nDone, trained Final_G_model saved at", G_model_path)





