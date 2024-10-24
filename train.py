from Models import Generator
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

    G = Generator().cuda()

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
        entity_features_path = './entity_features.pkl'
        train_label_path = './train_label_msrs.pkl'


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
            img_ir = utils.get_train_images_auto_ir(image_paths_ir, height=args_dict['height'], width=args_dict['width'], mode=img_model)

            img_ir = Variable(img_ir, requires_grad=False)
            img_vi = Variable(img_vi, requires_grad=False)

            img_ir = img_ir.cuda()
            img_vi = img_vi.cuda()

            image_paths_ir = [os.path.splitext(os.path.basename(path))[0] for path in image_paths_ir]
            image_paths_vi = [os.path.splitext(os.path.basename(path))[0] for path in image_paths_vi]

            labels_batch = torch.stack([torch.from_numpy(train_label[name]) for name in image_paths_ir])

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
            criterion=  nn.BCEWithLogitsLoss()


            class_loss = criterion(class_output,labels_batch.float())



            content_loss,  intensity_loss , texture_loss = g_content_criterion(img_ir, img_vi,img_fusion)  # models_4

            g_loss =  class_loss #+ content_loss

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





