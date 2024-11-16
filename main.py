import time
from train import train
from generate import generate
from args import Args
import utils
import torch
import os
from Models import EGNet
import pickle


flag = 0

if flag == 1:
    IS_TRAINING = True
else:
    IS_TRAINING = False

def load_model(model_path):
    G_model = EGNet()
    G_model.load_state_dict(torch.load(model_path))
    print('# generator parameters:', sum(param.numel() for param in G_model.parameters()))
    G_model.eval()
    G_model.cuda()
    return G_model

def load_text_features(text_path, image_name):
    # 加载.pkl文件中的文本特征
    with open(text_path, 'rb') as f:
        text_features = pickle.load(f)
    return text_features.get(image_name, None)


def main():
    if IS_TRAINING:
        data_dir_ir = utils.list_images(args_dict['train_ir'])
        data_dir_vi = utils.list_images(args_dict['train_vi'])

        train_data_ir = data_dir_ir
        train_data_vi = data_dir_vi


        print("\ntrain_data_ir num is ", len(train_data_ir))
        print("\ntrain_data_vi num is ", len(train_data_vi))

        train(train_data_ir, train_data_vi)

    else:
        print("\nBegin to generate pictures ...\n")

        model_name = 'G_Epoch_0.model'


        test_imgs_ir_path = "./test_imgs\msrs/ir/"#"./test_imgs/msrs/ir/"  # 修正路径分隔符
        test_imgs_vis_path ="./test_imgs\msrs/vi/"  # 修正路径分隔符
        test_text = './train_dataset_448/test_msrs_entity_orig.pkl'

        # test_imgs_ir_path = "./test_imgs\Roadscene/ir/"#"./test_imgs/msrs/ir/"  # 修正路径分隔符
        # test_imgs_vis_path ="./test_imgs\Roadscene/vis/"  # 修正路径分隔符
        # test_text = './test_imgs/Roadscene//test_msrs_entity_text_features.pkl'


        print('Model begin to test')
        result = "results"
        model_path = os.path.join(os.getcwd(), 'models_training', model_name)
        with torch.no_grad():
            model = load_model(model_path)
            model.eval()
            model.cuda()
            begin = time.time()

            # 获取所有红外图像文件名
            ir_files = os.listdir(test_imgs_ir_path)

            # 确保文件名列表是有序的
            ir_files.sort()
            # 确保文件名列表是有序的

            for ir_filename in ir_files:
                # 由于红外和可见光图像文件名完全一致，直接使用相同的文件名构建路径
                ir_path = os.path.join(test_imgs_ir_path, ir_filename)
                vis_path = os.path.join(test_imgs_vis_path, ir_filename)

                if ir_filename.endswith('.png'):
                    ir_filename = ir_filename[:-4]

                # 加载特征向量
                with open(test_text, 'rb') as f:
                    entity = pickle.load(f)[ir_filename]

                # 从文件名中提取索引
                #index = int(os.path.splitext(ir_filename)[0].split('_')[-1])

                # 假设 generate 函数接受特征向量作为参数
                generate(model, ir_path, vis_path, entity, model_path, mode='L')

            end = time.time()
            print("Consumption time of generating: %s seconds" % (end - begin))

    # else:
    #     print("\nBegin to generate pictures ...\n")
    #
    #     model_name = 'G_Epoch_19.model'
    #
    #
    #     test_imgs_ir_path = "D:/text_fusion\CrossFuse-main/test_imgs/tno/ir/"#"./test_imgs/msrs/ir/"  # 修正路径分隔符
    #     test_imgs_vis_path ="D:/text_fusion\CrossFuse-main/test_imgs/tno/vi/"  # 修正路径分隔符
    #     test_text = 'D:/text_fusion\CrossFuse-main/test_imgs/tno/entity_features_tno.pkl'
    #
    #
    #     print('Model begin to test')
    #     result = "results"
    #     model_path = os.path.join(os.getcwd(), 'models_training', model_name)
    #     with torch.no_grad():
    #         model = load_model(model_path)
    #         model.eval()
    #         model.cuda()
    #         begin = time.time()
    #
    #         # 获取所有红外图像文件名
    #         ir_files = os.listdir(test_imgs_ir_path)
    #
    #         # 确保文件名列表是有序的
    #         ir_files.sort()
    #         # 确保文件名列表是有序的
    #
    #         for ir_filename in ir_files:
    #             # 由于红外和可见光图像文件名完全一致，直接使用相同的文件名构建路径
    #             ir_path = os.path.join(test_imgs_ir_path, ir_filename)
    #             vis_path = os.path.join(test_imgs_vis_path, ir_filename)
    #
    #             if ir_filename.endswith('.png'):
    #                 ir_filename = ir_filename[:-4]
    #
    #             # 加载特征向量
    #             with open(test_text, 'rb') as f:
    #                 entity = pickle.load(f)[ir_filename]
    #
    #             # 从文件名中提取索引
    #             #index = int(os.path.splitext(ir_filename)[0].split('_')[-1])
    #
    #             # 假设 generate 函数接受特征向量作为参数
    #             generate(model, ir_path, vis_path, entity, model_path, mode='L')
    #
    #         end = time.time()
    #         print("Consumption time of generating: %s seconds" % (end - begin))







if __name__ == "__main__":
    args = Args()
    args_dict = args.get_args()
    main()