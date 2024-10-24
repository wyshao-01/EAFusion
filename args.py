import argparse


class Args:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Your Model Configuration')
        parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
        parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
        parser.add_argument('--datasets_prepared', action='store_true', help='Flag indicating if datasets are prepared')
        parser.add_argument('--train_ir', type=str, default='./clip picture last/ms_ir', help='Path to the IR dataset')
        parser.add_argument('--train_vi', type=str, default='./clip picture last/ms_vi', help='Path to the VI dataset')
        parser.add_argument('--height', type=int, default=256, help='Height of the input images')
        parser.add_argument('--width', type=int, default=256, help='Width of the input images')
        parser.add_argument('--image_size', type=int, default=256, help='Size of the input images')
        parser.add_argument('--save_model_dir', type=str, default="models_training", help='Directory to save trained models')
        parser.add_argument('--save_loss_dir', type=str, default="loss", help='Directory to save loss plots')
        parser.add_argument('--cuda', type=int, default=1, help='Use CUDA for training')
        parser.add_argument('--g_lr', type=float, default=0.0001, help='Learning rate for the generator')
        parser.add_argument('--log_interval', type=int, default=5, help='Interval for logging')
        parser.add_argument('--log_iter', type=int, default=1, help='Interval for logging iterations')

        parser.add_argument('--s_vit_embed_dim', type=int, default=1024, help='s_vit')
        parser.add_argument('--s_vit_patch_size', type=int, default=8, help='s_vit')
        parser.add_argument('--ini_channel', type=int, default=32, help='')
        # parser.add_argument('--log_iter', type=int, default=1, help='Interval for logging iterations')
        # parser.add_argument('--log_iter', type=int, default=1, help='Interval for logging iterations')
        # parser.add_argument('--log_iter', type=int, default=1, help='Interval for logging iterations')
        # parser.add_argument('--log_iter', type=int, default=1, help='Interval for logging iterations')



        self.args = parser.parse_args()

    def get_args(self):
        return vars(self.args)