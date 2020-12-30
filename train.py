'''
Training script.
'''

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from load_dataset import LoadDataset
from torchvision import transforms
from model_files.model_factory import ModelFactory
from data_transform import Grayscale, ToTensor
import config as cfg


args = cfg.args

#if any of the arguments for the model initialization needs to be changed, it can be passed as a parameter here. The factory will override the default
#parameters for the models set in the model configuration files.
model, criterion, optimizer, lr_decayer = ModelFactory().factory_call(requested_model=args.model, mode=args.mode,
                                                                      model_load_path=args.model_load_path, device=cfg.DEVICE, image_size=32)

train_dataset = LoadDataset(dataset_name=args.dataset_name, dataset_level=args.dataset_level, num_data_per_class=args.train_num_data_per_class,
                        load_from_disk=False, data_path=args.data_path ,transform=transforms.Compose([Grayscale(), ToTensor()]))
train_data_generator = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

test_dataset = LoadDataset(dataset_name=args.dataset_name, dataset_level=args.dataset_level, num_data_per_class=args.train_num_data_per_class,
                        load_from_disk=False, train=False, data_path=args.data_path ,transform=transforms.Compose([Grayscale(), ToTensor()]))
test_data_generator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)


model = model.to(cfg.DEVICE)
model.train()


print("Training and testing starts for :\nModel = %s\nDataset : %s\nDataset Level : %s\nNum data per class : %d\n"\
            %(args.model, args.dataset_name, args.dataset_level, int(args.num_data_per_class)))



if args.model == 'simple_capsnet':

    for epoch_idx in range(args.epoch):

        epoch_loss = 0

        accuracy = 0
        i = 0

        for i, sample in tqdm(enumerate(data_generator)):

            # print(sample)
            batch_x, batch_y = sample['image'].to(cfg.DEVICE), sample['label'].to(cfg.DEVICE)


            v_lengths, decoded_images = model(batch_x, batch_y)

            total_loss = model.optimize_model(v_lengths, batch_y, 0.5, batch_x, decoded_images, 0.5, criterion, optimizer, lr_decayer, cfg.DEVICE, False)

            epoch_loss += total_loss

            accuracy += model.calculate_accuracy(v_lengths, batch_y)

        print(epoch_loss)
        print("Accuracy :", accuracy/(i + 1))


else:

    for e in range(args.epoch):


        epoch_loss = 0
        accuracy = 0
        i = 0
        for i, sample in tqdm(enumerate(data_generator)):

            batch_x, batch_y = sample['image'].to(cfg.DEVICE), sample['label'].to(cfg.DEVICE)

            net_output = model(batch_x)

            loss = model.optimize_model(predicted=net_output, target=batch_y, loss_func=criterion, optim_func=optimizer,
                        decay_func=lr_decayer)

            epoch_loss += loss
            accuracy += model.calculate_accuracy(net_output, batch_y)

        print(epoch_loss)
        print("Accuracy : ", accuracy/(i + 1))

    torch.save(model.state_dict(), cfg.SIMPLE_CNN_FEATURE_MODEL_PATH)


