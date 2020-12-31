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
                        load_from_disk=args.load_from_disk, data_path=args.data_path ,transform=transforms.Compose([Grayscale(), ToTensor()]))
train_data_generator = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

test_dataset = LoadDataset(dataset_name=args.dataset_name, dataset_level=args.dataset_level, num_data_per_class=args.test_num_data_per_class,
                        load_from_disk=args.load_from_disk, train=False, data_path=args.data_path,
                        transform=transforms.Compose([Grayscale(), ToTensor()]))
test_data_generator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)


model = model.to(cfg.DEVICE)
model.train()


print("Training and testing starts for :\nModel = %s\nDataset : %s\nDataset Level : %s\nTraining Num data per class : %d\n"\
            %(args.model, args.dataset_name, args.dataset_level, int(args.train_num_data_per_class)))



if args.model == 'simple_capsnet':

    for epoch_idx in range(args.epoch):
        i = 0
        j = 0

        #training
        model.train()
        train_epoch_loss = 0
        train_epoch_accuracy = 0
        for i, sample in tqdm(enumerate(train_data_generator)):

            batch_x, batch_y = sample['image'].to(cfg.DEVICE), sample['label'].to(cfg.DEVICE)

            v_lengths, decoded_images = model(batch_x, batch_y)


            total_loss = model.optimize_model(predicted=v_lengths, target=batch_y, ori_imgs=batch_x, decoded=decoded_images,
                                              loss_func=criterion, optim_func=optimizer, decay_func=lr_decayer, lr_decay_step=False)

            train_epoch_loss += total_loss

            train_epoch_accuracy += model.calculate_accuracy(v_lengths, batch_y)

        #testing
        model.eval()
        test_epoch_loss = 0
        test_epoch_accuracy = 0
        for j, sample in tqdm(enumerate(test_data_generator)):

            batch_x, batch_y = sample['image'].to(cfg.DEVICE), sample['label'].to(cfg.DEVICE)

            v_lengths, decoded_images = model(batch_x, batch_y)

            total_loss = model.calculate_loss(predicted=v_lengths, target=batch_y, ori_imgs=batch_x, decoded=decoded_images,
                                              loss_func=criterion)

            test_epoch_loss += total_loss
            test_epoch_accuracy += model.calculate_accuracy(v_lengths, batch_y)



        print(f"Mean Training Loss : {train_epoch_loss/(i + 1)}")
        print(f"Mean Training Accuracy : {train_epoch_accuracy/(i + 1)}")
        print(f"Mean Testing Loss : {test_epoch_loss/(i + 1)}")
        print(f"Mean Testing Accuracy : {test_epoch_accuracy/(i + 1)}")

    if args.mode == 'transfer_learning':
        torch.save(model.state_dict(), cfg.SIMPLE_CAPSNET_OBJECT_MODEL_PATH)
    elif args.mode is None:
        torch.save(model.state_dict(), cfg.SIMPLE_CAPSNET_FEATURE_MODEL_PATH)


elif args.model == 'simple_cnn':

    for e in range(args.epoch):

        i=0
        j=0

        train_epoch_loss = 0
        train_epoch_accuracy = 0
        test_epoch_loss = 0
        test_epoch_accuracy = 0
        model.train()
        for i, sample in tqdm(enumerate(train_data_generator)):

            batch_x, batch_y = sample['image'].to(cfg.DEVICE), sample['label'].to(cfg.DEVICE)

            net_output = model(batch_x)

            loss = model.optimize_model(predicted=net_output, target=batch_y, loss_func=criterion, optim_func=optimizer,
                        decay_func=lr_decayer)

            train_epoch_loss += loss
            train_epoch_accuracy += model.calculate_accuracy(net_output, batch_y)

        model.eval()
        for j, sample in tqdm(enumerate(test_data_generator)):

            batch_x, batch_y = sample['image'].to(cfg.DEVICE), sample['label'].to(cfg.DEVICE)

            net_output = model(batch_x)

            loss = model.calculate_loss(predicted=net_output, target=batch_y, loss_func=criterion)

            test_epoch_loss += loss
            test_epoch_accuracy += model.calculate_accuracy(predicted=net_output, target=batch_y)


        print(f"Mean Training Loss : {train_epoch_loss/(i + 1)}")
        print(f"Mean Training Accuracy : {train_epoch_accuracy/(i + 1)}")
        print(f"Mean Testing Loss : {test_epoch_loss/(i + 1)}")
        print(f"Mean Testing Accuracy : {test_epoch_accuracy/(i + 1)}")


    if args.mode == 'transfer_learning':
        torch.save(model.state_dict(), cfg.SIMPLE_CNN_OBJECT_MODEL_PATH)
    elif args.mode is None :
        torch.save(model.state_dict(), cfg.SIMPLE_CNN_FEATURE_MODEL_PATH)


