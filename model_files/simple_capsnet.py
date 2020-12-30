'''
Capsules Network with Dynamic Routing by Sabour, S., Frosst, N., & Hinton, G. E. (2017).
'''

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
import torch.nn as nn
from meta_classes import ModelMeta




def squash(capsules, epsilon):
        '''
        Activation function for CapsNet.
        '''

        dot_prod      = torch.sum(capsules**2, axis=-2, keepdim=True) #dot product
        scalar_factor = dot_prod/(1+dot_prod)/torch.sqrt(dot_prod+epsilon)
        squashed      = scalar_factor * capsules

        return squashed


class PrimaryCapsules(nn.Module):
    '''
    Converts the output from previously defined conv layers into capsules.
    '''

    def __init__(self, primary_caps_length, epsilon):

        super(PrimaryCapsules, self).__init__()

        self.caps_length = primary_caps_length
        self.epsilon = epsilon

    def forward(self, x):
        '''
        Transform the given feature vector to capsules and apply the squashing act. func.
        '''

        return squash(x.view(x.size(0), -1, self.caps_length, 1), self.epsilon)




class DigitCapsules(nn.Module):
    '''
    Uses the Primary Capsules for routing process and returns the output after the process.
    '''

    def __init__(self, num_capsules, input_caps_length, digit_caps_length, num_labels, routing_iter, epsilon, device):

        super(DigitCapsules, self).__init__()

        self.in_caps_length  = input_caps_length
        self.in_caps_num     = num_capsules
        self.out_caps_length = digit_caps_length
        self.out_caps_num    = num_labels
        self.routing_iter    = routing_iter
        self.epsilon = epsilon

        #Initialize the weight and bias for the transformation of capsules later
        self.weight = nn.Parameter(0.05 * torch.randn(1, self.in_caps_num, self.out_caps_num,
                                                     self.in_caps_length, self.out_caps_length))

        self.bias  = nn.Parameter(0.05 * torch.randn(1, 1, self.out_caps_num, self.out_caps_length, 1))

        self.device = device



    def forward(self, x):
        '''
        Perform the transformation of capsules + dynamic routing.
        '''
        #copy the weight [batch_size] times to perform batch operation.
        tiled_weights = self.weight.repeat(x.size(0), 1, 1, 1, 1)
        #increase the dimension of the input capsules to perform batch multiplication.
        tiled_in_caps = x[:,:, None, :, :].repeat(1,1,self.out_caps_num, 1, 1)

        u_hat = torch.matmul(torch.transpose(tiled_weights, -1, -2), tiled_in_caps).to(self.device)

        u_hat_detached = u_hat.detach() #no gradient flow

        b_ij = nn.Parameter(torch.zeros(x.size(0), x.size(1), self.out_caps_num, 1, 1),
                            requires_grad=False).to(self.device) #coefficients for dynamic routing. No gradients.


        for r_iter in range(self.routing_iter):

            c_ij = nn.functional.softmax(b_ij, dim=2)

            if r_iter == self.routing_iter - 1: #final iteration

                s_j = torch.mul(c_ij, u_hat)
                s_j = torch.sum(s_j, dim=1, keepdim=True) + self.bias
                v_j = squash(s_j, epsilon=self.epsilon)

            else:

                s_j = torch.mul(c_ij, u_hat_detached)
                s_j = torch.sum(s_j, dim=1, keepdim=True) + self.bias
                v_j = squash(s_j, epsilon=self.epsilon)
                v_j_tiled = v_j.repeat(1, self.in_caps_num, 1, 1, 1)
                product = u_hat_detached * v_j_tiled
                u_produce_v = torch.sum(product, dim=3, keepdim=True)
                b_ij = b_ij + u_produce_v

        return v_j



class ReconstructionNetwork(nn.Module):
    '''
    Used to reconstruct the image back from the final layer capsule.
    '''

    def __init__(self, num_labels, final_capsule_num, fc1_num, fc2_num, img_size, img_depth):

        super(ReconstructionNetwork, self).__init__()

        self.num_labels       = num_labels
        self.input_neuron_num = self.num_labels*final_capsule_num
        self.fc1_out_num = fc1_num
        self.fc2_in_num  = fc1_num
        self.fc2_out_num = fc2_num
        self.fc3_in_num  = fc2_num
        self.output_size = img_size*img_size*img_depth

        #3 fully connected layers with (28*28=784) sized output to represent the original image.
        self.fc_layers = nn.Sequential(
                                    nn.Linear(self.input_neuron_num, self.fc1_out_num),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(self.fc2_in_num, self.fc2_out_num),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(self.fc3_in_num, self.output_size),
                                    nn.Sigmoid() #the output will be in range of 0-1
                                    )


    def forward(self, x, y):

        #only retain the value of the capsule that corresponds to the true label.
        masked_capsule = torch.squeeze(x, dim=-1) * y.view(-1, self.num_labels, 1)

        capsule_reshaped = masked_capsule.view(y.size(0), self.input_neuron_num)

        decoded = self.fc_layers(capsule_reshaped)

        return decoded



class SimpleCapsNet(nn.Module, ModelMeta):
    '''
    Implementation of simple CapsNet.
    '''

    def __init__(self, image_size, image_depth, num_classes, learning_rate, decay_rate, primary_caps_vlength, digit_caps_vlength, epsilon, lambda_, m_plus,
                    m_minus, reg_scale, routing_iteration, device):

        '''
        Initialize parameters.
        '''
        super(SimpleCapsNet, self).__init__()
        self.image_size = image_size
        self.image_depth = image_depth
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate,
        self.primary_caps_vlength = primary_caps_vlength
        self.digit_caps_vlength = digit_caps_vlength
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.reg_scale = reg_scale
        self.routing_iteration = routing_iteration
        self.device = device

    @staticmethod
    def one_hot_encoder(labels, device, num_labels):
        '''
        Convert the given integer batches to batches of one-hot encodings.
        '''

        one_hot_zeros = torch.zeros((labels.size(0), num_labels), requires_grad=False).to(device)
        one_hot_labels = one_hot_zeros.scatter_(1, labels.view(-1,1), 1.)

        return one_hot_labels

    def build_model(self):
        '''
        Build the architecture of CapsNet.
        '''

        #Convolutional layers
        self.cnn_blocks = nn.Sequential(
                                        nn.Conv2d(self.image_depth, 128, kernel_size=3,padding=0, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(128, 256, kernel_size=5, padding=0, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256, 256, kernel_size=5, padding=0, stride=1),
                                        nn.ReLU(inplace=True)
                                        )

        self.primary_caps = PrimaryCapsules(primary_caps_length=self.primary_caps_vlength, epsilon=self.epsilon)

        self.digit_caps = DigitCapsules(num_capsules=64, input_caps_length=self.primary_caps_vlength,
                                        digit_caps_length=self.digit_caps_vlength,
                                        num_labels=self.num_classes, routing_iter=self.routing_iteration, epsilon=self.epsilon, device=self.device)

        self.recon_net = ReconstructionNetwork(num_labels=self.num_classes, final_capsule_num=self.digit_caps_vlength, fc1_num=512,
                                               fc2_num=1024, img_size=self.image_size, img_depth=self.image_depth)



    def forward(self, x, y):
        '''
        Forward pass of CapsNet.
        '''

        conv_outputs = self.cnn_blocks(x)
        # print(conv_outputs.size())
        primary_caps_layer = self.primary_caps(conv_outputs)
        # print(primary_caps_layer.size())
        digit_caps_layer = self.digit_caps(primary_caps_layer)

        digits = torch.squeeze(digit_caps_layer, dim=1)

        v_lengths = torch.sqrt(torch.sum(digits**2, dim=2, keepdim=True) + self.epsilon)

        one_hot_y = self.one_hot_encoder(y, self.device, self.num_classes).detach()

        decoded = self.recon_net(digits, one_hot_y)

        return v_lengths, decoded




    def calculate_loss(self, v_lengths, y, lambda_, ori_imgs, decoded, reg_scale, device, num_classes):
        '''
        Calculates the classification loss of the network.
        '''
        #Classification Loss

        y = self.one_hot_encoder(y, device, num_classes).detach()

        max_l = torch.max(torch.zeros(1).to(device), 0.9 - v_lengths)**2
        max_r = torch.max(torch.zeros(1).to(device), v_lengths - 0.1)**2

        max_l = max_l.view(y.size(0), -1)
        max_r = max_r.view(y.size(0), -1)

        T_c = y #label

        L_c = T_c * max_l + lambda_*(1-T_c)*max_r

        margin_loss = torch.mean(torch.sum(L_c, dim=1))

        #Reconstruction Loss

        origin = ori_imgs.view(ori_imgs.size(0), -1)

        #mean squared error between the original vector and the reconstructed vector.
        recon_loss = reg_scale * nn.MSELoss()(decoded, origin)

        return margin_loss + recon_loss



    def loss_optim_init(self, model_obj):
        '''
        Initialize the loss, optim and lr decayer functions.
        '''

        criterion = self.calculate_loss
        optim_func = torch.optim.Adam(model_obj.parameters())
        lr_decay = torch.optim.lr_scheduler.ExponentialLR(optim_func, gamma=self.decay_rate)

        return criterion, optim_func, lr_decay




    def optimize_model(self, predicted, target, lambda_, ori_imgs, decoded, reg_scale, loss_func, optim_func, decay_func, device,
                        lr_decay_step=False):
        '''
        Optimize given CapsNet model.
        '''

        optim_func.zero_grad()
        total_loss = loss_func(predicted, target, lambda_, ori_imgs, decoded, reg_scale, device, self.num_classes)

        total_loss.backward()
        optim_func.step()

        return total_loss.item()


    @staticmethod
    def calculate_accuracy(v_lengths, y):
        '''
        Calculates the classification accuracy.
        '''

        softmaxed_res = nn.Softmax(dim=1)(v_lengths) #softmax the magnitudes of the vectors.

        #get the index of the highest magnitude vector.
        pred_index = torch.argmax(softmaxed_res, 1).view(v_lengths.size(0),1)


        correct_pred = torch.eq(pred_index, y.view(y.size(0), 1))

        accuracy = torch.mean(torch.as_tensor(correct_pred).type(torch.DoubleTensor))

        return accuracy.item()

