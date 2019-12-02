import network
import torch
import SimpleITK as sitk
import torch.optim as optim
from torch.utils import data
import numpy as np
import skimage.io as io
import os
from skimage.transform import resize
import time
from sklearn.model_selection import train_test_split

class VoxelMorph():
    """
    VoxelMorph Class is a higher level interface for both 2D and 3D
    Voxelmorph classes. It makes training easier and is scalable.
    """
    def __init__(self, input_dims, dim):
        self.dims = input_dims
        self.dim  = dim
        self.net = self.load_model()
        self.optimizer = optim.SGD(self.net.parameters(), lr=1e-4, momentum=0.99)
        self.params = {'batch_size': 3,
                       'shuffle': True,
                       'num_workers': 6,
                       'worker_init_fn': np.random.seed(42)
                       }
        self.criteria = self.cc_smooth

    def load_model(self):
        in_channel = self.dims[0] * 2
        self.net = network.VM_Net(in_channel, dim=self.dim)
        torch.cuda.set_device('cuda:0')
        self.net.cuda()
        return self.net

    def check_dims(self, x):
        try:
            if x.shape[1:] == self.dims:
                return
            else:
                raise TypeError
        except TypeError as e:
            print("Invalid Dimension Error. The supposed dimension is ",
                  self.dims, "But the dimension of the input is ", x.shape[1:])

    def forward(self, x):
        self.check_dims(x)

###### LOSSES
    def cross_correlation(self, I, J, n):
        I = I.permute(0, 3, 1, 2).cuda()
        J = J.permute(0, 3, 1, 2).cuda()
        batch_size, channels, xdim, ydim = I.shape
        I2 = torch.mul(I, I).cuda()
        J2 = torch.mul(J, J).cuda()
        IJ = torch.mul(I, J).cuda()
        sum_filter = torch.ones((1, channels, n, n)).cuda()

        I_sum = torch.conv2d(I, sum_filter, padding=1, stride=(1, 1))
        J_sum = torch.conv2d(J, sum_filter, padding=1, stride=(1, 1))

        I2_sum = torch.conv2d(I2, sum_filter, padding=1, stride=(1, 1))
        J2_sum = torch.conv2d(J2, sum_filter, padding=1, stride=(1, 1))
        IJ_sum = torch.conv2d(IJ, sum_filter, padding=1, stride=(1, 1))

        win_size = n ** 2
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size

        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + np.finfo(float).eps)

        return torch.mean(cc)

    def smooothing(self, y_pred):
        dy = torch.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :])
        dx = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])

        dx = torch.mul(dx, dx)
        dy = torch.mul(dy, dy)
        d = torch.mean(dx) + torch.mean(dy)
        return d / 2.0

    def cc_smooth(self, y, ytrue, n=9, lamda=0.01):
        cc = self.cross_correlation(y, ytrue, n)
        sm = self.smooothing(y)
        # print("CC Loss", cc, "Gradient Loss", sm)
        loss = -1.0 * cc + lamda * sm
        return loss

    def dice(self,pred, target):
        """This definition generalize to real valued pred and target vector. This should be differentiable.
        pred: tensor with first dimension as batch
        target: tensor with first dimension as batch
        """
        top = 2 * torch.sum(pred * target, [1, 2, 3])
        union = torch.sum(pred + target, [1, 2, 3])
        eps = torch.ones_like(union) * 1e-5
        bottom = torch.max(union, eps)
        dice = torch.mean(top / bottom)
        # print("Dice score", dice)
        return dice

    def mmi(self,I,J):
        I = sitk.Cast(sitk.GetImageFromArray(I), sitk.sitkFloat32)
        J = sitk.Cast(sitk.GetImageFromArray(J), sitk.sitkFloat32)

        # Hijack Simple ITK Registration method for Mattes MutualInformation metric
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMattesMutualInformation()
        MMI = R.MetricEvaluate(I, J)
        return MMI

    def mmi_smoth(self, y, ytrue, n=9, lamda=0.01):
        mmi = self.mmi(y,ytrue)
        sm = self.smoothing(y)
        loss = -1.0 * mmi + lamda * sm
        return loss

######## TRAIN MODEL
    def train_model(self, batch_moving, batch_fixed, n=9, lamda=0.01, calc_dice=False):
        # Reset Gradients
        self.optimizer.zero_grad()

        # Move images to gpu
        batch_fixed.cuda()
        batch_moving.cuda()

        # Forward Pass
        batch_registered = self.net(batch_moving, batch_fixed)

        # Calculate Loss
        train_loss = self.criteria(batch_registered, batch_fixed, n, lamda)

        # Have to figure out why batch_fixed pops off gpu -> cpu ?

        # Backward Pass
        train_loss.backward()

        # Step
        self.optimizer.step()

        # Return metrics
        if calc_dice:
            train_dice = self.dice(batch_registered, batch_fixed.cuda())
            return train_loss, train_dice
        return train_loss

######## Calculate Losses
    def get_test_loss(self, batch_moving, batch_fixed, n=9, lamda=0.01, calc_dice=False):
        with torch.set_grad_enabled(False):
            batch_moving.cuda()
            batch_fixed.cuda()
            batch_registered = self.net(batch_moving, batch_fixed)
            val_loss = self.criteria(batch_registered, batch_fixed, n, lamda)
            if calc_dice:
                val_dice_score = self.dice(batch_registered, batch_fixed.cuda())
                return val_loss, val_dice_score
            return val_loss


class Dataset(data.Dataset):
    """
    Dataset class for converting the data into batches.
    The data.Dataset class is a pyTorch class which help
    in speeding up  this process with effective parallelization
    """
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        fixed_image = torch.Tensor(resize(io.imread('./fire-fundus-image-registration-dataset/' + ID + '_1.jpg'), (256, 256, 3)))
        moving_image = torch.Tensor(resize(io.imread('./fire-fundus-image-registration-dataset/' + ID + '_2.jpg'), (256, 256, 3)))
        return fixed_image, moving_image


def main():
    '''
    In this I'll take example of FIRE: Fundus Image Registration Dataset
    to demostrate the working of the API.
    '''
    vm = VoxelMorph((3, 256, 256), dim=2)  # Object of the higher level class
    DATA_PATH = './fire-fundus-image-registration-dataset/'
    params = {'batch_size': 16,
              'shuffle': True,
              'num_workers': 6,
              'worker_init_fn': np.random.seed(42)
              }

    max_epochs = 5
    filename = list(set([x.split('_')[0]
                         for x in os.listdir('./fire-fundus-image-registration-dataset/')]))
    partition = {}
    partition['train'], partition['validation'] = train_test_split(
        filename, test_size=0.33, random_state=42)

    # Generators
    training_set = Dataset(partition['train'])
    training_generator = data.DataLoader(training_set, **params)

    validation_set = Dataset(partition['validation'])
    validation_generator = data.DataLoader(validation_set, **params)

    # Loop over epochs
    for epoch in range(max_epochs):
        start_time = time.time()
        train_loss = 0
        val_loss = 0

        # Training
        for batch_fixed, batch_moving in training_generator:
            loss = vm.train_model(batch_moving, batch_fixed)
            train_loss += loss.data
        elapsed = "{0:.2f}".format((time.time() - start_time) / 60)
        avg_loss = train_loss * params['batch_size'] / len(training_set)
        print('[', elapsed, 'mins]', epoch + 1, 'epochs, train loss = ', avg_loss)

        # Validation
        start_time = time.time()
        for batch_fixed, batch_moving in validation_generator:
            loss = vm.get_test_loss(batch_moving, batch_fixed)
            val_loss += loss.data
        elapsed = "{0:.2f}".format((time.time() - start_time) / 60)
        avg_loss = val_loss * params['batch_size'] / len(validation_set)
        print('[', elapsed, 'mins]', epoch + 1, 'epochs, val loss = ', avg_loss)


if __name__ == "__main__":
    main()
