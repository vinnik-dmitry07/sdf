from collections import Mapping

import PIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from matplotlib import image
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import distance_transform_edt
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from tqdm import tqdm

from render import coords_to_image


class SignedDistanceTransform:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, img_tensor):
        img_tensor[img_tensor < 0.5] = 0.
        img_tensor[img_tensor >= 0.5] = 1.

        img_tensor = img_tensor.numpy()  # type doesnt matter {0, 1}
        neg_distances = distance_transform_edt(img_tensor)
        sd_img = img_tensor - 1.
        sd_img = sd_img.astype(np.uint8)
        signed_distances = distance_transform_edt(sd_img) - neg_distances
        signed_distances /= img_tensor.shape[1]

        return (
            torch.tensor(signed_distances, dtype=self.dtype),
            torch.tensor(img_tensor, dtype=self.dtype)
        )


def get_mgrid(side_len, dtype):
    # Generate 2D pixel coordinates from an image of side_len x side_len
    pixel_coords = np.stack(np.mgrid[:side_len, :side_len], axis=-1)[None, ...].astype(np.float64)
    pixel_coords /= side_len
    pixel_coords -= 0.5
    pixel_coords = torch.tensor(pixel_coords, dtype=dtype).view(-1, 2)
    return pixel_coords


class MNISTSDFDataset(torch.utils.data.Dataset):
    def __init__(self, split, dtype, side_len):
        self.transform = transforms.Compose([
            transforms.Resize((side_len, side_len)),
            transforms.ToTensor(),  # float32, adds [1, ..] dim
            SignedDistanceTransform(dtype),
        ])

        self._img_dataset = torchvision.datasets.MNIST(
            root='../datasets',
            train=split == 'train',
            download=True,
        )

        self.preprocess = False
        if self.preprocess:
            self.transformed = []
            for i in tqdm(range(len(self._img_dataset)), desc='Preprocess'):
                self.transformed.append(self.transform(self.get_image(i)))

        self.mesh_grid = get_mgrid(side_len, dtype)  # perm2[0/n-0.5, .., (n-1)/n-0.5]

    def get_image(self, index) -> PIL.Image:
        img = self._img_dataset[index][0]
        return img

    def __len__(self):
        return len(self._img_dataset)

    def __getitem__(self, item):
        coords = self.mesh_grid.reshape(-1, 2)

        img = self.get_image(item)
        signed_distance_img, binary_img = self.transformed[item] if self.preprocess else self.transform(img)
        real_sdf = signed_distance_img.reshape((-1, 1))

        indices = torch.randperm(coords.shape[0])
        context_indices = indices[:indices.shape[0] // 2]
        query_indices = indices[indices.shape[0] // 2:]

        surface_coords = self.mesh_grid[binary_img.reshape(-1) == 1]
        idx = torch.randperm(surface_coords.size(0))[:200]
        surface_coords = surface_coords[idx]
        surface_sdf = torch.zeros((surface_coords.shape[0], 1))

        meta_dict = {
            'index': item,
            'context': {'coords': coords[context_indices], 'real_sdf': real_sdf[context_indices]},
            'query': {'coords': coords[query_indices], 'real_sdf': real_sdf[query_indices]},
            'surface': {'coords': surface_coords, 'real_sdf': surface_sdf},
            'all': {'coords': coords, 'real_sdf': real_sdf},
        }
        return meta_dict


class CIFAR10:
    def __init__(self, split, dtype):
        self.dtype = dtype
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # float32, adds [1, ..] dim
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self._img_dataset = torchvision.datasets.CIFAR10(
            root='../datasets',
            train=split == 'train',
            download=True,
        )
        self.mesh_grid = get_mgrid(side_len=32, dtype=dtype)

    @staticmethod
    def denormalize(img: torch.Tensor) -> torch.Tensor:
        img += 1
        img /= 2
        img = torch.clamp(img, 0, 1)
        return img

    def get_image(self, index) -> PIL.Image:
        img = self._img_dataset[index][0]
        return img

    def get_norm_image(self, index) -> torch.Tensor:
        img = torch.tensor(self.transform(self.get_image(index)), dtype=self.dtype)
        return img

    def __len__(self):
        return len(self._img_dataset)

    def __getitem__(self, item):
        coords = self.mesh_grid.reshape(-1, 2)

        norm_img = self.get_norm_image(item)
        norm_img_flat = norm_img.view(-1, 3)

        indices = torch.randperm(coords.shape[0])
        context_indices = indices[:indices.shape[0] // 2]
        query_indices = indices[indices.shape[0] // 2:]

        meta_dict = {
            'index': item,
            'context': {'coords': coords[context_indices], 'real_sdf': norm_img_flat[context_indices]},
            'query': {'coords': coords[query_indices], 'real_sdf': norm_img_flat[query_indices]},
            'all': {'coords': coords, 'real_sdf': norm_img_flat},
        }
        return meta_dict


class ShapeNetDataset(torch.utils.data.Dataset):
    def __init__(self, split, dtype):
        self.dtype = dtype

        self.store = pd.HDFStore('../datasets/shapenet_cropped.h5', 'r')
        all_keys = [k.lstrip('/') for k in self.store.keys()]
        self.keys = train_test_split(all_keys, train_size=0.8, shuffle=False)[0 if split == 'train' else 1]

    def get_image(self, index) -> torch.Tensor:
        img = image.imread(f'../datasets/points/{self.keys[index]}.jpg')
        return img

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        df = self.store[self.keys[item]]
        coords = torch.tensor(df.iloc[:, 0:3].values, dtype=self.dtype)
        real_sdf = torch.tensor(df.iloc[:, 3].values, dtype=self.dtype).unsqueeze(1)

        indices = torch.randperm(coords.shape[0])
        context_indices = indices[:indices.shape[0] // 2]
        query_indices = indices[indices.shape[0] // 2:]

        meta_dict = {
            'index': item,
            'context': {'coords': coords[context_indices], 'real_sdf': real_sdf[context_indices]},
            'query': {'coords': coords[query_indices], 'real_sdf': real_sdf[query_indices]},
            'all': {'coords': coords, 'real_sdf': real_sdf},
        }
        return meta_dict


# noinspection PyUnusedLocal
def l2_loss(predictions, gt, sigma=None):
    return ((predictions - gt) ** 2).mean()


# noinspection PyUnusedLocal
def l2_batch_loss(predictions, gt, sigma=None):
    batch_loss = ((predictions - gt) ** 2).sum(0).mean()  # sum along batch
    return batch_loss


# noinspection PyUnusedLocal
def l1_batch_loss(predictions, gt, sigma=None):
    batch_loss = torch.abs(predictions - gt).sum(0).mean()  # sum along batch
    return batch_loss


def multitask_loss(pred_input, gt_sdf, sigma):
    assert not torch.any(gt_sdf == -1.)
    gt_sign = (gt_sdf > 0).float()
    pred_sign = torch.sigmoid(pred_input[:, :, 0:1])  # : - to save [1, N, 1] last dim
    pred_sdf = pred_input[:, :, 1:2]

    # Binary Cross Entropy: y*log(x) + (1 - y)*log(1 - x)
    bce_loss = torch.nn.BCELoss(reduction='none')(pred_sign, gt_sign).sum(0).mean()  # sum along batch
    l1_loss_ = l1_batch_loss(pred_sdf, gt_sdf)

    batch_loss = bce_loss / (2 * sigma[0] ** 2) + l1_loss_ / (2 * sigma[1] ** 2) + torch.log(sigma.prod())
    return batch_loss


def dict_to_gpu(ob):
    if isinstance(ob, Mapping):
        return {k: dict_to_gpu(v) for k, v in ob.items()}
    else:
        return ob.cuda()


def lin2img(tensor):
    batch_size, num_samples, channels = tensor.shape
    side_len = np.sqrt(num_samples).astype(int)
    return tensor.view(batch_size, side_len, side_len, channels).squeeze(-1)


def next_step(
        model, dataset, epoch, step, batch_cpu, losses, get_context_params, has_inner_preds,
        log_every=100, batch_pos=0, test=False,
):
    batch_gpu = dict_to_gpu(batch_cpu)

    context_params = get_context_params(batch_gpu)
    pred_sdf = model.forward(batch_gpu['query']['coords'], context_params)

    loss = l2_loss(pred_sdf, batch_gpu['query']['real_sdf'])
    losses.append(loss.item())

    if step % log_every == 0:
        assert test != has_inner_preds
        tqdm.write(f'Epoch: {epoch} \t step: {step} \t loss: {loss}')

        inner_preds = []
        with torch.no_grad():
            if has_inner_preds:
                inner_preds = model.generate_params(batch_gpu['all'])[1]
            final_pred = model.forward(
                batch_gpu['surface']['coords'] if test else batch_gpu['all']['coords'],
                context_params
            )
        all_preds = inner_preds + [final_pred]

        plt.rcParams.update({'font.size': 22, 'font.family': 'monospace'})
        cols = len(all_preds) + 1
        axs = plt.subplots(nrows=1, ncols=cols, figsize=(5 * cols, 6))[1]
        for i, pred in enumerate(all_preds):
            axs[i].set_axis_off()
            axs[i].set_title(f'Shot {i}')

            if type(dataset) == MNISTSDFDataset:
                img = lin2img(pred)[batch_pos].detach().cpu()
                tmp = axs[i].imshow(img, cmap='seismic', vmin=-1, vmax=1)
                axs[i].contour(img, levels=[0.0], colors='black')
                if i == 0:
                    plt.colorbar(
                        mappable=tmp,
                        cax=inset_axes(axs[i], width='30%', height='3%'),
                        orientation='horizontal'
                    )
            elif type(dataset) == CIFAR10:
                img = lin2img(pred)[batch_pos].detach().cpu()
                axs[i].imshow(CIFAR10.denormalize(img))
            elif type(dataset) == ShapeNetDataset:
                sdf = pred[:, :, -1].squeeze()
                coords = batch_gpu['all']['coords'].squeeze().detach().cpu()
                coords = coords[sdf <= 0]
                if len(coords):
                    axs[i].imshow(coords_to_image(coords))
            else:
                raise TypeError

        index = batch_cpu['index'].numpy()[batch_pos]
        original_image = dataset.get_image(index)

        axs[-1].set_axis_off()
        axs[-1].set_title('Original')
        if type(dataset) == MNISTSDFDataset:
            axs[-1].imshow(original_image, cmap='gray_r')
        elif type(dataset) == CIFAR10 or type(dataset) == ShapeNetDataset:
            axs[-1].imshow(original_image)
        else:
            raise ValueError

        plt.suptitle(('train' if model.training else 'test ') + f' {loss:.5f}')
        plt.show()
    return loss
