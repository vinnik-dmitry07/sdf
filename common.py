from collections import Mapping

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch
import torchvision
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from tqdm import tqdm


class SignedDistanceTransform:
    def __call__(self, img_tensor):
        # Threshold.
        img_tensor[img_tensor < 0.5] = 0.
        img_tensor[img_tensor >= 0.5] = 1.

        # Compute signed distances with distance transform
        img_tensor = img_tensor.numpy()

        neg_distances = scipy.ndimage.morphology.distance_transform_edt(img_tensor)
        sd_img = img_tensor - 1.
        sd_img = sd_img.astype(np.uint8)
        signed_distances = scipy.ndimage.morphology.distance_transform_edt(sd_img) - neg_distances
        signed_distances /= float(img_tensor.shape[1])
        signed_distances = torch.tensor(signed_distances)

        return signed_distances, torch.tensor(img_tensor)


def get_mgrid(side_len):
    # Generate 2D pixel coordinates from an image of side_len x side_len
    pixel_coords = np.stack(np.mgrid[:side_len, :side_len], axis=-1)[None, ...].astype(np.float32)
    pixel_coords /= side_len
    pixel_coords -= 0.5
    pixel_coords = torch.tensor(pixel_coords).view(-1, 2)
    return pixel_coords


class MNISTSDFDataset(torch.utils.data.Dataset):
    def __init__(self, split, side_len=256):
        self.transform = transforms.Compose([
            transforms.Resize((side_len, side_len)),
            transforms.ToTensor(),
            SignedDistanceTransform(),
        ])

        self._img_dataset = torchvision.datasets.MNIST(
            root='./datasets',
            train=True if split == 'train' else False,
            download=True,
        )

        self.preprocess = False
        if self.preprocess:
            self.transformed = []
            for i in tqdm(range(len(self._img_dataset)), desc='Preprocess'):
                self.transformed.append(self.transform(self.get_image(i)))

        self.mesh_grid = get_mgrid(side_len)  # perm2[0/n-0.5, .., (n-1)/n-0.5]

    def get_image(self, index) -> torch.Tensor:
        img = self._img_dataset[index][0]
        return img

    def __len__(self):
        return len(self._img_dataset)

    def __getitem__(self, item):
        img = self.get_image(item)
        signed_distance_img, binary_img = self.transformed[item] if self.preprocess else self.transform(img)

        coord_values = self.mesh_grid.reshape(-1, 2)
        signed_distance_values = signed_distance_img.reshape((-1, 1))

        indices = torch.randperm(coord_values.shape[0])
        support_indices = indices[:indices.shape[0] // 2]
        query_indices = indices[indices.shape[0] // 2:]

        meta_dict = {
            'index': item,
            'context': {'coords': coord_values[support_indices], 'real_sdf': signed_distance_values[support_indices]},
            'query': {'coords': coord_values[query_indices], 'real_sdf': signed_distance_values[query_indices]},
            'all': {'coords': coord_values, 'real_sdf': signed_distance_values},
        }
        return meta_dict


class CIFAR10:
    def __init__(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self._img_dataset = torchvision.datasets.CIFAR10(
            root='./datasets', train=True, download=True, transform=transform
        )
        self.mesh_grid = get_mgrid(side_len=32)

    def get_norm_image(self, index) -> torch.Tensor:
        img = self._img_dataset[index][0].permute(1, 2, 0)
        return img

    @staticmethod
    def denormalize(img: torch.Tensor) -> torch.Tensor:
        img += 1
        img /= 2
        img = torch.clamp(img, 0, 1)
        return img

    def get_image(self, index) -> torch.Tensor:
        img = self.denormalize(self.get_norm_image(index))
        return img

    def __len__(self):
        return len(self._img_dataset)

    def __getitem__(self, item):
        norm_img = self.get_norm_image(item)
        norm_img_flat = norm_img.view(-1, 3)
        all_ = {'coords': self.mesh_grid, 'real_sdf': norm_img_flat}
        return {'index': item, 'context': all_, 'query': all_, 'all': all_}


# noinspection PyUnusedLocal
def sdf_loss(predictions, gt, **kwargs):
    return ((predictions - gt) ** 2).mean()


# noinspection PyUnusedLocal
def inner_maml_sdf_loss(predictions, gt, **kwargs):
    return ((predictions - gt) ** 2).sum(0).mean()


def dict_to_gpu(ob):
    if isinstance(ob, Mapping):
        return {k: dict_to_gpu(v) for k, v in ob.items()}
    else:
        return ob.cuda()


def lin2img(tensor):
    batch_size, num_samples, channels = tensor.shape
    side_len = np.sqrt(num_samples).astype(int)
    return tensor.view(batch_size, side_len, side_len, channels).squeeze(-1)


def next_step(model, dataset, epoch, step, batch_cpu, losses, get_forward_params, get_inner_preds):
    batch_gpu = dict_to_gpu(batch_cpu)

    forward_params = get_forward_params(batch_gpu)
    pred_sdf = model.forward(batch_gpu['query']['coords'], forward_params)

    loss = sdf_loss(pred_sdf, batch_gpu['query']['real_sdf'])
    losses.append(loss.item())

    if step % 100 == 0:
        tqdm.write(f'Epoch: {epoch} \t step: {step} \t loss: {loss}')

        inner_preds = []
        with torch.no_grad():
            if get_inner_preds:
                inner_preds = get_inner_preds(batch_gpu)
            final_pred = model.forward(batch_gpu['all']['coords'], forward_params)

        batch_pos = 0

        shots = [lin2img(pred)[batch_pos].detach().cpu() for pred in inner_preds + [final_pred]]

        index = batch_cpu['index'].numpy()[batch_pos]
        original_image = dataset.get_image(index)

        plt.rcParams.update({'font.size': 22, 'font.family': 'monospace'})
        cols = len(shots) + 1
        axs = plt.subplots(nrows=1, ncols=cols, figsize=(5 * cols, 6))[1]
        for i, img in enumerate(shots):
            axs[i].set_axis_off()
            axs[i].set_title(f'Shot {i}')
            if type(dataset) == MNISTSDFDataset:
                tmp = axs[i].imshow(img, cmap='seismic', vmin=-1, vmax=1)
                axs[i].contour(img, levels=[0.0], colors='black')
                if i == 0:
                    plt.colorbar(
                        mappable=tmp,
                        cax=inset_axes(axs[i], width='30%', height='3%'),
                        orientation='horizontal'
                    )
            elif type(dataset) == CIFAR10:
                axs[i].imshow(CIFAR10.denormalize(img))
            else:
                raise ValueError

        axs[-1].set_axis_off()
        axs[-1].set_title('Original')
        if type(dataset) == MNISTSDFDataset:
            axs[-1].imshow(original_image, cmap='gray_r')
        elif type(dataset) == CIFAR10:
            axs[-1].imshow(original_image)
        else:
            raise ValueError

        plt.suptitle(('train' if model.training else 'test ') + f' {loss:.5f}')
        plt.show()
    return loss
