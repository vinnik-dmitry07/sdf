import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common import MNISTSDFDataset, inner_maml_sdf_loss, next_step
from meta_modules import MetaSDF
from modules import ReLUFC, sal_init, sal_init_last_layer

train_dataset = MNISTSDFDataset(split='train', side_len=64)
val_dataset = MNISTSDFDataset(split='val', side_len=64)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)  # 16
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32)  # 16

hyponet = ReLUFC(in_features=2, out_features=1, num_hidden_layers=2, hidden_features=256)
hyponet.net.apply(sal_init)
hyponet.net[-1].apply(sal_init_last_layer)

model = MetaSDF(
    hyponet,
    inner_maml_sdf_loss,
    init_lr=1e-1,  # 1e-5
    num_meta_steps=3,
    first_order=False,
    lr_type='global',
).cuda()

optim = torch.optim.Adam(lr=1e-4, params=model.parameters())

writer = SummaryWriter()
train_losses = []
val_losses = []
for epoch in tqdm(range(3), desc='Epoch'):
    model.train()
    for step, batch_cpu in enumerate(tqdm(train_dataloader, desc='Train')):
        train_loss = next_step(
            model, train_dataset, epoch, step, batch_cpu, train_losses,
            get_forward_params=lambda batch_gpu: model.generate_params(batch_gpu['context'])[0],
            get_inner_preds=lambda batch_gpu: model.generate_params(batch_gpu['all'])[1],
        )

        writer.add_scalar('Loss/train', train_loss, step)

        optim.zero_grad()
        train_loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        for step, batch_cpu in enumerate(tqdm(val_dataloader, desc='Valid')):
            valid_loss = next_step(
                model, val_dataset, epoch, step, batch_cpu, val_losses,
                get_forward_params=lambda batch_gpu: model.generate_params(batch_gpu['context'])[0],
                get_inner_preds=lambda batch_gpu: model.generate_params(batch_gpu['all'])[1],
            )

            writer.add_scalar('Loss/valid', valid_loss, step)
