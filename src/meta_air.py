import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common import next_step, ShapeNetDataset, multitask_hypo_loss, multitask_hyper_loss, l2_hypo_loss, l1_hypo_loss, \
    l1_hyper_loss
from meta_modules import MetaSDF
from modules import ReLUFC, sal_init, sal_init_last_layer, PEFC

dtype = torch.float32
torch.set_default_dtype(dtype)
train_dataset = ShapeNetDataset('/home/share/datasets/shapenet_manifold_v2.h5', '02691156', split='train', dtype=dtype)
val_dataset = ShapeNetDataset('/home/share/datasets/shapenet_manifold_v2.h5', '02691156', split='val', dtype=dtype)

train_dataloader = DataLoader(train_dataset, batch_size=5)
val_dataloader = DataLoader(val_dataset, batch_size=5)

hypo_net = PEFC(
    in_features=3,
    out_features=1,  # 1 - l1, l2; 2 - multitask
    num_hidden_layers=8,
    hidden_features=512,
)
hypo_net.net.apply(sal_init)
hypo_net.net[-1].apply(sal_init_last_layer)

model = MetaSDF(
    hypo_net,
    hypo_loss=l1_hypo_loss,  # 3d - multitask_batch_loss, l2_batch_loss - mnist
    init_lr=5e-3,  # 5e-3
    num_meta_steps=5,  # 5
    first_order=False,
    lr_type='per_parameter_per_step',
).cuda()
optimizer = torch.optim.Adam(
    lr=5e-4 * np.sqrt(train_dataloader.batch_size / 32),  # 5e-4
    params=model.parameters()
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=350, gamma=0.5)

start_epoch = 0
load_path = None  # '../output/meta_air_e0122.pth'
if load_path:
    checkpoint = torch.load(load_path)
    model = checkpoint['model']
    assert next(model.parameters()).is_cuda
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # noinspection PyRedeclaration
    start_epoch = checkpoint['epoch'] + 1

writer = SummaryWriter()
for epoch in tqdm(range(start_epoch, 3000), desc='Epoch'):
    model.train()
    for step, batch_cpu in enumerate(train_dataloader):
        global_step = step + epoch * len(train_dataloader)

        train_loss = next_step(
            model=model, hyper_loss=l1_hyper_loss, dataset=train_dataset,
            epoch=epoch, step=step, batch_cpu=batch_cpu,
            log_every=100 / train_dataloader.batch_size,
            get_context_params=lambda batch_gpu: model.generate_params(batch_gpu['context']),
            # get_context_params_test=lambda batch_gpu: model.generate_params(batch_gpu['surface']),  TODO
        )

        writer.add_scalar('Loss/train', train_loss, global_step)

        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for step, batch_cpu in enumerate(val_dataloader):
            global_step = step + epoch * len(val_dataloader)

            valid_loss = next_step(
                model=model, hyper_loss=l1_hyper_loss, dataset=val_dataset,
                epoch=epoch, step=step, batch_cpu=batch_cpu, log_every=1,
                get_context_params=lambda batch_gpu: model.generate_params(batch_gpu['context']),
                # get_context_params_test=lambda batch_gpu: model.generate_params(batch_gpu['surface']),  TODO
            )

            writer.add_scalar('Loss/valid', valid_loss, global_step)

    scheduler.step()

    if epoch % 1 == 0:
        torch.save({
            'model': model,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
        }, f'../output/meta_air_e{epoch:04d}.pth')
