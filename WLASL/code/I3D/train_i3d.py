import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision import transforms
import numpy as np
from pathlib import Path

import videotransforms
from configs import Config
from pytorch_i3d import InceptionI3d
from datasets.nslt_dataset import NSLT as Dataset


def _auto_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train(
    videos_dir,
    split_json,
    weights_path,
    save_dir,
    lr=1e-3,
    weight_decay=1e-7,
    batch_size=4,
    max_epochs=50,
    dropout=0.5,
    n_tune_layers=0,
    experiment_name="exp",
    seed=0,
    device=None,
):
    """
    Fine-tune I3D on WLASL data.

    n_tune_layers controls how many backbone endpoints (from the end) are trained:
      0 or negative  → full fine-tuning (all layers)
      1-2            → only the classification head (logits)
      3              → Mixed_5c + head
      5              → Mixed_5b + MaxPool + Mixed_5c + head
      (see InceptionI3d.VALID_ENDPOINTS for the full list)

    Returns dict with keys train_loss, train_acc, val_loss, val_acc (list per epoch).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = _auto_device()

    use_pretrained = n_tune_layers > 0
    mode_str = f"partial freeze (n_tune_layers={n_tune_layers})" if use_pretrained else "full fine-tune"
    print(f"[{experiment_name}] device={device}  lr={lr}  {mode_str}")

    root = {'word': str(videos_dir)}
    train_tf = transforms.Compose([videotransforms.RandomCrop(224), videotransforms.RandomHorizontalFlip()])
    val_tf   = transforms.Compose([videotransforms.CenterCrop(224)])

    train_ds = Dataset(str(split_json), 'train', root, 'rgb', train_tf)
    val_ds   = Dataset(str(split_json), 'test',  root, 'rgb', val_tf)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    num_classes = train_ds.num_classes
    print(f"  classes={num_classes}  train={len(train_ds)}  val={len(val_ds)}")

    i3d = InceptionI3d(400, in_channels=3, dropout_keep_prob=dropout)
    i3d.load_state_dict(torch.load(str(weights_path), map_location='cpu'))
    i3d.replace_logits(num_classes)
    i3d = i3d.to(device)

    optimizer = optim.Adam(i3d.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.3)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        for phase, dl in [('train', train_dl), ('val', val_dl)]:
            i3d.train(phase == 'train')
            tot_loss = 0.0
            confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

            for inputs, labels, _ in dl:
                inputs = inputs.to(device)
                labels = labels.to(device)
                t = inputs.size(2)

                with torch.set_grad_enabled(phase == 'train'):
                    logits = i3d(inputs, pretrained=use_pretrained, n_tune_layers=max(n_tune_layers, 0))
                    logits = F.interpolate(logits, t, mode='linear', align_corners=False)
                    loc_loss = F.binary_cross_entropy_with_logits(logits, labels)
                    cls_loss = F.binary_cross_entropy_with_logits(
                        torch.max(logits, dim=2)[0], torch.max(labels, dim=2)[0])
                    loss = 0.5 * loc_loss + 0.5 * cls_loss
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                tot_loss += loss.item()
                preds = torch.argmax(torch.max(logits.detach(), dim=2)[0], dim=1).cpu().numpy()
                gts   = torch.argmax(torch.max(labels,          dim=2)[0], dim=1).cpu().numpy()
                for g, p in zip(gts, preds):
                    confusion[g, p] += 1

            acc      = float(np.trace(confusion)) / max(int(np.sum(confusion)), 1)
            avg_loss = tot_loss / max(len(dl), 1)
            history['train_loss' if phase == 'train' else 'val_loss'].append(avg_loss)
            history['train_acc'  if phase == 'train' else 'val_acc' ].append(acc)

            if phase == 'val':
                scheduler.step(avg_loss)
                if acc >= best_val_acc:
                    best_val_acc = acc
                    torch.save(i3d.state_dict(), str(Path(save_dir) / f"{experiment_name}_best.pt"))

        print(f"  Epoch {epoch:3d}/{max_epochs}  "
              f"loss={history['train_loss'][-1]:.4f}/{history['val_loss'][-1]:.4f}  "
              f"acc={history['train_acc'][-1]:.3f}/{history['val_acc'][-1]:.3f}")

    print(f"  Best val acc: {best_val_acc:.3f}")
    return history


def run(configs,
        mode='rgb',
        root='/ssd/Charades_v1_rgb',
        train_split='charades/charades.json',
        save_model='',
        weights=None):
    print(configs)

    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip()])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(train_split, 'train', root, mode, train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=configs.batch_size, shuffle=True, num_workers=0,
                                             pin_memory=True)
    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=True,
                                                 num_workers=2, pin_memory=False)

    dataloaders = {'train': dataloader, 'test': val_dataloader}

    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('weights/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('weights/rgb_imagenet.pt'))

    num_classes = dataset.num_classes
    i3d.replace_logits(num_classes)

    if weights:
        print('loading weights {}'.format(weights))
        i3d.load_state_dict(torch.load(weights))

    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = configs.init_lr
    weight_decay = configs.adam_weight_decay
    optimizer = optim.Adam(i3d.parameters(), lr=lr, weight_decay=weight_decay)

    num_steps_per_update = configs.update_per_step
    steps = 0
    epoch = 0
    best_val_score = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.3)
    while steps < configs.max_steps and epoch < 400:
        print('Step {}/{}'.format(steps, configs.max_steps))
        print('-' * 10)

        epoch += 1
        for phase in ['train', 'test']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()

            confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
            for data in dataloaders[phase]:
                num_iter += 1
                if data == -1:
                    continue

                inputs, labels, vid = data
                inputs = inputs.cuda()
                t = inputs.size(2)
                labels = labels.cuda()

                per_frame_logits = i3d(inputs, pretrained=False)
                per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear', align_corners=False)

                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss += loc_loss.data.item()

                predictions = torch.max(per_frame_logits, dim=2)[0]
                gt = torch.max(labels, dim=2)[0]

                cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0],
                                                              torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss.data.item()

                for i in range(per_frame_logits.shape[0]):
                    confusion_matrix[torch.argmax(gt[i]).item(), torch.argmax(predictions[i]).item()] += 1

                loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
                tot_loss += loss.data.item()
                if num_iter == num_steps_per_update // 2:
                    print(epoch, steps, loss.data.item())
                loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    if steps % 10 == 0:
                        acc = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                        print('Epoch {} {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}'.format(
                            epoch, phase,
                            tot_loc_loss / (10 * num_steps_per_update),
                            tot_cls_loss / (10 * num_steps_per_update),
                            tot_loss / 10, acc))
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.

            if phase == 'test':
                val_score = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                if val_score > best_val_score or epoch % 2 == 0:
                    best_val_score = val_score
                    model_name = save_model + "nslt_" + str(num_classes) + "_" + str(steps).zfill(
                        6) + '_%3f.pt' % val_score
                    torch.save(i3d.module.state_dict(), model_name)
                    print(model_name)

                print('VALIDATION: {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}'.format(
                    phase,
                    tot_loc_loss / num_iter,
                    tot_cls_loss / num_iter,
                    (tot_loss * num_steps_per_update) / num_iter,
                    val_score))

                scheduler.step(tot_loss * num_steps_per_update / num_iter)


if __name__ == '__main__':
    import argparse

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, help='rgb or flow')
    parser.add_argument('-save_model', type=str)
    parser.add_argument('-root', type=str)
    parser.add_argument('--num_class', type=int)
    args = parser.parse_args()

    mode = 'rgb'
    root = {'word': '../../data/WLASL2000'}
    save_model = 'checkpoints/'
    train_split = 'preprocess/nslt_2000.json'
    weights = None
    config_file = 'configfiles/asl2000.ini'

    configs = Config(config_file)
    print(root, train_split)
    run(configs=configs, mode=mode, root=root, save_model=save_model, train_split=train_split, weights=weights)
