import numpy as np
import h5py as h5
from time import time
from sys import stdout

import torch


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(state, model_name):
    torch.save(state, model_name)
    print('Model saved to ' + model_name)


def train(iterator, model, loss_func, optimizer, scheduler, epoch, iter, **kwargs):
    num_workers = kwargs.get('num_workers')
    train_mode = kwargs.get('train_mode')
    model_name = kwargs['path2data'] + 'models/' + kwargs.get('model_name')

    batch_time = AverageMeter()
    data_time = AverageMeter()

    NLL = AverageMeter()
    if train_mode == 'tln':
        CL2 = AverageMeter()
    elif train_mode == 'cvae':
        KLDI = AverageMeter()
    elif train_mode == 'dvae':
        KLDV = AverageMeter()
        KLDI = AverageMeter()
    IOU = AverageMeter()

    model.train()
    torch.set_grad_enabled(True)

    end = time()
    for i, batch in enumerate(iterator):
        if iter + i >= len(iterator):
            break
        data_time.update(time() - end)
        scheduler(optimizer, epoch, iter + i)

        images = batch['image'].cuda(async=True)
        targets = batch['grid'].cuda(async=True)
        outputs = model(images, targets)

        if train_mode == 'gnet':
            loss = nll = loss_func(outputs, targets)
        elif train_mode == 'tln':
            loss, nll, cl2 = loss_func(outputs, targets)
        elif train_mode == 'cvae':
            loss, nll, kldi = loss_func(outputs, targets)
        elif train_mode == 'dvae':
            loss, nll, kldv, kldi = loss_func(outputs, targets)

        with torch.no_grad():
            iou = 100. * (
                torch.sum(targets[:, 1] * torch.exp(outputs['logprobs'][:, 1]), dim=3).sum(2).sum(1) /
                torch.sum(torch.max(targets[:, 1], torch.exp(outputs['logprobs'][:, 1])), dim=3).sum(2).sum(1)
            ).mean()

        NLL.update(nll.item(), targets.shape[0])
        if train_mode == 'tln':
            CL2.update(cl2.item(), targets.shape[0])
        elif train_mode == 'cvae':
            KLDI.update(kldi.item(), targets.shape[0])
        elif train_mode == 'dvae':
            KLDV.update(kldv.item(), targets.shape[0])
            KLDI.update(kldi.item(), targets.shape[0])
        IOU.update(iou.item(), targets.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time() - end)
        if (iter + i + 1) % (num_workers // 2) == 0:
            line = 'Epoch: [{0}][{1}/{2}]\t'.format(epoch + 1, iter + i + 1, len(iterator))
            # line += 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(batch_time=batch_time)
            # line += 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(data_time=data_time)
            # line += 'LR {:.6f}\t'.format(optimizer.param_groups[0]['lr'])
            line += 'NLL {NLL.val:.2f} ({NLL.avg:.2f})\t'.format(NLL=NLL)
            if train_mode == 'tln':
                line += 'CL2 {CL2.val:.3f} ({CL2.avg:.3f})\t'.format(CL2=CL2)
            elif train_mode == 'cvae':
                line += 'KLDI {KLDI.val:.3f} ({KLDI.avg:.3f})\t'.format(KLDI=KLDI)
            elif train_mode == 'dvae':
                line += 'KLDV {KLDV.val:.3f} ({KLDV.avg:.3f})\t'.format(KLDV=KLDV)
                line += 'KLDI {KLDI.val:.3f} ({KLDI.avg:.3f})\t'.format(KLDI=KLDI)
            line += 'IOU {IOU.val:.2f} ({IOU.avg:.2f})\n'.format(IOU=IOU)
            stdout.write(line)
            stdout.flush()

        if (iter + i + 1) % (100 * num_workers) == 0:
            save_model({
                'epoch': epoch,
                'iter': iter + i + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }, model_name)

        end = time()

    save_model({
        'epoch': epoch + 1,
        'iter': 0,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }, model_name)

    iterator.dataset.close()


def evaluate(iterator, model, loss_func, **kwargs):
    train_mode = kwargs.get('train_mode')
    saving_mode = kwargs.get('saving_mode')
    threshold = kwargs.get('threshold')
    training_metrics = kwargs.get('training_metrics')
    acc_m = kwargs.get('accuracy')
    iou_m = kwargs.get('iou')

    if saving_mode:
        grids_fname = '{}_{}_grids.h5'.format(kwargs['model_name'][:-4], iterator.dataset.part)
        grids_fname = kwargs['path2data'] + grids_fname

        grids_file = h5.File(grids_fname, 'w')
        reco_grids = grids_file.create_dataset(
            'reco_grids',
            shape=(len(iterator.dataset), kwargs['grid_size']**3 // 8),
            dtype=np.uint8
        )
        reco_ious = grids_file.create_dataset(
            'reco_ious',
            shape=(len(iterator.dataset),),
            dtype=np.float32
        )

    batch_time = AverageMeter()
    data_time = AverageMeter()
    if training_metrics:
        NLL = AverageMeter()
        NLL_cat = [AverageMeter() for m in range(len(kwargs['cat_names']))]

        if train_mode == 'tln':
            CL2 = AverageMeter()
            CL2_cat = [AverageMeter() for m in range(len(kwargs['cat_names']))]
        elif train_mode == 'cvae':
            KLDI = AverageMeter()
            KLDI_cat = [AverageMeter() for m in range(len(kwargs['cat_names']))]
        elif train_mode == 'dvae':
            KLDV = AverageMeter()
            KLDV_cat = [AverageMeter() for m in range(len(kwargs['cat_names']))]
            KLDI = AverageMeter()
            KLDI_cat = [AverageMeter() for m in range(len(kwargs['cat_names']))]

        if acc_m:
            ACC = AverageMeter()

        if iou_m:
            IOU = AverageMeter()
            IOU_bins = np.arange(1, 101, 1)
            IOU_hist = np.zeros(100, dtype=np.float32)
            IOU_cat = [AverageMeter() for m in range(len(kwargs['cat_names']))]

    model.eval()
    torch.set_grad_enabled(False)

    end = time()
    for i, batch in enumerate(iterator):
        data_time.update(time() - end)

        images = batch['image'].cuda(async=True)
        targets = batch['grid'].cuda(async=True)
        outputs = model(images, targets)

        if saving_mode:
            reco_grids[kwargs['batch_size'] * i:kwargs['batch_size'] * (i + 1)] = np.packbits(
                (torch.exp(outputs['logprobs'][:, 1]) >=
                 threshold).cpu().numpy().astype(np.uint8).reshape(targets.shape[0], -1),
                axis=1
            )

        if training_metrics:
            sample = {}
            sample['true_grid'] = torch.max(targets, dim=1)[1].squeeze(1).cpu().numpy().astype(np.uint8)
            sample['reco_grid'] = (torch.exp(outputs['logprobs'][:, 1]) >= threshold).cpu().numpy().astype(np.uint8)

            if train_mode == 'gnet':
                nll = loss_func(outputs, targets)
            elif train_mode == 'tln':
                loss, nll, cl2 = loss_func(outputs, targets)
            elif train_mode == 'cvae':
                loss, nll, kldi = loss_func(outputs, targets)
            elif train_mode == 'dvae':
                loss, nll, kldv, kldi = loss_func(outputs, targets)

            cat = np.argmax(np.logical_and(np.array(kwargs['cat_idx']) <= i,
                                           i < np.array(kwargs['cat_idx'][1:] + [len(iterator)])))

            NLL.update(nll.item(), targets.shape[0])
            NLL_cat[cat].update(nll.item(), targets.shape[0])

            if train_mode == 'tln':
                CL2.update(cl2.item(), targets.shape[0])
                CL2_cat[cat].update(cl2.item(), targets.shape[0])

            elif train_mode == 'cvae':
                KLDI.update(kldi.item(), targets.shape[0])
                KLDI_cat[cat].update(kldi.item(), targets.shape[0])

            elif train_mode == 'dvae':
                KLDV.update(kldv.item(), targets.shape[0])
                KLDV_cat[cat].update(kldv.item(), targets.shape[0])
                KLDI.update(kldi.item(), targets.shape[0])
                KLDI_cat[cat].update(kldi.item(), targets.shape[0])

            if acc_m:
                sample['accuracy'] = 100. * (sample['true_grid'] == sample['reco_grid']).sum(axis=(1, 2, 3)) / \
                    sample['true_grid'].shape[1]**3
                ACC.update(np.mean(sample['accuracy']), sample['accuracy'].shape[0])

            if iou_m:
                sample['iou'] = 100. * np.logical_and(sample['true_grid'], sample['reco_grid']).sum(axis=(1, 2, 3)) / \
                    np.logical_or(sample['true_grid'], sample['reco_grid']).sum(axis=(1, 2, 3))

                if saving_mode:
                    reco_ious[kwargs['batch_size'] * i:kwargs['batch_size'] * (i + 1)] = \
                        sample['iou']

                sample_IOU = np.mean(sample['iou'])
                IOU_hist[(sample_IOU < IOU_bins).argmax()] += 1
                IOU.update(sample_IOU, sample['iou'].shape[0])
                cat = np.argmax(np.logical_and(np.array(kwargs['cat_idx']) <= i,
                                               i < np.array(kwargs['cat_idx'][1:] + [len(iterator)])))
                IOU_cat[cat].update(sample_IOU, sample['iou'].shape[0])

        batch_time.update(time() - end)
        line = 'Saving results: [{0}/{1}]\t'.format(i + 1, len(iterator))
        # line += 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(batch_time=batch_time)
        # line += 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(data_time=data_time)
        if training_metrics:
            line += 'NLL {NLL.val:.2f} ({NLL.avg:.2f})\t'.format(NLL=NLL)

            if train_mode == 'tln':
                line += 'CL2 {CL2.val:.2f} ({CL2.avg:.2f})'.format(CL2=CL2)

            elif train_mode == 'cvae':
                line += 'KLDI {KLDI.val:.2f} ({KLDI.avg:.2f})'.format(KLDI=KLDI)

            elif train_mode == 'dvae':
                line += 'KLDV {KLDV.val:.2f} ({KLDV.avg:.2f})\t'.format(KLDV=KLDV)
                line += 'KLDI {KLDI.val:.2f} ({KLDI.avg:.2f})'.format(KLDI=KLDI)

            if acc_m:
                line += '\tACC {ACC.val:.2f} ({ACC.avg:.2f})'.format(ACC=ACC)

            if iou_m:
                line += '\tIOU {IOU.val:.2f} ({IOU.avg:.2f})'.format(IOU=IOU)
        line += '\n'
        stdout.write(line)
        stdout.flush()
        end = time()

    if saving_mode:
        grids_file.close()

    for i, cat in enumerate(kwargs['cat_names']):
        if train_mode == 'gnet':
            print('{}:\tNLL {:.2f}'.format(cat, NLL_cat[i].avg))

        elif train_mode == 'tln':
            print('{}:\tNLL {:.2f}\tCL2 {:.2f}'.format(cat, NLL_cat[i].avg, CL2_cat[i].avg))

        elif train_mode == 'cvae':
            print('{}:\tNLL {:.2f}\tKLDI {:.2f}'.format(cat, NLL_cat[i].avg, KLDI_cat[i].avg))

        elif train_mode == 'dvae':
            print('{}:\tNLL {:.2f}\tKLDV {:.2f}\tKLDI {:.2f}'.format(cat,
                                                                     NLL_cat[i].avg,
                                                                     KLDV_cat[i].avg,
                                                                     KLDI_cat[i].avg))

    if iou_m:
        for i, cat in enumerate(kwargs['cat_names']):
            print('{}: {:.2f} %'.format(cat, IOU_cat[i].avg))
        print('MEAN: {:.2f} %'.format(sum(map(lambda x: x.avg, IOU_cat)) / len(IOU_cat)))
        IOU_hist /= IOU_hist.sum()
        print('[' + ''.join('{}, '.format(bv) for bv in IOU_hist) + ']')
