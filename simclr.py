import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

import torch_higher as higher


torch.manual_seed(0)




def obtain_model_grad(model):
    grad_ls = []
    for param in model.parameters():
        if param.grad is not None:
            grad_ls.append(param.grad.detach().clone())
    return grad_ls


def update_model_by_small_grad(model, model_grad_ls, eps):
    param_ls = list(model.parameters())
    for idx in range(len(param_ls)):
        param = param_ls[idx]
        param.data += eps*model_grad_ls[idx]


def compute_loss(sim_clr, model, images, criterion, labels):
    model_feat = model(images)
    logits, _ = sim_clr.info_nce_loss(model_feat)
    loss = criterion(logits, labels)
    return loss

def get_model_params(model):
    param_ls = []
    for param in list(model.parameters()):
        param_ls.append(param.data.clone())
    return param_ls

def set_model_params(model, param_ls):
    idx = 0
    for param in model.parameters():
        param.data.copy_(param_ls[idx])
        idx += 1


def compute_meta_grad_efficient(sim_clr, train_weights, model, train_images, train_labels, meta_images, meta_labels, criterion, optimizer, lr, eps = 1e-8):
    # meta_out = model(meta_images)
    # optimizer.zero_grad()

    # origin_model = get_model_params(model)

    # criterion.reduction = 'none'

    

    # agg_train_loss = torch.mean(train_loss1.view(-1)*train_weights.view(-1))

    # agg_train_loss.backward()

    # optimizer.step()

    optimizer.zero_grad()

    criterion.reduction = 'mean'

    meta_loss = compute_loss(sim_clr, model, meta_images, criterion, meta_labels)
    # meta_loss = criterion(meta_out, meta_labels)
    meta_loss.backward()
    meta_model_grad = obtain_model_grad(model)
    
    
    # set_model_params(model, origin_model)
    

    criterion.reduction = 'none'
    # train_out = model(train_images)
    # train_loss1 = criterion(train_out, train_labels)

    train_loss1 = compute_loss(sim_clr, model, train_images, criterion, train_labels)

    update_model_by_small_grad(model, meta_model_grad, eps)

    train_loss2 = compute_loss(sim_clr, model, train_images, criterion, train_labels)
    # train_out = model(train_images)
    # train_loss2 = criterion(train_out, train_labels)

    train_weight_grad = (train_loss2 - train_loss1)/(eps*len(train_images))*(-lr)

    del train_loss2, train_loss1, meta_model_grad

    return train_weight_grad.detach(), meta_loss

class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter(log_dir=self.args.output_dir)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        
        labels = torch.cat([torch.arange(self.args.batch_size, device = self.args.device) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).type(torch.float)
        self.mask = torch.eye(labels.shape[0], dtype=torch.bool, device = self.args.device)#.to(self.args.device)
        self.labels = labels[~self.mask].view(labels.shape[0], -1)

    def info_nce_loss(self, features):

        
        # labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        
        similarity_matrix = similarity_matrix[~self.mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[self.labels.bool()].view(self.labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~self.labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        #.to(self.args.device)

        logits = logits / self.args.temperature
        return logits, self.labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        labels = torch.zeros(self.args.batch_size*self.args.n_views, dtype=torch.long,device = self.args.device)

        for epoch_counter in range(self.args.start_epoch, self.args.epochs + self.args.start_epoch):
            for _,images, _ in tqdm(train_loader):
                # images = torch.cat(images, dim=0)

                # images = images.to(self.args.device)
                image_ls = []
                for idx in range(self.args.n_views):
                    image_ls.append(images[idx].to(self.args.device))
                images = torch.cat(image_ls, dim=0)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, _ = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            checkpoint_name = 'checkpoint_epoch_{:04d}.pth.tar'.format(epoch_counter)
            save_checkpoint({
                'epoch': epoch_counter,
                'arch': self.args.arch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
            logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")



            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
    
    def meta_train(self, train_loader, metaloader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        train_sample_count = len(train_loader.dataset)
        w_array = torch.rand([self.args.n_views, train_sample_count], device = self.args.device)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        metaloader = iter(metaloader)

        labels = torch.zeros(self.args.batch_size*self.args.n_views, dtype=torch.long,device = self.args.device)

        for epoch_counter in range(self.args.start_epoch, self.args.epochs + self.args.start_epoch):

            total_train_loss = 0

            total_meta_loss = 0

            total_top1 = 0

            for train_ids, images, _ in tqdm(train_loader):

                image_ls = []
                for idx in range(self.args.n_views):
                    image_ls.append(images[idx].to(self.args.device))
                images = torch.cat(image_ls, dim=0)

                # images = images.to(self.args.device)

                w_array.requires_grad = True

                with higher.innerloop_ctx(self.model, self.optimizer) as (meta_model, meta_opt):
                    eps = w_array[:,train_ids]
                    eps = eps.to(self.args.device)

                    with autocast(enabled=self.args.fp16_precision):
                        features = meta_model(images)
                        logits, _ = self.info_nce_loss(features)
                        self.criterion.reduction = 'none'


                        loss = torch.mean(self.criterion(logits, labels)*eps.view(-1))

                    meta_opt.step(loss)

                    meta_inputs =  next(metaloader)

                    meta_images = meta_inputs[1]

                    meta_image_ls = []
                    for idx in range(self.args.n_views):
                        meta_image_ls.append(meta_images[idx].to(self.args.device))
                    meta_images = torch.cat(meta_image_ls, dim=0)

                    # meta_images = torch.cat(meta_images, dim=0)

                    # meta_images = meta_images.to(self.args.device)

                    with autocast(enabled=self.args.fp16_precision):

                        
                        meta_features = meta_model(meta_images)
                        # meta_logits, meta_labels = meta_model(meta_images, compute_nce_loss = True, batch_size = self.args.batch_size, n_views = self.args.n_views, temperature=self.args.temperature, device = self.args.device)

                        self.criterion.reduction = 'mean'

                        meta_logits, _ = self.info_nce_loss(meta_features)

                        meta_loss = self.criterion(meta_logits, labels)

                    # eps_grads2 = compute_meta_grad_efficient(meta_model, self, eps, self.model, images, labels, meta_images, labels, self.criterion, self.optimizer, self.args.lr)

                    # eps_grads2 = eps_grads2.reshape(self.args.n_views,-1)
                    eps_grads = torch.autograd.grad(scaler.scale(meta_loss), eps)[0].detach()

                    w_array.requires_grad = False
            
                    # prev_w_array = w_array[train_ids].detach().clone()

                    w_array[:,train_ids] =  w_array[:,train_ids]-self.args.meta_lr*eps_grads

                    # print("max weight grad::", eps_grads.max().detach().cpu().item())
                    # print("max weight::", w_array.max().detach().cpu().item())
                    # print("min weight::", w_array.min().detach().cpu().item())
                    
                    w_array[:,train_ids] = torch.clamp(w_array[:,train_ids], max=1, min=1e-7)

                del eps, eps_grads, meta_images, meta_features, meta_logits,  meta_model

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, _ = self.info_nce_loss(features)
                    self.criterion.reduction = 'none'


                    loss = torch.mean(self.criterion(logits, labels)*w_array[:,train_ids].view(-1))

                total_train_loss += loss.cpu().detach().item()
                total_meta_loss += meta_loss.cpu().detach().item()
                

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                top1, top5 = accuracy(logits, labels, topk=(1, 5))

                if n_iter % self.args.log_every_n_steps == 0:
                    
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)
                total_top1 += top1[0]

                n_iter += 1

            total_train_loss = total_train_loss/len(train_loader)
            total_meta_loss = total_meta_loss/len(train_loader)
            total_top1 = total_top1/len(train_loader)

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {total_train_loss}\tValid loss:{total_meta_loss}\tTop1 accuracy: {total_top1}")
            checkpoint_name = 'checkpoint_epoch_{:04d}.pth.tar'.format(epoch_counter)
            save_checkpoint({
                'epoch': epoch_counter,
                'arch': self.args.arch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
            logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")



        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

    def meta_train2(self, train_loader, metaloader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        train_sample_count = len(train_loader.dataset)
        w_array = torch.rand([self.args.n_views, train_sample_count], device = self.args.device)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        metaloader = iter(metaloader)

        labels = torch.zeros(self.args.batch_size*self.args.n_views, dtype=torch.long,device = self.args.device)

        for epoch_counter in range(self.args.epochs):

            total_train_loss = 0

            total_meta_loss = 0

            total_top1 = 0

            for train_ids, images, _ in tqdm(train_loader):

                image_ls = []
                for idx in range(self.args.n_views):
                    image_ls.append(images[idx].to(self.args.device))
                images = torch.cat(image_ls, dim=0)

                # images = images.to(self.args.device)

                w_array.requires_grad = True

                # with higher.innerloop_ctx(self.model, self.optimizer) as (meta_model, meta_opt):
                eps = w_array[:,train_ids]
                eps = eps.to(self.args.device)
                meta_inputs =  next(metaloader)

                meta_images = meta_inputs[1]

                meta_image_ls = []
                for idx in range(self.args.n_views):
                    meta_image_ls.append(meta_images[idx].to(self.args.device))
                meta_images = torch.cat(meta_image_ls, dim=0)

                eps_grads, meta_loss = compute_meta_grad_efficient(self, eps, self.model, images, labels, meta_images, labels, self.criterion, self.optimizer, self.args.lr)

                eps_grads = eps_grads.reshape(self.args.n_views,-1)
                # eps_grads = torch.autograd.grad(scaler.scale(meta_loss), eps)[0].detach()

                w_array.requires_grad = False
        
                # prev_w_array = w_array[train_ids].detach().clone()

                w_array[:,train_ids] =  w_array[:,train_ids]-self.args.meta_lr*eps_grads

                # print("max weight grad::", eps_grads.max().detach().cpu().item())
                # print("max weight::", w_array.max().detach().cpu().item())
                # print("min weight::", w_array.min().detach().cpu().item())
                
                w_array[:,train_ids] = torch.clamp(w_array[:,train_ids], max=1, min=1e-7)

                #     with autocast(enabled=self.args.fp16_precision):
                #         features = meta_model(images)
                #         logits, _ = self.info_nce_loss(features)
                #         self.criterion.reduction = 'none'


                #         loss = torch.mean(self.criterion(logits, labels)*eps.view(-1))

                #     meta_opt.step(loss)

                    
                #     meta_image_ls = []
                #     for idx in range(self.args.n_views):
                #         meta_image_ls.append(meta_images[idx].to(self.args.device))
                #     meta_images = torch.cat(meta_image_ls, dim=0)

                #     # meta_images = torch.cat(meta_images, dim=0)

                #     # meta_images = meta_images.to(self.args.device)

                #     with autocast(enabled=self.args.fp16_precision):

                        
                #         meta_features = meta_model(meta_images)
                #         # meta_logits, meta_labels = meta_model(meta_images, compute_nce_loss = True, batch_size = self.args.batch_size, n_views = self.args.n_views, temperature=self.args.temperature, device = self.args.device)

                #         self.criterion.reduction = 'mean'

                #         meta_logits, _ = self.info_nce_loss(meta_features)

                #         meta_loss = self.criterion(meta_logits, labels)

                    
                # del eps, eps_grads, meta_images, meta_features, meta_logits,  meta_model

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, _ = self.info_nce_loss(features)
                    self.criterion.reduction = 'none'


                    loss = torch.mean(self.criterion(logits, labels)*w_array[:,train_ids].view(-1))

                total_train_loss += loss.cpu().detach().item()
                total_meta_loss += meta_loss.cpu().detach().item()
                
                del meta_loss
                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                top1, top5 = accuracy(logits, labels, topk=(1, 5))

                if n_iter % self.args.log_every_n_steps == 0:
                    
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)
                total_top1 += top1[0]

                n_iter += 1

            total_train_loss = total_train_loss/len(train_loader)
            total_meta_loss = total_meta_loss/len(train_loader)
            total_top1 = total_top1/len(train_loader)

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {total_train_loss}\tValid loss:{total_meta_loss}\tTop1 accuracy: {total_top1}")
            checkpoint_name = 'checkpoint_epoch_{:04d}.pth.tar'.format(epoch_counter)
            save_checkpoint({
                'epoch': epoch_counter,
                'arch': self.args.arch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
            logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")



        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

