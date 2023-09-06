import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from basicsr.utils.registry import MODEL_REGISTRY
from .srgan_model import SRGANModel


@MODEL_REGISTRY.register()
class ESRGANModel(SRGANModel):
    def cos_similarity(self, weight):
        weight = weight / weight.norm(dim=-1).unsqueeze(-1)
        cos_distance = torch.mm(weight, weight.transpose(1,0))
        cosine_matrix = cos_distance.pow(2)
        cosine_matrix.fill_diagonal_(0)
        return cosine_matrix.mean()
    def lda_loss(self, x):
        b, hw, c = x[0].shape
        # loss between
        loss_between = 0
        for i in range(len(x)):
            for j in range(i+1, len(x)):
                loss_between += torch.mean((x[i] * x[j]) / (torch.sum(x[i]) * torch.sum(x[j]) + 1e-5))
        # loss within
        loss_within = 0
        for k in range(len(x)):
            loss_within += torch.var(x[k])
        return loss_between + loss_within
    
    def load_balancing_loss(self, routing):
        values, index = routing.topk(k=1, dim=-1)
        values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
        mask = F.one_hot(index, routing.shape[-1]).float()
        density = mask.mean(dim=1)
        density_proxy = routing.mean(dim=1)
        balancing_loss = (density_proxy * density).mean()
        return balancing_loss * float(routing.shape[-1] ** 2) / routing.shape[1]

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss (relativistic gan)
            real_d_pred, routing, _, _ = self.net_d(self.gt)
            real_d_pred = real_d_pred.detach()
            fake_g_pred, _, _, _ = self.net_d(self.output, routing.detach())
            l_g_real = self.cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
            l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
            l_g_gan = (l_g_real + l_g_fake) / 2
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan
            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        torch.autograd.set_detect_anomaly(True)
        self.optimizer_d.zero_grad()
        
        # Real
        real_d_pred, routing, feature, weight = self.net_d(self.gt)
        fake_d_pred, _, _, _ = self.net_d(self.output, routing.detach())
        fake_d_pred = fake_d_pred.detach()
        # adversarial loss
        l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
        # orthogonal loss
        l_d_real += self.cos_similarity(weight) * 10.
        # LDA loss
        l_d_real += self.lda_loss(feature) * 10.
        # load_balancing_loss
        l_d_real += self.load_balancing_loss(routing) * 0.05
        l_d_real.backward()

        # fake
        fake_d_pred, _, _, _ = self.net_d(self.output.detach(), routing.detach())
        l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
        l_d_fake.backward()
        self.optimizer_d.step()

        loss_dict['l_d_real'] = l_d_real
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
