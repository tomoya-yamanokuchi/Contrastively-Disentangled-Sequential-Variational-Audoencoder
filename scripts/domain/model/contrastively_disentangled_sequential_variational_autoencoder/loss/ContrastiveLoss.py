import torch
import torch.nn as nn
import math



class ContrastiveLoss(nn.Module):
    def __init__(self, tau=0.5, normalize=True):
        super(ContrastiveLoss, self).__init__()
        self.tau        = tau
        self.normalize  = normalize


    def cosine_similarity_matrix(self, xi, xj):
        '''
        x: [num_data, dim]
        '''
        x                 = torch.cat((xi, xj), dim=0)                              # [2*num_data, dim]
        x_norm            = torch.norm(x, dim=1)                                    # [2*num_data,]
        x_norm_mat        = torch.mm(x_norm.unsqueeze(1), x_norm.unsqueeze(1).T)    # [2*num_data, 2*num_data]
        inner_product_mat = torch.mm(x, x.T)                                        # [2*num_data, 2*num_data]
        similarity        = inner_product_mat / x_norm_mat.clamp(min=1e-16)         # [2*num_data, 2*num_data]
        return similarity


    def cosine_similarity_vector(self, xi, xj):
        '''
        x: [num_data, dim]
        '''
        norm_for_each_positive_pair          = torch.norm(xi, dim=1) * torch.norm(xj, dim=1) # [num_data, dim]
        inner_product_for_each_positive_pair = torch.sum(xi * xj, dim=-1)
        similarity                           = inner_product_for_each_positive_pair / norm_for_each_positive_pair.clamp(min=1e-16)
        return similarity


    def forward(self, xi, xj):
        '''
            xi: [num_batch, dim] == f_mean (or z_mean)
            xj: [num_batch, dim] == f_mean_aug (or z_mean_aug)
            -------
            calcurate 0.5*(C(z) + C(z^m)) jointly
        '''
        num_bacth                   = xi.shape[0]

        # "negative" sequences
        cosine_similarity_minibacth = self.cosine_similarity_matrix(xi, xj)
        phi_minibacth               = torch.exp(cosine_similarity_minibacth / self.tau)         # to be sumed
        phi_minibacth_diag          = torch.exp(torch.ones(2*num_bacth) / self.tau).type_as(xi) # cosine_sim(x, x) = 1
        phi_minibacth               = torch.sum(phi_minibacth, dim=-1) - phi_minibacth_diag     # includes both phi_positive and phi_negative
        # num_negative_sequence       = torch.Tensor([2*num_bacth - 2]).type_as(xi)               # - 2 = (diagnal element) + (positive pair)

        # "positive" sequences
        cosine_similarity_positive  =  self.cosine_similarity_vector(xi, xj)
        phi_positive                = torch.exp(cosine_similarity_positive / self.tau)
        phi_positive                = torch.cat((phi_positive, phi_positive), dim=0)

        # contrastive_loss            = torch.mean(torch.log(phi_positive / phi_minibacth)) + torch.log(num_negative_sequence + torch.Tensor([1]).type_as(xi))
        contrastive_loss            = torch.mean(torch.log(phi_positive / phi_minibacth))
        return contrastive_loss