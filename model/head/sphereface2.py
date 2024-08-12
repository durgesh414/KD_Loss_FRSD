import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class SphereFace2(nn.Module):
    """ reference: <SphereFace2: Binary Classification is All You Need
                    for Deep Face Recognition>
        margin='C' -> SphereFace2-C
        margin='A' -> SphereFace2-A
        marign='M' -> SphereFAce2-M
    """

    def __init__(self, feat_dim, num_class, magn_type='C',
            alpha=0.7, r=40., m=0.4, t=3., lw=50.):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.magn_type = magn_type

        # alpha is the lambda in paper Eqn. 5
        self.alpha = alpha     # Lambdaaa
        self.r = r
        self.m = m
        self.t = t
        self.lw = lw

        # init weights
        self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        nn.init.xavier_normal_(self.w)

        # init bias
        z = alpha / ((1. - alpha) * (num_class - 1.))

        if magn_type == 'C':
            ay = r * (2. * 0.5** t- 1. - m)    
            ai = r * (2. * 0.5**t - 1. + m)    
        else:
            raise NotImplementedError

        temp = (1. - z)**2 + 4. * z * math.exp(ay - ai)

        b = (math.log(2. * z) - ai
             - math.log(1. - z +  math.sqrt(temp)))
        self.b = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.b, b)

    def forward(self, x, y):

        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        #delta theta with margin
        cos_theta = F.normalize(x, dim=1).mm(self.w)
        
        one_hot = torch.zeros_like(cos_theta)  
        one_hot.scatter_(1, y.view(-1, 1), 1.)  #onehot encoding

        with torch.no_grad():
            if self.magn_type == 'C':
                g_cos_theta = 2. * ((cos_theta + 1.) / 2.).pow(self.t) - 1. # g(cos_theta) 
                g_cos_theta = g_cos_theta - self.m * (2. * one_hot - 1.)    # g(cos_theta) +- m  by changing 0, 1 to -1, 1 which for +ve samples -m and -ve samples +m                                                            
    
            else:
                raise NotImplementedError
            d_theta = g_cos_theta - cos_theta
        

        logits = self.r * (cos_theta + d_theta) + self.b
        weight = self.alpha * one_hot + (1. - self.alpha) * (1. - one_hot)  # aplying lambda to balance the classes   0.3000  and 0.7000
        weight = self.lw * self.num_class / self.r * weight                 # 647.3250  and 1510.4249  (2157.75)


        loss = F.binary_cross_entropy_with_logits(logits, one_hot, weight=weight)
        return loss


        # def mhe_loss(self, w, y):
        #     # Select weight vectors for the unique classes in the batch
        #     sel_w = w#[:, torch.unique(y)]
            
        #     # Compute the Gram matrix of angles
        #     gram_mat = torch.acos(torch.matmul(sel_w.t(), sel_w).clamp(-1.+1e-5, 1.-1e-5))
            
        #     # Calculate MHE loss
        #     shape_gram = gram_mat.size()
        #     MHE_loss = torch.sum(torch.triu(torch.pow(gram_mat, -2), diagonal=1))
        #     MHE_loss = MHE_loss / (shape_gram[0] * (shape_gram[0] - 1) * 0.5)
            
        #     return MHE_loss


        # binary_loss = F.binary_cross_entropy_with_logits(logits, one_hot, weight=weight)
        # #mhe_loss_value = self.mhe_loss_cosine(self.w)        

        # mhe_loss_value = self.mhe_loss(self.w, y)  # Compute MHE loss
        # total_loss = binary_loss + self.lambda_mhe * mhe_loss_value

        # return total_loss, self.lambda_mhe * mhe_loss_value  


        # total_loss = binary_loss + self.lambda_mhe * mhe_loss_value      #Combine the binary loss and MHE loss
        # return total_loss, self.lambda_mhe * mhe_loss_value  # Sending the MHE Loss as well
    
        #print("Loss scales" ,binary_loss.item(), mhe_loss_value.item(), self.lambda_mhe*mhe_loss_value.item())
        # print("Tots", total_loss)