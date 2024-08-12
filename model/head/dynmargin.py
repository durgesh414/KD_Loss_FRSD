import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class Dynmargin(nn.Module):
    def __init__(self, feat_dim, num_class, magn_type='C',
                 alpha=0.7, r=40., m=0.4, t=3., lw=10., lambda_mhe=0.10):
    
        super().__init__()

        self.feat_dim = feat_dim
        self.num_class = num_class
        self.magn_type = magn_type

        self.alpha = alpha
        self.r = r
        self.m0 = m  # Initial margin
        self.m = m  # Current margin
        self.t = t
        self.lw = lw
        self.lambda_mhe = lambda_mhe

        self.max_loss = 10.0  # Maximum expected loss
        self.alpha_loss = 0.99  # Smoothing factor for moving average

        self.moving_avg_loss = None
        self.moving_avg_r = r  # Initialize moving average for r
        self.moving_avg_m = m  # Initialize moving average for m

        self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        nn.init.xavier_normal_(self.w)

        z = alpha / ((1. - alpha) * (num_class - 1.))

        if magn_type == 'C':
            ay = r * (2. * 0.5**t - 1. - self.m)
            ai = r * (2. * 0.5**t - 1. + self.m)
        else:
            raise NotImplementedError

        temp = (1. - z)**2 + 4. * z * math.exp(ay - ai)

        b = (math.log(2. * z) - ai
             - math.log(1. - z + math.sqrt(temp)))
        self.b = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.b, b)



    def forward(self, x, y):
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        # delta theta with margin
        cos_theta = F.normalize(x, dim=1).mm(self.w)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, y.view(-1, 1), 1.)  # onehot encoding

        with torch.no_grad():
            if self.magn_type == 'C':
                g_cos_theta = 2. * ((cos_theta + 1.) / 2.).pow(self.t) - 1.  # g(cos_theta) 
                g_cos_theta = g_cos_theta - self.m * (2. * one_hot - 1.)    # g(cos_theta) +- m by changing 0, 1 to -1, 1 which for +ve samples -m and -ve samples +m                                                            
            else:
                raise NotImplementedError
            d_theta = g_cos_theta - cos_theta
        
        logits = self.r * (cos_theta + d_theta) + self.b
        weight = self.alpha * one_hot + (1. - self.alpha) * (1. - one_hot)  # applying lambda to balance the classes
        weight = self.lw * self.num_class / self.r * weight                 # 

        loss = F.binary_cross_entropy_with_logits(logits, one_hot, weight=weight)
        print("self.r, self.m", self.r, self.m)

        # Update the moving average of the loss
        with torch.no_grad():
            if self.moving_avg_loss is None:
                self.moving_avg_loss = loss.item()
            else:
                self.moving_avg_loss = self.alpha_loss * loss.item() + (1 - self.alpha_loss) * self.moving_avg_loss

            # Update r based on the moving average
            feature_norms = torch.norm(x, dim=1)
            mean_norm = torch.mean(feature_norms).item()

            if self.moving_avg_r is None:
                self.moving_avg_r = mean_norm
            else:
                self.moving_avg_r = self.alpha_loss * mean_norm + (1 - self.alpha_loss) * self.moving_avg_r

            # Update m based on the moving average of the loss
            target_m = self.m0 * (1 - self.moving_avg_loss / self.max_loss)

            if self.moving_avg_m is None:
                self.moving_avg_m = target_m
            else:
                self.moving_avg_m = self.alpha_loss * target_m + (1 - self.alpha_loss) * self.moving_avg_m

            # Update self.r and self.m
            self.r = self.moving_avg_r
            self.m = self.moving_avg_m

        return loss




# print("self.r, self.m", self.r, self.m)



# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import math

# class Dynmargin(nn.Module):
#     """ reference: <SphereFace2: Binary Classification is All You Need
#                     for Deep Face Recognition>
#         margin='C' -> SphereFace2-C
#     """

#     def __init__(self, feat_dim, num_class, magn_type='C',
#             alpha=0.7, r=40., m=0.4, t=3., lw=50., lambda_mhe=0.10):
#         super().__init__()
#         self.feat_dim = feat_dim
#         self.num_class = num_class
#         self.magn_type = magn_type

#         print(alpha, r, m, t, lw)

#         # alpha is the lambda in paper Eqn. 5
#         self.alpha = alpha     
#         self.r = r
#         self.m = m
#         self.t = t
#         self.lw = lw
#         self.lambda_mhe = lambda_mhe

#         # init weights
#         self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
#         nn.init.xavier_normal_(self.w)

#         # init bias
#         z = alpha / ((1. - alpha) * (num_class - 1.))

#         if magn_type == 'C':
#             ay = r * (2. * 0.5**t - 1. - m) 
#             ai = r * (2. * 0.5**t - 1. + m) 
#         else:
#             raise NotImplementedError

#         temp = (1. - z)**2 + 4. * z * math.exp(ay - ai)

#         b = (math.log(2. * z) - ai
#              - math.log(1. - z +  math.sqrt(temp)))
#         self.b = nn.Parameter(torch.Tensor(1))
#         nn.init.constant_(self.b, b)


#     def forward(self, x, y, iter, max_iter):

#         with torch.no_grad():
#             self.w.data = F.normalize(self.w.data, dim=0)

#         #delta theta with margin
#         cos_theta = F.normalize(x, dim=1).mm(self.w)
        
#         one_hot = torch.zeros_like(cos_theta)  
#         one_hot.scatter_(1, y.view(-1, 1), 1.)  #one hot encoding

#         # Dynamic margin
#         m_dynamic = self.m + (1 - cos_theta) * 0.1 # Example dynamic margin
#         # Adaptive scaling
#         r_adjusted = self.r * (1 + 0.5 * (iter / max_iter))  # Example adaptive scaling

#         with torch.no_grad():
#             if self.magn_type == 'C':
#                 g_cos_theta = 2. * ((cos_theta + 1.) / 2.).pow(self.t) - 1.  
#                 g_cos_theta = g_cos_theta - m_dynamic * (2. * one_hot - 1.)                                                             
#             else:
#                 raise NotImplementedError
#             d_theta = g_cos_theta - cos_theta

#         logits = r_adjusted * (cos_theta + d_theta) + self.b
#         weight = self.alpha * one_hot + (1. - self.alpha) * (1. - one_hot)  
#         weight = self.lw * self.num_class / r_adjusted * weight    #**0.666

#         loss = F.binary_cross_entropy_with_logits(logits, one_hot, weight=weight)
#         return loss
   