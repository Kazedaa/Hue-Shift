import torch

class LinearNoiseSchedule:
    def __init__(self, T):
        super().__init__()
        beta_start = 1E-4
        beta_end = 0.02

        self.beta = torch.linspace(beta_start,beta_end,T,dtype=torch.float32)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # for forward process
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat) # for mean
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat) #for std_dev

        # for sampling process
        self.one_by_sqrt_alpha = 1. / torch.sqrt(self.alpha) # for mean
        self.one_by_sqrt_one_minus_alpha_hat = 1. / self.sqrt_one_minus_alpha_hat # for mean
        self.sqrt_beta = torch.sqrt(self.beta) # for std_dev
        
    def forward(self, x0, t):
        noise = torch.randn_like(x0).to(x0.device)
        # Reshaping
        sqrt_alpha_hat = self.sqrt_alpha_hat.to(x0.device)[t]
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat.to(x0.device)[t]

        # reshape to match no of dims (b,) -> (b,c,h,w)
        for _ in range(len(x0.shape) - 1):
            sqrt_alpha_hat = sqrt_alpha_hat.unsqueeze(-1)
            sqrt_one_minus_alpha_hat = sqrt_one_minus_alpha_hat.unsqueeze(-1)

        mean = sqrt_alpha_hat.to(x0.device) * x0
        std_dev = sqrt_one_minus_alpha_hat.to(x0.device)

        sample = mean + std_dev * noise 

        return sample, noise # noise --> predicted by the model
    
    def backward(self,xt,noise_pred,t):
        # Reshaping
        one_by_sqrt_alpha = self.one_by_sqrt_alpha.to(xt.device)[t]
        beta = self.beta.to(xt.device)[t]
        one_by_sqrt_one_minus_alpha_hat = self.one_by_sqrt_one_minus_alpha_hat.to(xt.device)[t]
        sqrt_beta = self.sqrt_beta.to(xt.device)[t]

        mean = one_by_sqrt_alpha * (xt - beta * one_by_sqrt_one_minus_alpha_hat * noise_pred)
        std_dev = sqrt_beta
        
        if t==0:
            return mean
        else:
            z = torch.randn_like(xt).to(xt.device)
            return mean + std_dev * z


