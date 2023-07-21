import torch
from torch.nn.functional import adaptive_avg_pool2d, pairwise_distance


class TripletMaringLoss():
    def __init__(self,margin,type_loss='default'):
        self.margin = margin
        self.type_loss = type_loss

        if(type_loss == 'default'):
            return self.margin_triplet_loss

    def margin_triplet_loss(self,anchor: torch.tensor, positive: torch.tensor, negative: torch.tensor,reduction='mean'):

        anchor = adaptive_avg_pool2d(anchor, 1).reshape(-1,anchor.shape[1])
        positive = adaptive_avg_pool2d(positive, 1).reshape(positive.shape[0],positive.shape[1],-1)
        negative = adaptive_avg_pool2d(negative, 1).reshape(negative.shape[0],negative.shape[1],-1)

        aux = []

        for anchor_, positive_, negative_ in zip(anchor,positive,negative):
            loss_ = torch.clamp(
                (pairwise_distance(anchor_,positive_,p=self.margin) - pairwise_distance(anchor_,negative_,p=self.margin) + 1),
                0
            )

            aux.append(loss_)

        loss = torch.concat(aux)
        loss.requires_grad_()

        if(reduction == 'mean'):
            return loss.mean()
        elif(reduction == 'sum'):
            return loss.mean()
        elif(reduction == 'sqrt'):
            return torch.sqrt(torch.mean(torch.square(loss)))
        
        else:
            return Exception(f'Reduction not specify: {reduction}')

