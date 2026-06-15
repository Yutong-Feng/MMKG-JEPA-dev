import torch
import torch.nn.functional as F

def lejepa_loss(pred_embed, tail_embed, a, regularization):
    tail_frozen = tail_embed.detach() 
    rec_loss = (1.0 - F.cosine_similarity(pred_embed, tail_frozen, dim=-1)).mean()
    reg_loss = regularization(torch.cat([pred_embed, tail_frozen], dim=0)) 
    return rec_loss + a * reg_loss, reg_loss