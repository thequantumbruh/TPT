import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


state_dict = torch.load('/content/drive/MyDrive/tpt_checkpoint1.pt', map_location=device)
#model_v1 = TPT_v1(tpt_config)
#model_v1.to(device)
#model_v1.load_state_dict(state_dict['model_state_dict'] if 'model_state_dict' in state_dict else state_dict)
#model_v1.eval()


tok = AutoTokenizer.from_pretrained('/content/drive/MyDrive/Temi_pretrained_model_tknzr')

@torch.no_grad()
def generate(model, tok, prompt, max_new=120, temp=0.8, top_k=40, top_p=0.9, rep_pen=1.15, ctx=512):
    model.eval()
    ids = torch.tensor([tok.encode(prompt[0])], device=device)
    for _ in range(max_new):
        lg = model(ids[:, -ctx:])[:, -1, :].float()
        for t in set(ids[0, -64:].tolist()):                 
            lg[0, t] = lg[0, t] / rep_pen if lg[0, t] > 0 else lg[0, t] * rep_pen
        lg = lg / temp
        lg[lg < torch.topk(lg, top_k).values[:, -1:]] = -float("inf")
        p = lg.softmax(-1)
        sp, si = p.sort(descending=True)
        sp[(sp.cumsum(-1) - sp) > top_p] = 0                 
        nxt = si.gather(-1, torch.multinomial(sp / sp.sum(), 1))
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == 0: break
    return tok.decode(ids[0].tolist())
