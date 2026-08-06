import json
import os

file_path = '코드 파일/cfg_starter (30)_2.ipynb'

new_source = '''# Samplers
@torch.no_grad()
def sample_unconditional(n=64, steps=None):
    """
    TODO (C): Unconditional ancestral DDPM sampling (no guidance).
      Use the unconditional null label at every step.
      Pseudocode:
        x = N(0,I)
        for t in reversed(range(T)):
            use your unet to get eps_hat
            use, x, eps_hat to get mu and var
            if t > 0: x = mu + sqrt(var) * z, z~N(0,1)
            else:     x = mu
        return x.clamp(-1,1)
    """
    net.eval()
    T = sched.T if steps is None else steps
    # 1. Initialize with pure random noise (x_T)
    x = torch.randn(n, 1, 28, 28, device=device)
    # Fill with null_id for unconditional sampling
    y_null = torch.full((n,), net.null_id, device=device, dtype=torch.long)
    # 2. Iterate backward from T-1 to 0
    for ti in reversed(range(T)):
        t = torch.full((n,), ti, device=device, dtype=torch.long)
        # use your unet to get eps_hat (estimated error)
        eps_hat = net(x, t, y_null)

        # then get mu and var of the previous step, p(x(t-1)|x_t) (hint: sched.posterior_mean_variance())
        mu, var = sched.posterior_mean_variance(x, eps_hat, t)

        # 5. Denoise step by step to update x (Langevin dynamics)
        if ti > 0:
            x = mu + torch.sqrt(var) * torch.randn_like(x)
        else:
            x = mu
    return x.clamp(-1,1)


@torch.no_grad()
def sample_conditional(label, gamma=3.0, n=64, steps=None):
    """
    TODO (D): Conditional sampling with classifier-free guidance.
    """
    net.eval()
    T = sched.T if steps is None else steps
    # 1. Initialize with pure random noise
    x = torch.randn(n, 1, 28, 28, device=device)
    # Prepare conditional label (y_lab) and unconditional null label (y_null)
    y_lab  = torch.full((n,), int(label), device=device, dtype=torch.long)
    y_null = torch.full((n,), net.null_id, device=device, dtype=torch.long)

    for ti in reversed(range(T)):
        t = torch.full((n,), ti, device=device, dtype=torch.long)
        \'\'\'
        your code here
        1. use your unet to get eps_c, eps_u (estimated error), you should inference twice for both conditional and unconditional
        2. get eps_hat by following CFG weighting: (1.0 + gamma) * eps_c - gamma * eps_u
        3. then get mu and var of the previous step, p(x(t-1)|x_t) (hint: sched.posterior_mean_variance())
        \'\'\'
        # 1) Call U-Net twice for conditional and unconditional noise prediction
        eps_c = net(x, t, y_lab)
        eps_u = net(x, t, y_null)

        # 2) Calculate final eps_hat using CFG formula
        eps_hat = (1.0 + gamma) * eps_c - gamma * eps_u

        # 3) Calculate posterior mean and variance for the previous step
        mu, var = sched.posterior_mean_variance(x, eps_hat, t)

        # 4) Denoise step by step to update x
        if ti > 0:
            x = mu + torch.sqrt(var) * torch.randn_like(x)
        else:
            x = mu
    return x.clamp(-1,1)
'''

new_source_lines = [line + '\n' for line in new_source.split('\n')]
if new_source_lines:
    new_source_lines[-1] = new_source_lines[-1][:-1]

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for cell in data.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        if 'def sample_unconditional' in source and 'def sample_conditional' in source:
            cell['source'] = new_source_lines
            break

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1, ensure_ascii=False)

print(f"Updated {file_path}")
