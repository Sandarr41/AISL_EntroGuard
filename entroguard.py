import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def entropy_loss(block_dists):
    """eq.(8): средняя энтропия промежуточных (logit-lens) распределений
    слов, усреднённая по длине последовательности и по обучаемым
    блокам/слоям. Чем выше — тем лучше для приватности (во время обучения
    мы хотим её МАКСИМИЗИРОВАТЬ)."""
    total = 0.0
    for d in block_dists:
        ent = -(d * (d.clamp_min(1e-9)).log()).sum(-1)   # (B,T)
        total = total + ent.mean()
    return total / len(block_dists)


def cross_entropy_loss(logits, target_ids, pad_id):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1),
        ignore_index=pad_id,
    )


def similarity_loss(e0, e1):
    """eq.(9): 1 - косинусное сходство."""
    return 1 - F.cosine_similarity(e0, e1, dim=-1)


def rescale_to_e0_norm(e0, e_prime, eps=1e-8):
    """Перенормирует сырой выход G_theta к ||e0|| ещё до того, как его
    увидит любой лосс.

    similarity_loss считается через косинусное сходство, то есть
    инвариантна к масштабу: ничто в eq.(10) не штрафует ||e_prime||
    напрямую, поэтому градиентный спуск может (и на практике этим
    пользуется) взорвать ||e_prime|| до любой величины, лучше всего
    сбивающей атакующего с толку, — возмущение получается на порядки
    больше заявленного epsilon-шара. bound_aware_adaptation (Algorithm 1)
    потом не успевает спроецировать его обратно за ограниченное число
    итераций — алгоритм рассчитан на "разумно" отмасштабированный вход.
    Перенормировка здесь заставляет генератор соревноваться только
    НАПРАВЛЕНИЕМ — именно это и должна формировать eq.(10)."""
    return e0.norm(dim=-1, keepdim=True) * F.normalize(e_prime, dim=-1, eps=eps)


def entroguard_train_loss(attacker, e0, e_prime, input_ids, target_ids, pad_id,
                           alpha=6.0, beta=1.0, gamma=1.0):
    """eq.(10): min  alpha*L_sim - beta*L_entropy - gamma*L_CE"""
    e_prime = rescale_to_e0_norm(e0, e_prime)
    logits, block_outs = attacker(e_prime, input_ids, return_block_outputs=True)
    dists = attacker.block_distributions(block_outs)
    l_ent = entropy_loss(dists)
    l_ce = cross_entropy_loss(logits, target_ids, pad_id)
    l_sim = similarity_loss(e0, e_prime).mean()
    loss = alpha * l_sim - beta * l_ent - gamma * l_ce
    return loss, {"sim": l_sim.item(), "entropy": l_ent.item(), "ce": l_ce.item()}


# ---------------------------------------------------------------------------
# Algorithm 1: Bound-aware Perturbation Adaptation
# ---------------------------------------------------------------------------
@torch.no_grad()
def bound_aware_adaptation(e0, e_prime, epsilon=0.15, lam=0.95, max_iter=60):
    """
    e0, e_prime: (B, d) исходный и возмущённый (энтропией) эмбеддинги.
    epsilon: граница возмущения по косинусному расстоянию.
    lam: коэффициент масштабирования (<1), используется и для наращивания
         разреженного шума, и для уменьшения избыточного шума — в точности
         как в алгоритме статьи.
    """
    B, d = e0.shape
    device = e0.device
    rho = 1 - F.cosine_similarity(e0, e_prime, dim=-1)  # (B,)
    e_out = e_prime.clone()

    below = rho < epsilon
    above = ~below

    # --- случай 1: возмущение ниже границы -> добавляем случайный шум на
    # случайное подмножество измерений, затем уменьшаем именно этот
    # *добавленный* шум, пока не вернёмся к границе eps
    if below.any():
        idx = below.nonzero(as_tuple=True)[0]
        k = max(1, d // 4)
        for i in idx.tolist():
            dims = torch.randperm(d, device=device)[:k]
            sign = torch.randint(0, 2, (k,), device=device) * 2 - 1  # +-1
            noise = torch.randn(k, device=device).abs() * sign.float() + sign.float()
            n = torch.zeros(d, device=device)
            n[dims] = noise
            e2 = e_prime[i] + n
            cur_rho = 1 - F.cosine_similarity(e0[i:i+1], e2.unsqueeze(0)).item()
            it = 0
            while cur_rho > epsilon and it < max_iter:
                n = n * lam
                e2 = e_prime[i] + n
                cur_rho = 1 - F.cosine_similarity(e0[i:i+1], e2.unsqueeze(0)).item()
                it += 1
            e_out[i] = e2

    # --- случай 2: возмущение выше границы -> масштабируем весь вектор
    # возмущения (e_prime - e0) вниз экспоненциально, пока не впишемся в границу
    if above.any():
        idx = above.nonzero(as_tuple=True)[0]
        for i in idx.tolist():
            I_noise = (e_prime[i] - e0[i]).clone()
            e2 = e0[i] + I_noise
            cur_rho = 1 - F.cosine_similarity(e0[i:i+1], e2.unsqueeze(0)).item()
            it = 0
            while cur_rho > epsilon and it < max_iter:
                I_noise = I_noise * lam
                e2 = e0[i] + I_noise
                cur_rho = 1 - F.cosine_similarity(e0[i:i+1], e2.unsqueeze(0)).item()
                it += 1
            e_out[i] = e2

    return e_out


# ---------------------------------------------------------------------------
# Базовые защиты (точки сравнения из статьи)
# ---------------------------------------------------------------------------
def gaussian_noise_defense(e0, sigma=0.15):
    return e0 + torch.randn_like(e0) * sigma


def pgd_defense(attacker, e0, target_ids, input_ids, pad_id, epsilon=0.15,
                 step_size=0.05, n_steps=10):
    """Оптимизационное (white-box) возмущение: градиентный подъём по loss
    реконструкции атакующего, с проекцией обратно в epsilon-шар по
    косинусному расстоянию вокруг e0 после каждого шага (стандартный PGD)."""
    e = e0.clone().detach().requires_grad_(True)
    for _ in range(n_steps):
        logits = attacker(e, input_ids)
        loss = cross_entropy_loss(logits, target_ids, pad_id)
        grad = torch.autograd.grad(loss, e)[0]
        with torch.no_grad():
            e = e + step_size * grad.sign()
            rho = 1 - F.cosine_similarity(e0, e, dim=-1, eps=1e-8)
            over = rho > epsilon
            if over.any():
                # проекция: уменьшаем возмущение для строк, вышедших за границу
                direction = (e - e0)
                scale = torch.ones(e.size(0), device=e.device)
                for i in over.nonzero(as_tuple=True)[0].tolist():
                    d = direction[i].clone()
                    r = 1 - F.cosine_similarity(e0[i:i+1], (e0[i] + d).unsqueeze(0)).item()
                    it = 0
                    while r > epsilon and it < 40:
                        d = d * 0.9
                        r = 1 - F.cosine_similarity(e0[i:i+1], (e0[i] + d).unsqueeze(0)).item()
                        it += 1
                    e[i] = e0[i] + d
        e = e.clone().detach().requires_grad_(True)
    return e.detach()
