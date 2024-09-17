"""    Has a function 'find accuracy' for the DNN (unnecessary because the same function is in simulation_utils)"""import torchimport torch.distributions as distclass TransDist:    def __init__(self, r_min=0, r_max=2985, Nr=200, vr_min=-369.1406, vr_max=369.1406, Nv=64, T=1, sigma_r=0.00001, sigma_v=0.00001):        self.range_vec = torch.linspace(r_min,r_max,Nr)        self.vr_vec = torch.linspace(vr_min,vr_max,Nv)        self.r_bin = float((self.range_vec[1]-self.range_vec[0])/2)        self.v_bin = float((self.vr_vec[1]-self.vr_vec[0])/2)        self.T = T        self.Nr = Nr        self.Nv = Nv        self.log_prob_r = torch.zeros([self.Nr])        self.log_prob_v = torch.zeros([self.Nv])        self.sigma_r = sigma_r        self.sigma_v = sigma_v        dist_r = dist.Normal(0, torch.tensor(sigma_r))        dist_v = dist.Normal(0, torch.tensor(sigma_v))        self.dist_r = dist_r        self.dist_v = dist_v        for r in range(self.Nr):            r_max = torch.tensor((2*r+1) * self.r_bin)            r_min = torch.tensor((2*r-1) * self.r_bin)            prob = (dist_r.cdf(r_max) - dist_r.cdf(r_min))            self.log_prob_r[r] = torch.log(prob+0.000000001)        for v in range(self.Nv):            v_max = torch.tensor((2*v+1)*self.v_bin)            v_min = torch.tensor((2*v-1)*self.v_bin)            prob = (dist_v.cdf(v_max) - dist_v.cdf(v_min))            self.log_prob_v[v] += torch.log(prob+0.000000001)    def backward_step(self, r, v, tracks):        N = len(tracks)        costs = torch.zeros([N])        r_idx, v_idx = self.ind2val(r, v)        r_prev, v_prev = self.prev(r_idx, v_idx)        r_prev, v_prev = self.val2idx(r_prev, v_prev)        for i in range(N):            track = tracks[i]            r_diff = int(abs(r_prev-track.r))            v_diff = int(abs(v_prev-track.v))            costs[i] = track.cost + self.log_prob_r[r_diff] + self.log_prob_v[v_diff]        prev_idx = torch.argmax(costs).item()        return prev_idx, costs[prev_idx]    def val2idx(self, r_val, v_val):        r = torch.argmin(torch.abs(self.range_vec-r_val))        v = torch.argmin(torch.abs(self.vr_vec-v_val))        return int(r), int(v)    def ind2val(self, r, v):        r_val = self.range_vec[r]        v_val = self.vr_vec[v]        return r_val,v_val    def next(self, r, v, rnd=False):        if rnd:            dr = torch.randn(1)*self.sigma_r            dv = torch.randn(1)*self.sigma_v        else:            dr = 0            dv = 0        r_next = r + v*self.T + dr        v_next = v + dv        return r_next,v_next    def prev(self, r, v):        v_prev = v        r_prev = r - v * self.T        return r_prev, v_prevclass Track:    def __init__(self, r, v, cost=0, prev=None):        self.r = r        self.v = v        if prev is None:            self.track = list()        else:            self.track = prev.track.copy()        self.cost = cost        self.track.append((r,v))class Particle:    def __init__(self, state_val, state_idx, weight, prev=None):        self.r = state_val[0]        self.v = state_val[1]        self.r_idx = state_idx[0]        self.v_idx = state_idx[1]        if prev is None:            self.track = list()        else:            self.track = prev.track.copy()        self.weight = weight        self.track.append(state_idx)def beam_mask(emis_k, beta=0.5):    val_max = torch.max(emis_k)    val_min = torch.min(emis_k)    thresh = val_min + ((val_max - val_min) * beta)    mask = (emis_k >= thresh)    return maskdef masked_argmax(original_tensor, mask):    Nv = original_tensor.size(1)    Nr = original_tensor.size(0)    val = float('-inf')    r_best = -1    v_best = -1    for r in range(Nr):        for v in range(Nv):            if mask[r, v]:                if original_tensor[r,v] >= val:                    val = original_tensor[r,v]                    r_best = int(r)                    v_best = int(v)    return r_best, v_bestdef Find_Accuracy(track, label, name=None):    acc = 0.0    track_len = len(label)    spm = 0.0    dist = 0.0    for k in range(track_len):        r = label[k][0][0].item()        v = label[k][1][0].item()        if (r, v) == track[k]:            acc += 1        dr = abs(r-track[k][0])        dv = abs(v-track[k][1])        dist += dr + dv        if max(dr,dv) == 1:            spm += 1    acc /= track_len    spm /= track_len    dist /= track_len    if name is not None:        print(f'{name} accuracy =  {acc*100:.2f}%')        print(f'{name} soft accuracy =  {(spm + acc) * 100:.2f}%')        print(f'{name} average distance =  {dist:.2f} pixels')    return acc, acc + spm, distdef get_top_n_mask(tensor, mask=None, N=100):    # Check if N is None or greater than the number of True values in the mask    if mask is None:        # Apply the mask to the tensor        masked_tensor = tensor[mask]    else:        masked_tensor = tensor        if N is None or N > sum(sum(mask)):            return mask  # Return the original mask    # Find the indices of the N highest values in the masked tensor    top_indices = torch.topk(masked_tensor, k=N)[1]    # Create a new mask with the N highest values    new_mask = torch.zeros_like(mask)    new_mask[mask] = torch.isin(masked_tensor, masked_tensor[top_indices])    return new_mask