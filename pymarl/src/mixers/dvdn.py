import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils.projection import proj
from utils.conv import bconv
import numpy as np

class DVDNMixer(nn.Module):
    def __init__(self, args):
        super(DVDNMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.n_atom = args.n_atom
        self.v_min = args.v_min
        self.v_max = args.v_max
        self.value_range = np.linspace(args.v_min, args.v_max, args.n_atom)
        self.value_range = th.FloatTensor(self.value_range)
        self.q_tot_value_range1 = np.linspace(args.n_agents * args.v_min, args.n_agents * args.v_max, 
                                             args.n_agents * args.n_atom - args.n_agents + 1)
        self.q_tot_value_range1 = th.FloatTensor(self.q_tot_value_range1)
        if args.use_cuda:
            self.value_range = self.value_range.cuda()
            self.q_tot_value_range1 = self.q_tot_value_range1.cuda()

    def forward(self, agent_qs_distri, states):
        #agent_qs_distri: [bs, episode_len, n_agents, n_atom]
        #states.size(): [bs, episode_len, state_shape]
        bs = agent_qs_distri.size(0)
        episode_len = agent_qs_distri.size(1)
        n_agents = agent_qs_distri.size(2)
        q_tot = agent_qs_distri[:, :, 0]
        for i in range(1, n_agents):
            q_tot = bconv(q_tot, agent_qs_distri[:, :, i])
        
        q_tot_value_range = self.q_tot_value_range1.unsqueeze(0).unsqueeze(0).expand(bs, episode_len, -1) #[bs, episode_len, N_ATOM]

        q_tot = proj(q_tot_value_range, q_tot, self.v_min, self.v_max, self.n_atom)

        return q_tot  #[bs, episode_len, n_atom]

