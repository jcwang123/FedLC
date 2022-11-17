import numpy as np
import torch
from torch.autograd import Variable


def set_global_grad(net, keys, tag):
    for name, param in net.named_parameters():
        if name[:2] == 'hc':
            continue
        if name in keys:
            param.requires_grad = (tag == 1)
        else:
            param.requires_grad = (tag == 0)


def check_equal(net_clients):
    client_num = len(net_clients)
    for param in zip(net_clients[0].parameters(), net_clients[1].parameters(),
                     net_clients[2].parameters(), net_clients[3].parameters()):
        for i in range(1, client_num):
            assert torch.max(param[i].data - param[i - 1].data) == 0


def update_global_model(net_clients, client_weight):
    client_num = len(net_clients)
    if len(net_clients) == 4:
        iter_container = zip(net_clients[0].parameters(),
                             net_clients[1].parameters(),
                             net_clients[2].parameters(),
                             net_clients[3].parameters())
    elif len(net_clients) == 6:
        iter_container = zip(net_clients[0].parameters(),
                             net_clients[1].parameters(),
                             net_clients[2].parameters(),
                             net_clients[3].parameters(),
                             net_clients[4].parameters(),
                             net_clients[5].parameters())

    for param in iter_container:
        new_para = Variable(torch.Tensor(np.zeros(param[0].shape)),
                            requires_grad=False).cuda()
        for i in range(client_num):
            new_para.data.add_(client_weight[i], param[i].data)

        for i in range(client_num):
            param[i].data.mul_(0).add_(new_para.data)


def update_global_model_with_keys(net_clients, client_weight, keys):
    client_num = len(net_clients)
    if len(net_clients) == 4:
        iter_container = zip(net_clients[0].named_parameters(),
                             net_clients[1].named_parameters(),
                             net_clients[2].named_parameters(),
                             net_clients[3].named_parameters())
    elif len(net_clients) == 6:
        iter_container = zip(net_clients[0].named_parameters(),
                             net_clients[1].named_parameters(),
                             net_clients[2].named_parameters(),
                             net_clients[3].named_parameters(),
                             net_clients[4].named_parameters(),
                             net_clients[5].named_parameters())

    for data in iter_container:
        name = [d[0] for d in data]
        param = [d[1] for d in data]
        if not name[0] in keys:
            continue
        new_para = Variable(torch.Tensor(np.zeros(param[0].shape)),
                            requires_grad=False).cuda()
        for i in range(client_num):
            new_para.data.add_(client_weight[i], param[i].data)

        for i in range(client_num):
            param[i].data.mul_(0).add_(new_para.data)


# prr-fl functions


def avg_freq(weights, L=0.1, is_conv=True):
    client_num = len(weights)

    if is_conv:
        N, C, D1, D2 = weights[0].size()
    else:
        N = 1
        C = 1
        D1, D2 = weights[0].size()

    temp_low = np.zeros((C * D1, D2 * N), dtype=float)
    for i in range(client_num):
        if is_conv:
            weights[i] = weights[i].permute(1, 2, 3, 0).reshape(
                (C * D1, D2 * N))
        weights[i] = weights[i].cpu().numpy()

        client_fft = np.fft.fft2(weights[i], axes=(-2, -1))
        amp_fft, pha_fft = np.abs(client_fft), np.angle(client_fft)  # FFT
        low_part = np.fft.fftshift(amp_fft, axes=(-2, -1))
        temp_low += low_part
    temp_low = temp_low / 4  # avg the low-frequency

    for i in range(client_num):
        client_fft = np.fft.fft2(weights[i], axes=(-2, -1))
        amp_fft, pha_fft = np.abs(client_fft), np.angle(client_fft)
        low_part = np.fft.fftshift(amp_fft, axes=(-2, -1))

        h, w = low_part.shape
        b_h = (np.floor(h * L / 2)).astype(int)
        b_w = (np.floor(w * L / 2)).astype(int)
        c_h = np.floor(h / 2.0).astype(int)
        c_w = np.floor(w / 2.0).astype(int)

        h1 = c_h - b_h
        h2 = c_h + b_h
        w1 = c_w - b_w
        w2 = c_w + b_w
        low_part[h1:h2, w1:w2] = temp_low[
            h1:h2, w1:w2]  # averaged low-freq + individual high-freq
        low_part = np.fft.ifftshift(low_part, axes=(-2, -1))

        fft_back_ = low_part * np.exp(1j * pha_fft)  #
        # get the mutated image
        fft_back_ = np.fft.ifft2(fft_back_, axes=(-2, -1))
        weights[i] = torch.FloatTensor(np.real(fft_back_))
        if is_conv:
            weights[i] = weights[i].reshape(C, D1, D2, N).permute(3, 0, 1, 2)
    return weights


def PFA(weights, L, is_conv):
    return avg_freq(weights=weights, L=L, is_conv=is_conv)


def communication(server_model, models, client_weights, a_iter):
    l_rate = 0.7
    iters = 200
    pfa_rate = l_rate + (a_iter / iters) * (0.95 - l_rate)
    client_num = len(client_weights)  #
    with torch.no_grad():
        # aggregate params
        for key in server_model.state_dict().keys():
            if 'bn' not in key:  #not bn
                if 'conv' in key and 'weight' in key:
                    cur_weights = [
                        models[kk].state_dict()[key].data
                        for kk in range(client_num)
                    ]
                    temp_weights = PFA(cur_weights, L=pfa_rate, is_conv=True)
                    for client_idx in range(
                            client_num):  # copy from server to each client
                        models[client_idx].state_dict()[key].data.copy_(
                            temp_weights[client_idx])
                elif 'linear' in key and 'weight' in key:
                    cur_weights = [
                        models[kk].state_dict()[key].data
                        for kk in range(client_num)
                    ]
                    temp_weights = PFA(cur_weights, L=pfa_rate, is_conv=False)
                    for client_idx in range(client_num):  #
                        models[client_idx].state_dict()[key].data.copy_(
                            temp_weights[client_idx])
                else:
                    print(key, '\t not bn, conv, fc layer, with param!')
                    temp = torch.zeros_like(server_model.state_dict()[key],
                                            dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[
                            client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(
                        temp)  # non-bn layer，update the server model
                    for client_idx in range(
                            client_num
                    ):  # non-bn layer, from server to each client
                        models[client_idx].state_dict()[key].data.copy_(
                            server_model.state_dict()[key])

    return server_model, models