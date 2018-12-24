# TO DO
# Plot predicted output from training and make sure it's not complete crap


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
import glob
import matplotlib.pyplot as plt
import math
import time

import matlab.engine
import scipy.signal
eng = matlab.engine.start_matlab()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pdb

def one_hot(data, alphabet):
    return (np.arange(len(alphabet)+1) == data[...,None]).astype(int)

import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# def indexes_to_number(indexes):

def readLangs(filter):
    print("Reading data...")

    Fs = 1/(10/1000)
    Wn1 = 5/(Fs/2)
    Wn2 = .8/(Fs/2)

    bl, al = scipy.signal.butter(N=4, Wn=Wn1, btype='low')
    bh, ah = scipy.signal.butter(N=4, Wn=Wn2, btype='high')
    # Read the file and split into lines
    start_time = 0;
    cycle_mat = [];
    data_fin = np.empty((0,7))
    strides_bool_fin = np.empty(0)
    for f in glob.glob("*.csv"):
        print(f)
        x = np.genfromtxt(f,delimiter=',', skip_header=1)[1:,[1,3,4,5,6,7,8]]
        x[:,1:] = x[:,1:] - np.mean(x[:,1:], axis = 0)
        fin_time = np.arange(start_time, x[-1,0]+start_time, x[-1,0]/x.shape[0])
        fin_time = fin_time[:x.shape[0]]
        x = x[:len(fin_time),:]
        # if len(fin_time) != x.shape[0]:
        x_0 = np.empty(x.shape)

        x_int = np.empty(x.shape)
        for i in range(1,x.shape[1]):
            x_int[:,i] = np.interp(fin_time, x[:,0]+start_time, x[:,i])
            x_0[:,i] = x_int[:,i];
            if filter:
                x_int[:,i] = scipy.signal.filtfilt(bl, al, x_int[:,i])
            # high pass filtering
                x_int[:,i] = scipy.signal.filtfilt(bh, ah, x_int[:,i])

        avg_int = np.mean(fin_time[2:-1]-fin_time[1:-2])
        for foot = 1:2
            strt = (foot-1)*2 + 1
            sig = np.sqrt(np.sum(np.square(np.divide(x_int[:,strt:strt+3], x_int[:,strt:strt+3].max(axis=0))),axis=1))
            if f[0] == 'S':
                pk_perc = 50
                pk_dist = 130
            else:
                pk_perc = 90
                pk_dist = 90
            x_int[:,0] = fin_time
            sig_in = matlab.double(list(sig))
            sig_in2 = matlab.double(list(sig[1:750]))
            strides, locs = eng.findpeaks(sig_in,"MinPeakHeight",float(np.percentile(sig,pk_perc)),"MinPeakDistance",pk_dist,nargout=2)
            strides2, locs2 = eng.findpeaks(sig_in2,"MinPeakHeight",float(np.percentile(sig[1:750],97)),nargout=2)
            calib_pt[foot] = np.mean(locs2)

            strides = np.array(strides)
            locs = np.array(locs).astype(int)
            locs2 = np.array(locs2)
            strides[foot] = strides[locs>1000]
            locs[foot] = locs[locs>1000]

        # plt.plot(fin_time,np.squeeze(sig_in))
        # plt.scatter(fin_time[locs],strides)

        strides_bool = np.zeros(len(fin_time))
        strides_bool[locs] = np.ones(len(locs))

        calib_pt = round(np.mean(locs2))
        keep_inds = np.zeros(1,locs[0])
        for l, sloc in enumerate(locs[1:]):
            s_to_s = abs(sloc-locs[l])+1
            if s_to_s*avg_int > 2200:
                keep_inds = np.append(keep_inds, np.zeros(s_to_s+1))
            else:
                keep_inds = np.append(keep_inds, np.ones(s_to_s+1))
                cycle_mat = np.append(cycle_mat, np.arange(s_to_s+1)/s_to_s)
        keep_inds = np.append(keep_inds, np.zeros(len(fin_time)-len(keep_inds)))
        keep_inds = np.where(keep_inds)[0]
        data = x_int[keep_inds,:]
        data_fin = np.vstack((data_fin, data))
        strides_bool = strides_bool[keep_inds]
        strides_bool_fin = np.append(strides_bool_fin, strides_bool)
        start_time = fin_time[-1] + avg_int
    return data_fin, strides_bool, cycle_mat


class DataProcessor():
    def __init__(self,training_sigs=3, bin_step=100, SOS=-1, EOS = 0, one_hot = False, filter = False):
        self.training_sigs = training_sigs
        self.bin_step = bin_step
        self.SOS = SOS
        self.EOS = EOS
        self.one_hot = one_hot
        self.filter = filter

    def input_data(self):
        self.data, self.strides, self.cycle = readLangs(self.filter);
    # plt.show()

    def number_to_index(self, numbers):
        return {j:i for i,j in enumerate(np.sort(numbers))}

    def convert_w_dict(self, numbers, my_dict):
        return [my_dict[n] for n in numbers]

    def process_data(self):
        self.avg_int = np.mean(self.data[:,2:-1] - self.data[:,1:-2])
        self.data[:,1:] = self.data[:,1:] - np.min(self.data[:,1:],axis=0)
        # self.data = self.data+min(self.data)

        self.stride_idxs = np.where(self.strides.flatten())[0]
        self.cycle_time = self.stride_idxs[1:]-self.stride_idxs[0:-1]

        self.text = self.data[:,self.training_sigs]

        binned_txt = np.floor((self.text - min(self.text))/self.bin_step) + 1
        self.text = binned_txt

        self.alphabet = np.append(np.unique(self.text),[self.SOS,self.EOS])
        self.alphabet_dict = self.number_to_index(self.alphabet)
        if self.one_hot:
            self.text = one_hot(self.text, self.alphabet)

        self.labels = [np.round(np.arange(ct)/ct,2) + .01 for ct in self.cycle_time]
        self.training = [self.text[self.stride_idxs[i]:self.stride_idxs[i+1]] for i in range(len(self.stride_idxs)-1)]

        self.seqlenx = int(np.percentile(self.cycle_time,75))
        self.seqleny = self.seqlenx

        self.num_words_in = np.append(np.unique(np.concatenate(self.training)),[self.SOS,self.EOS])
        self.num_words_out = np.append(np.unique(np.concatenate(self.labels)),[self.SOS,self.EOS])

        self.words_in_dict = self.number_to_index(self.num_words_in)
        self.words_out_dict = self.number_to_index(self.num_words_out)

        self.in_invmap = {v: k for k, v in self.words_in_dict.items()}
        self.out_invmap = {v: k for k, v in self.words_out_dict.items()}

        self.training = [self.convert_w_dict(t, self.words_in_dict) for t in self.training]
        self.labels = [self.convert_w_dict(l, self.words_out_dict) for l in self.labels]

    def split_tst_tr(self,tr_perc):
        num_pairs = len(self.cycle_time)
        self.id1 = random.sample(range(num_pairs),math.floor(tr_perc*num_pairs))
        self.id2 = list(set(range(num_pairs))-set(self.id1))
        # import pdb; pdb.set_trace()

        self.x_tr = list(np.array(self.training)[self.id1])
        self.y_tr = list(np.array(self.labels)[self.id1])

        self.x_tst = np.array(self.training)[self.id2]
        self.y_tst = np.array(self.labels)[self.id2]

        self.tr_pairs = zip(dat.x_tr, dat.y_tr)
        self.tst_pairs = zip(dat.x_tst, dat.y_tst)
        # zip(x_tst, y_tst)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, do_one_hot = False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.do_one_hot = do_one_hot
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.input_size = input_size
        # if do_one_hot:
        self.gru = nn.GRU(hidden_size, hidden_size)
    # else:
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden):
        input = input.type(torch.long)
        input = input.unsqueeze(0)
        # if self.do_one_hot:
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        # else:
        #     output = input
        #     output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p, max_length ,do_one_hot = False):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.do_one_hot = do_one_hot

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # if self.do_one_hot:
        embedded = self.embedding(input).view(1, 1, -1)

        embedded = self.dropout(embedded)
        input = embedded[0]

        attn_weights = F.softmax(
            self.attn(torch.cat((input, hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((input, attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
            decoder_optimizer, criterion, max_length, teacher_forcing_ratio=.2):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[dat.words_out_dict[dat.SOS]]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    output_tracker1=[]
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            # import pdb; pdb.set_trace()
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            v, i = decoder_output.max(-1)
            output_tracker1.append(i.item())
            # import pdb; pdb.set_trace()
            loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
            decoder_input = target_tensor[di]  # Teacher forcing
            output_tracker1.append(decoder_output)

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            # import pdb; pdb.set_trace()
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            v, i = decoder_output.max(-1)
            # import pdb; pdb.set_trace()
            loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
            output_tracker1.append(i.item())
            if decoder_input.item() == dat.words_out_dict[dat.EOS]:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length, output_tracker1

def trainIters(encoder, decoder, epochs, x_tr, y_tr, max_length,
                print_every=10, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    # criterion = nn.MSELoss()
    criterion = nn.NLLLoss()
    # lang.split_tst_tr(.8)

    for iter in range(1, epochs + 1):
        if iter == 1:
            start = 0
        for x_in, y_in in zip(x_tr, y_tr):
            k = max_length-len(x_in);
            if k < 0:
                x_in = x_in[:max_length]
                y_in = y_in[:max_length]
            elif k>0:
                x_in = np.append(x_in, dat.convert_w_dict(np.zeros(k),dat.words_in_dict))
                y_in = np.append(y_in, dat.convert_w_dict(np.zeros(k),dat.words_out_dict))
            # import pdb; pdb.set_trace()
            input_tensor = torch.tensor(x_in)
            target_tensor = torch.tensor(y_in)
            loss, outputs = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer,
                         criterion, max_length)
            print_loss_total += loss
            plot_loss_total += loss

            # if iter == epochs+1:
            #     l3=plt.plot(range(start+len(x_in),len(x_in)+start+len(outputs)),outputs, color = 'b')
        # plt.legend((l1,l2,l3),('Input data','Labels','Model Guess'))

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            y_in0 = dat.convert_w_dict(y_in, dat.out_invmap)
            # try:
            outputs0 = dat.convert_w_dict(outputs, dat.out_invmap)
            # except:
            #     import pdb; pdb.set_trace()
            print('Targets: ', [str(y) for y in y_in])
            print('Guesses: ', [str(o) for o in outputs])
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / epochs),
                                         iter, iter / epochs * 100, print_loss_avg))
            plt.show()

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    # plt.show()
    showPlot(plot_losses)

def evaluate(encoder, decoder, x_in, EOS_token, max_length):
    with torch.no_grad():
        input_tensor = torch.tensor(x_in)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([dat.words_out_dict[dat.SOS]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == dat.words_out_dict[dat.EOS]:
                decoded_words.append(dat.words_out_dict[dat.EOS])
                break
            else:
                decoded_words.append(topi.item())

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

hidden_size = 256
dat = DataProcessor()
dat.input_data()
dat.process_data()

MAX_LENGTH = dat.seqlenx
tr_perc = .8
dat.split_tst_tr(tr_perc)
epochs = 1000

xxx=[]; yyy=[];
start = 0
# plt.figure()
# for x,y in dat.tr_pairs:
#     plt.plot(range(start,len(x)+start),x)
#     plt.plot(range(start+len(x),len(x)+start+len(y)),y)
#     start += len(x)+len(y)+1
#
# plt.show()

encoder1 = EncoderRNN(len(dat.num_words_in), hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, len(dat.num_words_out), dropout_p=0.1, max_length = dat.seqlenx).to(device)

trainIters(encoder1, attn_decoder1, epochs, dat.x_tr, dat.y_tr, max_length = dat.seqlenx)

# n = dat.tstx.shape[0]
for x_tst, y_tst in zip(dat.x_tst,dat.y_tstx):
    predicted, attns = evaluate(encoder1, attn_decoder1, x_tst, dat.EOS, dat.seqlenx)
    if len(predicted)>len(y_tst):
        y_tst.extend([0]*(len(predicted)-len(y_tst)))
    else:
        predicted.extend([0]*(len(y_test)-len(predicted)))

    mse_l = np.sqrt(np.sum(np.square(np.array(predicted)- np.array(y_tst))))
    print('Testing MSE:' + str(mse_l))
