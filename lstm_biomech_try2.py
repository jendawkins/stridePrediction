# TO DO: debug for multiple features; initialize hidden state


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
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

# various helper functions
def one_hot(data, alphabet):
    return (np.arange(len(alphabet)+1) == data[...,None]).astype(int)

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


# data loader/processor function
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
            # low pass filtering
            x_0[:,i] = x_int[:,i];
            if filter:
                x_int[:,i] = scipy.signal.filtfilt(bl, al, x_int[:,i])
            # high pass filtering
                x_int[:,i] = scipy.signal.filtfilt(bh, ah, x_int[:,i])

        avg_int = np.mean(fin_time[2:-1]-fin_time[1:-2])
        sig = np.sqrt(np.sum(np.square(np.divide(x_int[:,1:4], x_int[:,1:4].max(axis=0))),axis=1))
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
        strides = np.array(strides)
        locs = np.array(locs).astype(int)
        locs2 = np.array(locs2)
        strides = strides[locs>1000]
        locs = locs[locs>1000]

        # fig, axes = plt.subplots(2, 1)
        # axes[0].plot(fin_time, x_0[:,3])
        # axes[1].plot(fin_time, x_int[:,3])
        # axes[1].scatter(fin_time[locs], strides, c="g")
        # axes[0].scatter(fin_time[locs], strides, c="g")

        # import pdb; pdb.set_trace()
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
    return data_fin, strides_bool


class DataProcessor():
    # data processor; allows access of data variables later and calls readLangs()
    # splits training and testing data too
    def __init__(self,training_sigs=3, bin_step=100, SOS=-1, EOS = 0, one_hot = False, filter = False):
        self.training_sigs = training_sigs
        self.bin_step = bin_step
        self.SOS = SOS
        self.EOS = EOS
        self.one_hot = one_hot
        self.filter = filter

    def input_data(self, filename):
        self.data_all = np.genfromtxt(filename,delimiter=',')
        self.timepts = self.data_all[:,0]
        self.heelStrike = [np.where(self.data_all[:,7]==1)[0], np.where(self.data_all[:,-2]==1)[0]]
        self.startPF = [np.where(self.data_all[:,7]==2)[0], np.where(self.data_all[:,-2]==2)[0]]
        self.endPF = [np.where(self.data_all[:,7]==3)[0], np.where(self.data_all[:,-2]==3)[0]]
        self.labels = self.data_all[:,-1]
        self.data = self.data_all[:,[1,2,3,4,5,6,8,9,10,11,12,13]]
        # plt.show()

    def createDict(self, numbers):
        return {j:i for i,j in enumerate(np.sort(numbers))}

    def number2idx(self, numbers, my_dict):
        return [my_dict[n] for n in numbers]

    def process_data(self, two_feet=False):
        self.data = (self.data - np.min(self.data,axis=0)/np.max(self.data,axis=0))
        self.data = np.floor((self.data - np.min(self.data,0))/self.bin_step) + 1
        dat.alphabet = np.unique(self.data)
        self.alphabet_dict = self.createDict(dat.alphabet)
        self.alphabet_data = np.array([self.number2idx(self.data[:,i], self.alphabet_dict) for i in range(self.data.shape[1])]).T
        self.inv_alphabet_dict = {v: k for k, v in self.alphabet_dict.items()}
        # import pdb; pdb.set_trace()
        self.strides = [self.alphabet_data[self.endPF[0][i]:self.endPF[0][i+1],:] for i in range(len(self.endPF[0])-1)]
        self.stridesF2 = [self.alphabet_data[self.endPF[1][i]:self.endPF[1][i+1],:] for i in range(len(self.endPF[1])-1)]

        start_startPF = [np.where(self.startPF[0]>self.endPF[0][0])[0], np.where(self.startPF[1]>self.endPF[1][0])[0]]
        # import pdb; pdb.set_trace()
        self.localStartPF = [self.startPF[0][start_startPF[0]] - self.endPF[0][0:len(start_startPF[0])]]
        self.localStartPF2 = [self.startPF[1][start_startPF[1]] - self.endPF[1][0:len(start_startPF[1])]]

        m1 = max([len(str) for str in self.strides])
        m2 = max([len(str) for str in self.stridesF2])
        self.maxlength = max(m1, m2)
        if ~two_feet:
            self.strides.extend(self.stridesF2)
            self.localStartPF = self.localStartPF[0].tolist()
            self.localStartPF2 = self.localStartPF2[0].tolist()
            self.localStartPF.extend(self.localStartPF2)

    def split_tst_tr(self,tr_perc,two_feet=False):
        num_pairs = len(self.localStartPF)
        # import pdb; pdb.set_trace()
        id1 = random.sample(range(num_pairs),math.floor(tr_perc*num_pairs))
        id2 = list(set(range(num_pairs))-set(id1))
        if ~two_feet:
            # self.trainStrides1 = self.stridesF1[id1]
            # self.trainStrides2 = self.stridesF2[id1]
            # self.testStrides1 = self.stridesF1[id2]
            # self.testStrides2 = self.stridesF2[id2]
            #
            # self.trainStartPF1 = self.localStartPF1[id1]
            # self.trainStartPF2 = self.localStartPF2[id1]
            # self.testStartPF1 = self.localStartPF1[id2]
            # self.testStartPF2 = self.localStartPF2[id2]
        # else:
            self.trainStrides = np.array(self.strides)[id1].tolist()
            self.testStrides = np.array(self.strides)[id2].tolist()
            self.trainStartPF = np.array(self.localStartPF)[id1].tolist()
            self.testStartPF = np.array(self.localStartPF)[id2].tolist()


class EncoderRNN(nn.Module):
    # encoder class; from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    def __init__(self, alphabet_size, num_features, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.embedding = nn.Embedding(alphabet_size, hidden_size)
        self.input_size = alphabet_size
        # if do_one_hot:
        self.gru = nn.LSTM(hidden_size*self.num_features, self.input_size*self.num_features)
    # else:

    def forward(self, input, hidden):
        input = input.type(torch.long)
        # input = input.unsqueeze(0)
        # if self.do_one_hot:
        embedded = torch.empty(input.shape[0], self.hidden_size)
        for feature in range(self.num_features):
            embedded = torch.cat((embedded, self.embedding(input[:,feature])),1)
            # try:
            #     embedded[feature,:,:] = self.embedding(input[:,feature])
            # except:
            #     import pdb; pdb.set_trace()
        # embedded = embedded.view(-1,input.shape[0])
        # embedded = torch.transpose(embedded,0,1)
        embedded = embedded[:,self.hidden_size:]
        embedded = embedded.unsqueeze(1)
        output, hidden = self.gru(embedded)
        output = F.softmax(output.squeeze(1),dim=1)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# training function
def train(input_tensor, startPFpt, encoder, encoder_optimizer, criterion, max_length):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    # populate hidden states with known values
    encoder_output, encoder_hidden = encoder(input_tensor[:startPFpt-1,:], encoder_hidden)
    try:
        input_length = input_tensor[startPFpt:,:].size(0)
    except:
        import pdb; pdb.set_trace()
    input_pt = input_tensor[startPFpt,:].unsqueeze(0)
    num_features = input_pt.shape[1]
    try:
        for ei in range(input_length-1):
            encoder_output, encoder_hidden = encoder(input_pt, encoder_hidden)
            encoder_output = encoder_output.view(1,12,-1)
            val, input_pt = torch.max(encoder_output,2)
            for f in range(num_features):
                loss += criterion(encoder_output[:,f,:],input_tensor[ei+1,f].unsqueeze(0))
    except:
        import pdb; pdb.set_trace()
    loss.backward()

    encoder_optimizer.step()

    return loss.item() / input_length

# train for each data point; calls train
def trainIters(encoder, epochs, train_data, trainPF, max_length,
                print_every=1.0, plot_every=100.0, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    for iter in range(1, epochs + 1):
        if iter == 1:
            start = 0
        for i in range(len(train_data)):
            input_tensor = torch.tensor(train_data[i])
            loss = train(input_tensor, trainPF[i],encoder,encoder_optimizer,criterion, max_length)
            print_loss_total += loss
            plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / epochs),
                                         iter, iter / epochs * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    # plt.show()
    showPlot(plot_losses)

# evaluate; same as train, but evaluates instead of training
def evaluate(encoder, x_in, startPF, max_length):
    with torch.no_grad():
        input_start = x_in[:startPF,:]

        real_tars = x_in[startPF:,:]
        feature_size = input_start.shape(1)
        input_tensor = torch.tensor(input_start[:-1,:])
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        encoder_out, encoder_hidden = encoder(input_tensor, encoder_hidden)
        input_point = torch.tensor(input_start[-1,:]);
        for i in range(len(real_tars)):
            encoder_out[i,:], encoder_hidden = encoder(input_point, encoder_hidden)
            input_point = encoder_out[i,:]

        encoder_words = np.argmax(encoder_out)
        for i in range(feature_size):
            decoded_words[:,i] = np.array([dat.inv_alphabet_dict[encoder_outputs.item()[i,i]] for i in range(x_in.shape(0))])

        return decoded_words, real_tars

hidden_size = 256
dat = DataProcessor()
dat.input_data('processed_data.csv')
dat.process_data()

MAX_LENGTH = dat.maxlength
tr_perc = .8
dat.split_tst_tr(tr_perc)
epochs = 1000
#
# xxx=[]; yyy=[];
# start = 0
# plt.figure()
# for x,y in dat.tr_pairs:
#
#     plt.plot(range(start,len(x)+start),dat.n2idx(x, dat.in_invmap), c='g')
#     plt.plot(range(start+len(x),len(x)+start+len(y)),dat.n2idx(y,dat.out_invmap), c = 'b')
#     start += len(x)+len(y)+1
#
# plt.show()
num_features = dat.data.shape[1]
encoder1 = EncoderRNN(len(dat.alphabet), num_features, hidden_size).to(device)

trainIters(encoder1, epochs, dat.trainStrides, dat.localStartPF, MAX_LENGTH)

# n = dat.tstx.shape[0]
for i in range(dat.trainStrides.shape[0]):
    x_tst = dat.trainStrides[i,:]
    pf_start = dat.testStartPF[i]
    predicted, attns = evaluate(encoder1, x_tst, pf_start, MAX_LENGTH)

    mse_l = np.sqrt(np.sum(np.square(np.array(predicted)- np.array(y_tst))))
    print('Testing MSE:' + str(mse_l))
