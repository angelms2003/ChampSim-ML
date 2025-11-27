# The MPMLP: A Case for Multi-Page Multi-Layer Perceptron Prefetcher

import math
from abc import ABC, abstractmethod
from collections import defaultdict
import os

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from time import time


class CacheSimulator(object):
    def __init__(self, sets, ways, block_size) -> None:
        super().__init__()
        self.ways = ways
        self.sets = sets
        self.set_shift = int(math.log2(sets))
        self.block_size = block_size
        self.block_shift = int(math.log2(block_size))
        self.storage = defaultdict(list)
        self.label_storage = defaultdict(list)

    def parse_address(self, address):
        block_addr = address >> self.block_shift
        cache_set = block_addr % self.sets
        cache_tag = block_addr >> self.set_shift
        return cache_set, cache_tag

    def load(self, address, label=None, overwrite=False):
        cache_set, cache_tag = self.parse_address(address)
        hit, l = self.check(address)
        if not hit:
            self.storage[cache_set].append(cache_tag)
            self.label_storage[cache_set].append(label)
            if len(self.storage[cache_set]) > self.ways:
                evicted_tag = self.storage[cache_set].pop(0)
                evicted_label = self.label_storage[cache_set].pop(0)
        else:
            current_index = self.storage[cache_set].index(cache_tag)
            _t, _l = self.storage[cache_set].pop(current_index), self.label_storage[cache_set].pop(current_index)
            self.storage[cache_set].append(_t)
            self.label_storage[cache_set].append(_l)
        if overwrite:
            self.label_storage[cache_set][self.storage[cache_set].index(tag)] = label
        return hit, l

    def check(self, address):
        cache_set, cache_tag = self.parse_address(address)
        if cache_tag in self.storage[cache_set]:
            return True, self.label_storage[cache_set][self.storage[cache_set].index(cache_tag)]
        else:
            return False, None


class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            # nn.Linear(128, 57 * 8),
            nn.Linear(64, 57 * 8),
            nn.ReLU(inplace=True),
            nn.Linear(57 * 8, 50 * 8),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Dropout(),
            # nn.Linear(8 * 50, 128),
            nn.Linear(8 * 50, 64),
        )
        self.softmax = nn.Softmax()

    def forward(self, input):
        return self.softmax(self.decoder(self.feature(input).view(-1, 50 * 8)))


class MLPrefetchModel(object):
    '''
    Abstract base class for your models. For HW-based approaches such as the
    NextLineModel below, you can directly add your prediction code. For ML
    models, you may want to use it as a wrapper, but alternative approaches
    are fine so long as the behavior described below is respected.
    '''

    @abstractmethod
    def load(self, path):
        '''
        Loads your model from the filepath path
        '''
        pass

    @abstractmethod
    def save(self, path):
        '''
        Saves your model to the filepath path
        '''
        pass

    @abstractmethod
    def train(self, data):
        '''
        Train your model here. No return value. The data parameter is in the
        same format as the load traces. Namely,
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        '''
        pass
    
    @abstractmethod
    def train_and_test(self, train_data, test_data, graph_name = None):
        '''
        Train and test your model here using the train data and the test data

        The data is the same format given in the load traces. Namely:
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        '''
        print('Training and testing BestOffset')

    @abstractmethod
    def generate(self, data):
        '''
        Generate your prefetches here. Remember to limit yourself to 2 prefetches
        for each instruction ID and to not look into the future :).

        The return format for this will be a list of tuples containing the
        unique instruction ID and the prefetch. For example,
        [
            (A, A1),
            (A, A2),
            (C, C1),
            ...
        ]

        where A, B, and C are the unique instruction IDs and A1, A2 and C1 are
        the prefetch addresses.
        '''
        pass


class BestOffset(MLPrefetchModel):
    offsets = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9, -9, 10, -10, 11, -11, 12, -12, 13, -13, 14,
               -14, 15, -15, 16, -16, 18, -18, 20, -20, 24, -24, 30, -30, 32, -32, 36, -36, 40, -40]
    scores = [0 for _ in range(len(offsets))]
    round = 0
    best_index = 0
    second_best_index = 0
    best_index_score = 0
    temp_best_index = 0
    score_scale = eval(os.environ.get('BO_SCORE_SCALE', '1'))
    bad_score = int(10 * score_scale)
    low_score = int(20 * score_scale)
    max_score = int(31 * score_scale)
    max_round = int(100 * score_scale)
    # llc = CacheSimulator(16, 2048, 64)
    llc = CacheSimulator(64, 3072, 128)
    rrl = {}
    rrr = {}
    dq = []
    acc = []
    acc_alt = []
    active_offsets = set()
    p = 0
    memory_latency = 200
    rr_latency = 60
    fuzzy = eval(os.environ.get('FUZZY_BO', 'False'))

    def load(self, path):
        # Load your pytorch / tensorflow model from the given filepath
        print('Loading ' + path + ' for BestOffset')

    def save(self, path):
        # Save your model to a file
        print('Saving ' + path + ' for BestOffset')

    def train(self, data):
        '''
        Train your model here using the data

        The data is the same format given in the load traces. Namely:
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        '''
        print('Training BestOffset')
    
    def train_and_test(self, train_data, test_data, graph_name = None):
        '''
        Train and test your model here using the train data and the test data

        The data is the same format given in the load traces. Namely:
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        '''
        print('Training and testing BestOffset')

    def rr_hash(self, address):
        return ((address >> 6) + address) % 64

    def rr_add(self, cycles, address):
        self.dq.append((cycles, address))

    def rr_add_immediate(self, address, side='l'):
        if side == 'l':
            self.rrl[self.rr_hash(address)] = address
        elif side == 'r':
            self.rrr[self.rr_hash(address)] = address
        else:
            assert False

    def rr_pop(self, current_cycles):
        while self.dq:
            cycles, address = self.dq[0]
            if cycles < current_cycles - self.rr_latency:
                self.rr_add_immediate(address, side='r')
                self.dq.pop(0)
            else:
                break

    def rr_hit(self, address):
        return self.rrr.get(self.rr_hash(address)) == address or self.rrl.get(self.rr_hash(address)) == address

    def reset_bo(self):
        self.temp_best_index = -1
        self.scores = [0 for _ in range(len(self.offsets))]
        self.p = 0
        self.round = 0
        # self.acc.clear()
        # self.acc_alt.clear()

    def train_bo(self, address):
        testoffset = self.offsets[self.p]
        testlineaddr = address - testoffset

        if address >> 6 == testlineaddr >> 6 and self.rr_hit(testlineaddr):
            self.scores[self.p] += 1
            if self.scores[self.p] >= self.scores[self.temp_best_index]:
                self.temp_best_index = self.p

        if self.p == len(self.scores) - 1:
            self.round += 1
            if self.scores[self.temp_best_index] == self.max_score or self.round == self.max_round:
                self.best_index = self.temp_best_index if self.temp_best_index != -1 else 1
                self.second_best_index = sorted([(s, i) for i, s in enumerate(self.scores)])[-2][1]
                self.best_index_score = self.scores[self.best_index]
                if self.best_index_score <= self.bad_score:
                    self.best_index = -1
                self.active_offsets.add(self.best_index)
                self.reset_bo()
                return
        self.p += 1
        self.p %= len(self.scores)

    def generate(self, data):
        '''
        Generate the prefetches for the prefetch file for ChampSim here

        As a reminder, no looking ahead in the data and no more than 2
        prefetches per unique instruction ID

        The return format for this function is a list of (instr_id, pf_addr)
        tuples as shown below
        '''
        print('Generating for BestOffset')
        prefetches = []
        prefetch_requests = []
        percent = len(data) // 100
        for i, (instr_id, cycle_count, load_addr, load_ip, llc_hit) in enumerate(data):
            # Prefetch the next two blocks
            hit, prefetched = self.llc.load(load_addr, False)
            while prefetch_requests and prefetch_requests[0][0] + self.memory_latency < cycle_count:
                fill_addr = prefetch_requests[0][1]
                h, p = self.llc.load(fill_addr, True)
                if not h:
                    if self.best_index == -1:
                        fill_line_addr = fill_addr >> 6
                        if self.best_index != -1:
                            offset = self.offsets[self.best_index]
                        else:
                            offset = 0
                        self.rr_add_immediate(fill_line_addr - offset)
                prefetch_requests.pop(0)
            self.rr_pop(cycle_count)
            if not hit or prefetched:
                line_addr = (load_addr >> 6)
                self.train_bo(line_addr)
                self.rr_add(cycle_count, line_addr)
                if self.best_index != -1 and self.best_index_score > self.low_score:
                    addr_1 = (line_addr + 1 * self.offsets[self.best_index]) << 6
                    addr_2 = (line_addr + 2 * self.offsets[self.best_index]) << 6
                    addr_2_alt = (line_addr + 1 * self.offsets[self.second_best_index]) << 6
                    acc = len({addr_2 >> 6, addr_1 >> 6} & set(d[2] >> 6 for d in data[i + 1: i + 25]))
                    self.acc.append(acc)
                    acc_alt = len({addr_2_alt >> 6, addr_1 >> 6} & set(d[2] >> 6 for d in data[i + 1: i + 25]))
                    self.acc_alt.append(acc_alt)
                    # if acc_alt > acc:
                    #     addr_2 = addr_2_alt
                    prefetches.append((instr_id, addr_1))
                    prefetches.append((instr_id, addr_2))
                    prefetch_requests.append((cycle_count, addr_1))
                    prefetch_requests.append((cycle_count, addr_2))
            else:
                pass
            if i % percent == 0:
                print(i // percent, self.active_offsets, self.best_index_score,
                      sum(self.acc) / 2 / (len(self.acc) + 1),
                      sum(self.acc_alt) / 2 / (len(self.acc_alt) + 1))
                self.acc.clear()
                self.acc_alt.clear()
                self.active_offsets.clear()
        return prefetches


class MLPBasedSubPrefetcher(MLPrefetchModel):
    """
    This class effectively functions as a wrapper around the above custom
    pytorch nn.Module. You can approach this in another way so long as the the
    load/save/train/generate functions behave as described above.
    """

    degree = 2
    k = int(os.environ.get('CNN_K', '2'))
    model_class = eval(os.environ.get('CNN_MODEL_CLASS', 'MLP'))
    history = int(os.environ.get('CNN_HISTORY', '4'))
    lookahead = int(os.environ.get('LOOKAHEAD', '5'))
    bucket = os.environ.get('BUCKET', 'ip')
    epochs = int(os.environ.get('EPOCHS', '30'))
    lr = float(os.environ.get('CNN_LR', '0.002'))
    window = history + lookahead + k
    filter_window = lookahead * degree
    next_page_table = defaultdict(dict)
    batch_size = 256

    def __init__(self):
        self.model = self.model_class()

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def save(self, path):
        ## torch.save(self.model.state_dict(), path)
        self.model.eval()

        example_input = torch.randn(1, 64)
        traced = torch.jit.trace(self.model, example_input)
        traced.save(path)

    # Batches the mem access data for the ML model
    def batch(self, data, batch_size=None):
        # The default batch size is 256
        if batch_size is None:
            batch_size = self.batch_size
        

        bucket_data = defaultdict(list)
        bucket_instruction_ids = defaultdict(list)
        batch_instr_id, batch_page, batch_next_page, batch_x, batch_y = [], [], [], [], []

        # Each line (each mem access of the original applicacion) is read
        for line in data:
            # Each mem access contains five pieces of data
            instr_id, cycles, load_address, ip, hit = line

            # We assume that pages are 4 KiB
            page = load_address >> 12

            # This tuple stores the instruction pointer and the page
            ippage = (ip, page)

            # The bucket key is ip by default
            bucket_key = eval(self.bucket)

            # We access bucket_data[ip], which is a list
            bucket_buffer = bucket_data[bucket_key]

            # The load instruction is appended to the list
            bucket_buffer.append(load_address)

            # Just like we appended the address to bucket_data[ip], we
            # append the instruction ID to bucket_instruction_ids[ip]
            bucket_instruction_ids[bucket_key].append(instr_id)

            # The bucket_buffer size is limited to the size of the window that
            # we have. This window is big enough to hold three things:
            #   - Four history mem accesses (the mem accesses that the prefetcher
            #     will use as input to predict)
            #   - Five lookahead mem accesses (the prefetcher won't predict the
            #     access right next to itself, but rather the access that is five
            #     steps further)
            #   - Two goal mem accesses (these must be the output of the prefetcher)
            if len(bucket_buffer) >= self.window:

                # The current page is the page of the latest access
                current_page = bucket_buffer[self.history - 1] >> 12
                
                # We also record the page of the previous mem access
                last_page = bucket_buffer[self.history - 2] >> 12

                # If these two pages are different, that means that we jumped
                # from one page to another: we need to record it in the PTT. Maybe
                # this should be done out of this if block?
                if last_page != current_page:
                    self.next_page_table[ip][last_page] = current_page
                
                # The page for this input will be the lastest page
                batch_page.append(bucket_buffer[self.history - 1] >> 12)

                # The predicted next page will be the page according to the PTT given
                # our current page and IP. If there is no entry, the next page is the same page
                batch_next_page.append(self.next_page_table[ip].get(current_page, current_page))
                
                # TODO send transition information for labels to represent

                # We can create an input and add it to the batch. The input uses the addresses
                # of the 4 first accesses in the window and the current page. It is necessary to
                # represent this information so that it makes sense to the MLP model.
                batch_x.append(self.represent(bucket_buffer[:self.history], current_page))

                # The output is created the same way, but using the two last accesses, which are the
                # accesses to be prefetched
                batch_y.append(self.represent(bucket_buffer[-self.k:], current_page, box=False))

                # The instruction ID of this input will be the ID of the last mem access
                batch_instr_id.append(bucket_instruction_ids[bucket_key][self.history - 1])

                # The first mem access is discarded as the window will progress one access further
                bucket_buffer.pop(0)
                bucket_instruction_ids[bucket_key].pop(0)
            
            # If we already have enough info for a batch, we can use it with the MLP model
            if len(batch_x) == batch_size:
                if torch.cuda.is_available():
                    yield batch_instr_id, batch_page, batch_next_page, torch.Tensor(batch_x).cuda(), torch.Tensor(batch_y).cuda()
                else:
                    yield batch_instr_id, batch_page, batch_next_page, torch.Tensor(batch_x), torch.Tensor(batch_y)
                
                # The batch info is cleared and we begin filling it again with new info
                batch_instr_id, batch_page, batch_next_page, batch_x, batch_y = [], [], [], [], []

    def accuracy(self, output, label):
        return torch.sum(
            torch.logical_and(
                torch.scatter(
                    torch.zeros(output.shape, device=output.device), 1, torch.topk(output, self.k).indices, 1
                ),
                label
            )
        ) / label.shape[0] / self.k

    def train_and_test(self, train_data, test_data, graph_name = None):
        print('LOOKAHEAD =', self.lookahead)
        print('BUCKET =', self.bucket)

        # This is the optimization function for finding the correct values for the
        # weights of the MLP
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # This is the loss function for defining how far we are from the correct output
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.BCELoss()
        
        # We check if there is a GPU available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            criterion = criterion.cuda()

        # idk what this is (the train function for the MLP is empty, inherited from
        # MLPrefetchModel)
        self.model.train()

        # The average accuracy for train and test in each epoch
        avg_train_accs = []
        avg_test_accs = []

        # The loss for train and test in each epoch
        total_train_loss = []
        total_test_loss = []

        # For each epoch, the model is trained with all training data and
        # tested with all test data
        for epoch in range(self.epochs):
            # TRAIN PART

            # Accuracies
            train_accs = []

            # Calculated losses
            train_losses = []

            train_percent = len(train_data) // self.batch_size // 100

            # The data is batched. For each batch...
            for i, (instr_id, page, next_page, x_train, y_train) in enumerate(self.batch(train_data)):

                # clearing the Gradients of the model parameters
                optimizer.zero_grad()

                # prediction for training set
                output_train = self.model(x_train)

                # computing the training loss and accuracy
                loss_train = criterion(output_train, y_train)
                acc = self.accuracy(output_train, y_train)
                # print('Acc {}: {}'.format(epoch, acc))

                # computing the updated weights of all the model parameters
                loss_train.backward()
                optimizer.step()
                tr_loss = loss_train.item()
                train_accs.append(float(acc))
                train_losses.append(float(tr_loss))
                if train_percent != 0 and i % train_percent == 0:
                    print('.', end='')
            print()
            print('Training accuracy {}: {}'.format(epoch, sum(train_accs) / len(train_accs)))
            print('Training epoch : ', epoch + 1, '\t', 'training loss :', sum(train_losses))

            avg_train_accs.append(sum(train_accs) / len(train_accs))
            total_train_loss.append(sum(train_losses))
            
            ########################################################
            # TEST PART

            # Accuracies
            test_accs = []

            # Calculated losses
            test_losses = []

            test_percent = len(test_data) // self.batch_size // 100

            # The data is batched. For each batch...
            for i, (instr_id, page, next_page, x_test, y_test) in enumerate(self.batch(test_data)):
                with torch.no_grad():
                    # prediction for testing set
                    output_test = self.model(x_test)

                    # computing the testing loss and accuracy
                    loss_test = criterion(output_test, y_test)
                    acc = self.accuracy(output_test, y_test)

                    tr_loss = loss_test.item()
                    test_accs.append(float(acc))
                    test_losses.append(float(tr_loss))
                    if test_percent != 0 and i % test_percent == 0:
                        print('.', end='')
            print()
            print('Test accuracy {}: {}'.format(epoch, sum(test_accs) / len(test_accs)))
            print('Test Epoch : ', epoch + 1, '\t', 'test loss :', sum(test_losses))

            avg_test_accs.append(sum(test_accs) / len(test_accs))
            total_test_loss.append(sum(test_losses))

            # Early stopping: if the last two epochs showed a decrease in test accuracy, it means
            # that the model is suffering from overfitting
            if len(avg_test_accs) >= 3 and avg_test_accs[-1] < avg_test_accs[-2] and avg_test_accs[-2] < avg_test_accs[-3]:
                print('EARLY STOPPED')
                break
    
        # Once the model was trained and tested, the accuracies and losses are plotted
        if graph_name is not None:
            epochs = range(1, len(avg_train_accs) + 1)

            plt.figure(figsize=(10, 5))

            # Accuracy plot
            plt.subplot(1, 2, 1)
            plt.plot(epochs, avg_train_accs, label='Train Accuracy', marker='o')
            plt.plot(epochs, avg_test_accs, label='Test Accuracy', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Train vs Test Accuracy')
            plt.legend()
            plt.grid(True)

            # =======================
            # 2. Loss plot
            # =======================
            plt.subplot(1, 2, 2)
            plt.plot(epochs, total_train_loss, label='Train Loss', marker='o')
            plt.plot(epochs, total_test_loss, label='Test Loss', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Train vs Test Loss')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            
            plt.savefig(f"{graph_name}.png", dpi=300)

    def generate(self, data):

        # The model is instanced
        self.model.eval()

        prefetches = []
        accs = []

        # This stores a sequential index and the instruction ID for each access
        order = {i: line[0] for i, line in enumerate(data)}

        # This stores the same thing as order but the key is the value and vice versa
        reverse_order = {v: k for k, v in order.items()}

        # We batch the data and iterate through each batch
        for i, (instr_ids, pages, next_pages, x, y) in enumerate(self.batch(data)):
            # breakpoint()

            # The current page of each input (page of the last access of the history)
            pages = torch.LongTensor(pages).to(x.device)

            # The next pages of each input (predicted next page according to the PTT)
            next_pages = torch.LongTensor(next_pages).to(x.device)

            # The instruction IDs of the last access of the history of each input
            instr_ids = torch.LongTensor(instr_ids).to(x.device)

            # The variable x contains the input batch
            y_preds = self.model(x)

            # We calculate the accuracy with the two next accesses that should have been predicted
            accs.append(float(self.accuracy(y_preds, y)))

            # We obtain the top 2 offsets of each output. This is an array of size (N, deg), where
            # N is the amount of elements in the batch and deg is the prefetching degree (how many
            # prefetches per access)
            topk = torch.topk(y_preds, self.degree).indices

            # We don't want a 2D array of size (N, deg), but a 1D array of size (N * deg)
            shape = (topk.shape[0] * self.degree,)
            topk = topk.reshape(shape)

            # Since we have a topk array of shape (N * deg) but the pages, next_pages and instr_ids are
            # arrays of shape (N), we need to repeat each element deg times. For example, if pages is an array
            # with this content: [34, 90, 10], it must be [34, 34, 90, 90, 10, 10] because deg is 2
            pages = torch.repeat_interleave(pages, self.degree)
            next_pages = torch.repeat_interleave(next_pages, self.degree)
            instr_ids = torch.repeat_interleave(instr_ids, self.degree)

            # addresses = (topk < 64) * (pages << 12) + (topk >= 64) * ((next_pages << 12) - (64 << 6)) + (topk << 6)

            # We calculate the addresses once the pages, next_pages and isntr_ids arrays are of the right size
            addresses = (topk < 32) * (pages << 12) + (topk >= 32) * ((next_pages << 12) - (32 << 7)) + (topk << 7)
            #addresses = (pages << 12) + (topk << 6)
            prefetches.extend(zip(map(int, instr_ids), map(int, addresses)))
            if i % 100 == 0:
                print('Chunk', i, 'Accuracy', sum(accs) / len(accs))
        prefetches = sorted([(reverse_order[iid], iid, addr) for iid, addr in prefetches])
        prefetches = [(iid, addr) for _, iid, addr in prefetches]
        return prefetches

    def represent(self, addresses, first_page, box=True):
        # This function takes a list of memory accesses and returns
        # a valid input or output for the MLP model

        #blocks = [(address >> 6) % 64 for address in addresses]

        # This is a list that contains block offsets for every address
        blocks = [(address >> 7) % 32 for address in addresses]

        # This is a list that contains a page address for every address
        pages = [(address >> 12) for address in addresses]

        # raw = [0 for _ in range(128)]

        # This represents the input layer (or output layer)
        raw = [0 for _ in range(64)]

        # For each block...
        for i, block in enumerate(blocks):
            # If the block is located in the current page, then add it to 
            # the first positions of the layer
            if first_page == pages[i]:
                raw[block] = 1
            else:
                raw[32 + block] = 1
        
        # Box is true when creating the input, and false when creating
        # the output
        if box:
            return [raw]
        else:
            return raw


class Hybrid(MLPrefetchModel):

    prefetcher_classes = (BestOffset,
                          MLPBasedSubPrefetcher, )

    def __init__(self) -> None:
        super().__init__()
        self.prefetchers = [prefetcher_class() for prefetcher_class in self.prefetcher_classes]

    def load(self, path):
        for prefetcher in self.prefetchers:
            prefetcher.load(path)
        pass

    def save(self, path):
        for prefetcher in self.prefetchers:
            prefetcher.save(path)

    def train(self, data):
        for prefetcher in self.prefetchers:
            prefetcher.train(data)
    
    def train_and_test(self, train_data, test_data, graph_name = None):
        for prefetcher in self.prefetchers:
            prefetcher.train_and_test(train_data,test_data)

    def generate(self, data):
        # Data is a list. Each entry is another list that contains info for an access.
        prefetch_sets = defaultdict(lambda: defaultdict(list))
        for p, prefetcher in enumerate(self.prefetchers):
            prefetches = prefetcher.generate(data)
            for iid, addr in prefetches:
                prefetch_sets[p][iid].append((iid, addr))
        total_prefetches = []

        for i, (instr_id, cycle_count, load_addr, load_ip, llc_hit) in enumerate(data):

            instr_prefetches = []
            for d in range(2):
                for p in range(len(self.prefetchers)):
                    if prefetch_sets[p][instr_id]:
                        instr_prefetches.append(prefetch_sets[p][instr_id].pop(0))
            instr_prefetches = instr_prefetches[:2]
            total_prefetches.extend(instr_prefetches)
        return total_prefetches


ml_model_name = os.environ.get('ML_MODEL_NAME', 'Hybrid')
Model = eval(ml_model_name)