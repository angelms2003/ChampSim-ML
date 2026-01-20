# The MPMLP: A Case for Multi-Page Multi-Layer Perceptron Prefetcher

import math
from abc import ABC, abstractmethod
from collections import defaultdict
import os

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from time import time

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
    def train_and_test(self, train_data, test_data, model_name = None, graph_name = None):
        '''
        Train and test your model here using the train data and the test data

        The data is the same format given in the load traces. Namely:
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        '''

    @abstractmethod
    def test(self, train_data, test_data, model_name = None, graph_name = None):
        '''
        Test your model here using the train data and the test data

        The data is the same format given in the load traces. Namely:
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        '''

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

class MLPBasedSubPrefetcher(MLPrefetchModel):
    """
    This class effectively functions as a wrapper around the above custom
    pytorch nn.Module. You can approach this in another way so long as the the
    load/save/train/generate functions behave as described above.
    """

    degree = 2
    k = int(os.environ.get('CNN_K', '2'))
    
    # Instead of using hard labels and tell the prefetcher to guess
    # the k next accesses, this allows the prefetcher to guess any of
    # many next accesses, where the closest ones are more valuable
    tolerance = 20
    
    model_class = eval(os.environ.get('CNN_MODEL_CLASS', 'MLP'))
    history = int(os.environ.get('CNN_HISTORY', '4'))
    lookahead = int(os.environ.get('LOOKAHEAD', '5'))
    bucket = os.environ.get('BUCKET', 'ip')
    epochs = int(os.environ.get('EPOCHS', '30'))
    lr = float(os.environ.get('CNN_LR', '0.002'))

    # Now we want the model to guess any 2 of the accesses in
    # the tolerance window
    # window = history + lookahead + k
    window = history + lookahead + k + tolerance

    filter_window = lookahead * degree
    next_page_table = defaultdict(dict)
    batch_size = 256

    def __init__(self):
        self.model = self.model_class()

    def load(self, path):
        self.model = torch.jit.load(path)

    def save(self, path):
        ## torch.save(self.model.state_dict(), path)
        self.model.eval()

        example_input = torch.randn(1, 64).cuda()
        traced = torch.jit.trace(self.model, example_input)
        traced.save(path)

    # Batches the mem access data for the ML model
    def batch(self, data, batch_size=None):
        # The default batch size is 256
        if batch_size is None:
            batch_size = self.batch_size

        bucket_data = defaultdict(list)
        bucket_instruction_ids = defaultdict(list)
        batch_instr_id, batch_page, batch_next_page, batch_x, batch_y, whole_windows = [], [], [], [], [], []

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

                # The current page is the page of the latest access in the history
                current_page = (bucket_buffer[self.history - 1]) >> 12
                
                # We also record the page of the mem access that is right after the lookahead
                true_next_page = (bucket_buffer[self.history + self.lookahead]) >> 12

                # According to the PTT, this is the page that should follow the current page
                predicted_next_page = self.next_page_table[ip].get(current_page, current_page)

                # If we access a new page, that means that we jumped
                # from one page to another: we need to record it in the PTT
                # CAREFUL: if page B follows page A, and then page A follows page A,
                # the table is not updated with info about page A following page A (page
                # B still follows page A)
                if true_next_page != current_page:
                    self.next_page_table[ip][current_page] = true_next_page
                
                # The page for this input will be the page of the latest access in the history
                batch_page.append(current_page)

                # The predicted next page will be the page according to the PTT given
                # our current page and IP. If there is no entry, the next page is the same page
                batch_next_page.append(predicted_next_page)
                
                # TODO send transition information for labels to represent

                # We can create an input and add it to the batch. The input uses the addresses
                # of the 4 first accesses in the window and the current page. It is necessary to
                # represent this information so that it makes sense to the MLP model.
                batch_x.append(self.represent(bucket_buffer[:self.history], current_page))

                # The output is created the same way, but using the accesses in the tolerance window,
                # which are the accesses that should be guessed
                #batch_y.append(self.represent(bucket_buffer[-self.k:], current_page, box=False))
                batch_y.append(self.represent_with_tolerance(bucket_buffer[-(self.lookahead + self.k + self.tolerance):], current_page))

                # The memory addresses of every access in the window are stored for future use
                whole_windows.append(bucket_buffer)

                # The instruction ID of this input will be the ID of the last mem access
                batch_instr_id.append(bucket_instruction_ids[bucket_key][self.history - 1])

                # The first mem access is discarded as the window will progress one access further
                bucket_buffer.pop(0)
                bucket_instruction_ids[bucket_key].pop(0)
            
            # If we already have enough info for a batch, we can use it with the MLP model
            if len(batch_x) == batch_size:
                if torch.cuda.is_available():
                    yield batch_instr_id, batch_page, batch_next_page, torch.Tensor(batch_x).cuda(), torch.Tensor(batch_y).cuda(), whole_windows
                else:
                    yield batch_instr_id, batch_page, batch_next_page, torch.Tensor(batch_x), torch.Tensor(batch_y), whole_windows
                
                # The batch info is cleared and we begin filling it again with new info
                batch_instr_id, batch_page, batch_next_page, batch_x, batch_y, whole_windows = [], [], [], [], [], []

    # This accuracy function is too harsh. It forces the model to exactly predict the next k
    # blocks after the lookahead blocks, but in data prefetching you should be more open-minded.
    # A prefetched block can be useful even if it is accessed many accesses later
    # def accuracy(self, output, label):
    #     return torch.sum(
    #         torch.logical_and(
    #             torch.scatter(
    #                 torch.zeros(output.shape, device=output.device), 1, torch.topk(output, self.k).indices, 1
    #             ),
    #             label
    #         )
    #     ) / label.shape[0] / self.k

    # This functions takes an output vector directly taken from the MLP which should be
    # of shape (N, C), where N is the batch size and C is the size of the last layer of
    # the MLP. It alse uses a label tensor of the same shape.
    def accuracy(self, output, label, predicted_page, whole_window):
        # Get the top k predictions for each output in the batch. Shape (N, k)
        topk_idx = torch.topk(output, self.k).indices

        # This is the total accuracy obtained by all predictions in this batch
        total_score = 0.0

        # This is the total PTT accuracy obtained by all predictions in this batch
        ptt_score = 0.0

        batch_size = output.shape[0]

        # For each prediction in the batch...
        for i in range(batch_size):
            # Count how many of the predicted blocks were useful
            useful = 0
            for idx in topk_idx[i]:
                if label[i, idx] > 0:
                    useful += 1

            # This can add 0, 0.5 or 1 if k=2
            total_score += useful / self.k


            ptt_useful = 0
            # Count how many pages predicted by the PTT were useful
            for idx in topk_idx[i]:
                if idx < 32:
                    next_page = whole_window[i][self.history-1] >> 12
                else:
                    next_page = predicted_page[i]
                    idx-=32

                block_offset = idx
                
                # if i == 0:
                #     print("-"*30)
                #     print(f"Predicted access to page {next_page} and block offset {idx}.")
                #     print(f"Next {len(whole_window[i][self.history:])} accesses are:")
                #     print(whole_window[i][self.history:])

                good_access_found = False
                
                for access in whole_window[i][self.history:]:
                    access_page = access >> 12
                    access_block_offset = (access >> 7) % 32
                    
                    # if i == 0:
                    #     print(f"\t- Page {access_page}, block offset {access_block_offset}.")

                    if access_page == next_page and access_block_offset == block_offset:
                        good_access_found = True

                        # if i == 0:
                        #     print(f"\t\tMATCH!")
                
                if good_access_found:
                    ptt_useful+=1
                
            ptt_score += ptt_useful / self.k


        # This will be 1 if everything was correct and 0 if everything was a disaster. But
        # since nothing in this life is either black or white, this can also be any value
        # between 0 and 1 depending on how the different predictions in this batch performed
        return total_score / batch_size, ptt_score / batch_size

    def train_and_test(self, train_data, test_data, model_name = None, graph_name = None):
        print('LOOKAHEAD =', self.lookahead)
        print('BUCKET =', self.bucket)

        # Number of epochs without improvement before stopping
        patience = 3

        # Best value of loss during test
        best_test_loss = float("inf")

        # For how many epochs the test loss didn't improve
        epochs_without_improvement = 0

        # This is the optimization function for finding the correct values for the
        # weights of the MLP
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # This is the loss function for defining how far we are from the correct output
        # criterion = nn.CrossEntropyLoss()
        # criterion = nn.BCELoss()
        criterion = nn.BCEWithLogitsLoss()

        # We check if there is a GPU available
        if torch.cuda.is_available():
            print("Using CUDA")
            self.model = self.model.cuda()
            criterion = criterion.cuda()
        else:
            print("NOT using CUDA")

        # idk what this is (the train function for the MLP is empty, inherited from
        # MLPrefetchModel)
        self.model.train()

        # The average accuracy for train and test in each epoch
        avg_train_accs = []
        avg_train_ptt_accs = []
        avg_test_accs = []
        avg_test_ptt_accs = []

        # The loss for train and test in each epoch
        total_train_loss = []
        total_test_loss = []

        # For each epoch, the model is trained with all training data and
        # tested with all test data
        for epoch in range(self.epochs):
            # TRAIN PART

            # Accuracies
            train_accs = []
            train_ptt_accs = []

            # Calculated losses
            train_losses = []

            train_percent = len(train_data) // self.batch_size // 100

            # The data is batched. For each batch...
            for i, (instr_id, page, next_page, x_train, y_train, whole_windows) in enumerate(self.batch(train_data)):
                # clearing the Gradients of the model parameters
                optimizer.zero_grad()

                # prediction for training set
                output_train = self.model(x_train)

                # computing the training loss and accuracy
                loss_train = criterion(output_train, y_train)
                acc, ptt_acc = self.accuracy(output_train, y_train, next_page, whole_windows)
                # print('Acc {}: {}'.format(epoch, acc))

                # computing the updated weights of all the model parameters
                loss_train.backward()
                optimizer.step()
                tr_loss = loss_train.item()
                train_accs.append(float(acc))
                train_ptt_accs.append(float(ptt_acc))
                train_losses.append(float(tr_loss))
                if train_percent != 0 and i % train_percent == 0:
                    print('.', end='')
            print()
            print('Training accuracy for epoch {}: {}'.format(epoch+1, sum(train_accs) / len(train_accs)))
            print('Training PTT accuracy for epoch {}: {}'.format(epoch+1, sum(train_ptt_accs) / len(train_ptt_accs)))
            print('Training epoch : ', epoch + 1, '\t', 'training loss :', sum(train_losses))

            avg_train_accs.append(sum(train_accs) / len(train_accs))
            avg_train_ptt_accs.append(sum(train_ptt_accs) / len(train_ptt_accs))
            total_train_loss.append(sum(train_losses))
            
            ########################################################
            # TEST PART

            # Accuracies
            test_accs = []
            test_ptt_accs = []

            # Calculated losses
            test_losses = []

            test_percent = len(test_data) // self.batch_size // 100

            # The data is batched. For each batch...
            for i, (instr_id, page, next_page, x_test, y_test, whole_windows) in enumerate(self.batch(test_data)):
                with torch.no_grad():
                    # prediction for testing set
                    output_test = self.model(x_test)

                    # computing the testing loss and accuracy
                    loss_test = criterion(output_test, y_test)
                    acc, ptt_acc = self.accuracy(output_test, y_test, next_page, whole_windows)

                    tr_loss = loss_test.item()
                    test_accs.append(float(acc))
                    test_ptt_accs.append(float(ptt_acc))
                    test_losses.append(float(tr_loss))
                    if test_percent != 0 and i % test_percent == 0:
                        print('.', end='')
            print()
            print('Test accuracy for epoch {}: {}'.format(epoch+1, sum(test_accs) / len(test_accs)))
            print('Test PTT accuracy for epoch {}: {}'.format(epoch+1, sum(test_ptt_accs) / len(test_ptt_accs)))
            print('Test epoch : ', epoch + 1, '\t', 'test loss :', sum(test_losses))

            avg_test_accs.append(sum(test_accs) / len(test_accs))
            avg_test_ptt_accs.append(sum(test_ptt_accs) / len(test_ptt_accs))
            total_test_loss.append(sum(test_losses))

            # A snapshot of the model for this epoch is saved
            self.save(model_name+"-epoch"+str(epoch+1)+".pt")

            # Early stopping: if the test loss isn't improving, stop training
            current_loss = sum(test_losses)
            
            if current_loss < best_test_loss:
                print(f"Validation improved from {best_test_loss:.6f} to {current_loss:.6f}")
                best_test_loss = current_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(f"No improvement. Patience: {epochs_without_improvement}/{patience}")

            if epochs_without_improvement >= patience:
                print("Early stopping triggered")
                break
    
        # Once the model was trained and tested, the accuracies and losses are plotted
        if graph_name is not None:
            epochs = range(1, len(avg_train_accs) + 1)

            plt.figure(figsize=(22, 5))

            # 1: Accuracy plot (only offsets)
            plt.subplot(1, 4, 1)
            plt.plot(epochs, avg_train_accs, label='Train Accuracy', marker='o')
            plt.plot(epochs, avg_test_accs, label='Test Accuracy', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Train vs Test Accuracy (only offsets)')
            plt.legend()
            plt.grid(True)

            # 2: Accuracy plot (offsets and PTT)
            plt.subplot(1, 4, 2)
            plt.plot(epochs, avg_train_ptt_accs, label='Train Accuracy', marker='o')
            plt.plot(epochs, avg_test_ptt_accs, label='Test Accuracy', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Train vs Test Accuracy (offsets and PTT)')
            plt.legend()
            plt.grid(True)

            # 3: Loss plot (train)
            plt.subplot(1, 4, 3)
            plt.plot(epochs, total_train_loss, label='Train Loss', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Train Loss')
            plt.legend()
            plt.grid(True)

            # 4: Loss plot (test)
            plt.subplot(1, 4, 4)
            plt.plot(epochs, total_test_loss, label='Test Loss', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Test Loss')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            
            plt.savefig(f"{graph_name}.png", dpi=300)
    
    def test(self, train_data, test_data, model_name = None, graph_name = None):
        print('LOOKAHEAD =', self.lookahead)
        print('BUCKET =', self.bucket)

        # The average accuracy for train and test in each epoch
        avg_train_accs = []
        avg_test_accs = []

        # The loss for train and test in each epoch
        total_train_loss = []
        total_test_loss = []

        # This is the loss function for defining how far we are from the correct output
        # criterion = nn.CrossEntropyLoss()
        # criterion = nn.BCELoss()
        criterion = nn.BCEWithLogitsLoss()

        # For each epoch, the model is trained with all training data and
        # tested with all test data
        for epoch in range(self.epochs):
            
            # If there is no file saved for this epoch, exit
            if not os.path.isfile(model_name+"-epoch"+str(epoch)+".pt"):
                break

            self.load(model_name+"-epoch"+str(epoch)+".pt")

            # We check if there is a GPU available
            if torch.cuda.is_available():
                print("Using CUDA")
                self.model = self.model.cuda()
                criterion = criterion.cuda()
            else:
                print("NOT using CUDA")

            # We are no longer training: we will test in both the training data and test data
            self.model.eval()
            # TRAIN PART

            # Accuracies
            train_accs = []

            # Calculated losses
            train_losses = []

            train_percent = len(train_data) // self.batch_size // 100

            # The data is batched. For each batch...
            for i, (instr_id, page, next_page, x_train, y_train) in enumerate(self.batch(train_data)):
                with torch.no_grad():
                    # prediction for training set
                    x_train = x_train.view(-1, 64)
                    output_train = self.model(x_train)

                    # computing the training loss and accuracy
                    loss_train = criterion(output_train, y_train)
                    acc = self.accuracy(output_train, y_train)

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
                    x_test = x_test.view(-1, 64)
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

    def represent_with_tolerance(self, addresses, first_page, box=False):
        # This function takes a list of memory accesses and returns
        # a valid output for the MLP model, taking the tolerance into
        # consideration

        # This is a list that contains block offsets for every address
        blocks = [(address >> 7) % 32 for address in addresses]

        # This is a list that contains a page address for every address
        pages = [(address >> 12) for address in addresses]

        # This represents the output layer
        raw = [0 for _ in range(64)]
        
        # For each block...
        for i, block in enumerate(blocks):

            # If the block is one of the lookahead blocks, give it
            # a progressively ascending weight
            if i < self.lookahead:
                weight = ((i+1) / (self.lookahead+1))

            # If the block is one of the k blocks after the
            # lookahead, the weight will be 1
            elif i < self.lookahead + self.k:
                weight = 1.0
            # If not, the weight will start decreasing as it gets
            # further. I use tolerance+1 so that the last block
            # of the tolerance window doesn't have a weight of 0
            else:
                weight = 1.0 - ((i-(self.lookahead+self.k)+1) / (self.tolerance+1))
            
            # If the block is located in the current page, then add it to 
            # the first positions of the layer. Also, always keep the highest
            # value on each position of the layer
            if first_page == pages[i]:
                raw[block] = max(raw[block], weight)
            else:
                raw[32 + block] = max(raw[32 + block], weight)
                
        # Box is true when creating the input, and false when creating
        # the output
        if box:
            return [raw]
        else:
            return raw

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
    
    def train_and_test(self, train_data, test_data, model_name = None, graph_name = None):
        for prefetcher in self.prefetchers:
            prefetcher.train_and_test(self, train_data, test_data, model_name = None, graph_name = None)

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


ml_model_name = os.environ.get('ML_MODEL_NAME', 'MLPBasedSubPrefetcher')
Model = eval(ml_model_name)