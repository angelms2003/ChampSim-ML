# LSTM-Based Prefetcher (model-lstm-delta-stateful-thread-id-optuna)
# Implementación de LSTM stateful usando TBPTT (Truncated Backpropagation
# Through Time) para poder mantener la historia de todos los accesos vistos

from abc import abstractmethod
from collections import defaultdict

import torch
import torch.nn as nn
import sys

import optuna
import joblib
import os
import traceback

import matplotlib.pyplot as plt

# El tamaño de entrada del embedding de delta de página.
# Con un valor de 4097 se tienen:
# · 2048 páginas hacia atrás [-2048,-1]
# · 2048 páginas hacia adelante [1,2048]
# · La página actual [0]
# En caso de que el delta sea más grande, se tendrá que
# hacer un clamp (llevarlo a -2048 o 2048)
DELTA_PAGE_EMBEDDING_INPUT = 4097

# El tamaño del vector de embedding de delta de página
DELTA_PAGE_EMBEDDING_OUTPUT = 24


# El tamaño de entrada del embedding de offset de bloque.
# Como las páginas son de 4096 bytes y los bloques de
# caché son de 128 bytes, se utilizan 32 bits de entrada
# (el offset de bloque dentro de la página)
BLOCK_OFFSET_EMBEDDING_INPUT = 32

# El tamaño del vector de embedding de offset de bloque. Es
# el vector más pequeño porque hay pocos offsets de bloque
# posibles (solo 32)
BLOCK_OFFSET_EMBEDDING_OUTPUT = 12


# El tamaño del estado oculto de la LSTM. 128 es un valor
# lo suficientemente grande para calcular patrones complejos.
# Internamente, la LSTM tiene 4 x 128 = 512 neuronas, porque
# tiene 4 puertas o "gates": input, output, forget y cell
LSTM_HIDDEN_SIZE = 128

# El número de capas LSTM a utilizar. Se utilizan 2: la primera
# aprende patrones sencillos mientras que la segunda aprende
# patrones más complejos
LSTM_NUM_LAYERS = 2

# The dropout probability to use in the LSTM (this will only be
# used if there are 2 layers or more)
LSTM_DROPOUT = 0.0


class LSTM(nn.Module):
    def __init__(self,
                 delta_page_embed_in:int=DELTA_PAGE_EMBEDDING_INPUT,
                 page_embed_dim:int=DELTA_PAGE_EMBEDDING_OUTPUT,
                 block_embed_dim:int=BLOCK_OFFSET_EMBEDDING_OUTPUT,
                 hidden_size:int=LSTM_HIDDEN_SIZE,
                 num_layers:int=LSTM_NUM_LAYERS,
                 dropout:float=LSTM_DROPOUT):
        """
            This is the constructor for the LSTM class

            Args:
                delta_page_embed_in (int, optional):    The input size for the delta page embedding
                page_embed_dim (int, optional):         The output size for the delta page embeddings
                block_embed_dim (int, optional):        The output size for the block offset embedding
                hidden_size (int, optional):            The size of the hidden state for the LSTM
                num_layers (int, optional):             The number of LSTM layers
                dropout (float, optional):              The dropout probability to use
        """
        super().__init__()

        # The hyperparameters are printed for debugging purposes
        print("Initializing LSTM object with the following parameters:")
        print(f"\t- delta_page_embed_in = {delta_page_embed_in}")
        print(f"\t- page_embed_dim = {page_embed_dim}")
        print(f"\t- block_embed_dim = {block_embed_dim}")
        print(f"\t- hidden_size = {hidden_size}")
        print(f"\t- num_layers = {num_layers}")
        print(f"\t- dropout = {dropout}")

        # The hyperparameters are stored for later
        self.delta_page_embed_in = delta_page_embed_in
        self.page_embed_dim = page_embed_dim
        self.block_embed_dim = block_embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Tabla de embeddings para representar los deltas
        # de páginas a partir de la segunda página de la
        # secuencia de accesos
        self.page_delta_embedding = nn.Embedding(delta_page_embed_in, page_embed_dim)  # ±2048
        
        # Tabla de embeddings para representar el offset
        # de bloque dentro de la página
        self.block_offset_embedding = nn.Embedding(BLOCK_OFFSET_EMBEDDING_INPUT, block_embed_dim)
        
        # Esta es la capa LSTM que procesará las secuencias
        # de accesos
        self.lstm = nn.LSTM(
            # El tamaño del vector de entrada en cada paso
            # temporal, donde se mezclan los valores del
            # embedding de delta de página y los
            # valores del embedding de offset de bloque
            input_size=page_embed_dim + block_embed_dim,

            # Dimensión del estado oculto de la LSTM. Es el
            # vector de memoria que la LSTM mantiene mientras
            # se procesa la secuencia
            hidden_size=hidden_size,

            # El número de capas LSTM apiladas. Las primeras
            # capas capturan patrones simples, mientras
            # que las últimas capturan patrones de mayor nivel
            # sobre los patrones capturados por las anteriores
            num_layers=num_layers,

            # Define el formato de los tensores de entrada/salida.
            # La forma es (batch_size, sequence_length_features).
            # Por ejemplo: (256, 4, 36) indica que:
            # · Hay 256 ejemplos en cada batch.
            # · Cada ejemplo es una secuencia de 4 accesos.
            # · Cada acceso tiene 36 elementos (repartidos entre
            #   elementos del embedding de delta de página y
            #   elementos del embedding de offset de bloque)
            batch_first=True,

            # Se añade dropout si hay 2 capas o más. Esto permite
            # desactivar algunas neuronas al pasar de una capa a
            # otra, permitiendo obtener resultados más diversos y
            # disminuyendo el overfitting
            dropout = dropout if num_layers > 1 else 0.0
        )
        
        # Una capa fully connected que toma el último estado oculto
        # de la LSTM y devuelve un delta de página
        self.fc_page_delta = nn.Linear(hidden_size, delta_page_embed_in)

        # Una capa fully connected que toma el último estado oculto
        # de la LSTM y devuelve un offset de bloque
        self.fc_block = nn.Linear(hidden_size, BLOCK_OFFSET_EMBEDDING_INPUT)

        # A dictionary that saves the state of the model for each different
        # thread ID, since different threads will have different memory
        # access patterns, and mixing them will only confuse the model
        self.states = dict()
    
    def forward_sequence(self,
                         sequence:torch.Tensor,
                         thread_id:int,
                         prev_state:tuple=None):
        """
            This method defines the forward pass of data through the model.
            It processes a whole sequence of accesses for a given thread ID.
        
        Args:
            sequence (torch.Tensor):    A tensor of shape (seq_len, 2), where seq_len
                                        is the length of the sequence, and 2 means that,
                                        for each element, the page address and the block
                                        offset are given

            thread_id (int):            The ID of the thread that requested the memory
                                        accesses in the sequence

            prev_state (tuple):         The previous state for this thread ID. If this is
                                        the first sequence for this thread ID, prev_state
                                        should be None

        Returns:
            predictions:    List of tuples, where each tuple contains the page delta
                            logits and the block logits
            
            final_state:    Tuple with the hidden state, cell state and last page of
                            the sequence, allowing the state to be reused for the next
                            sequence of the same thread ID
        """

        # The device where the model is located is obtained. This will help
        # put all tensors on the same device
        device = next(self.parameters()).device

        # The length of the sequence
        seq_len = sequence.shape[0]

        # Initialize a new state or use previous state
        if prev_state is None:
            hidden = torch.zeros(self.num_layers, 1, self.hidden_size, device=device)
            cell = torch.zeros(self.num_layers, 1, self.hidden_size, device=device)
            last_page = None
        else:
            hidden, cell, last_page = prev_state
        
        # Here we will store the embeddings for each input
        embedded_inputs = []

        # Here we will store the predicted page deltas and block offsets for
        # each input
        predictions = []

        # The embeddings are prepared for the whole sequence
        for t in range(seq_len):
            # The page address and the block offset are obtained
            page = sequence[t, 0].to(device)
            block = sequence[t, 1].to(device)

            # The delta with respect to the previous page is calculated
            if last_page is not None:
                delta = (page - last_page)
            else:
                delta = torch.tensor(0, dtype=torch.long, device=device)
            
            # The delta is clamped to the given limits. The variable
            # max_value_delta stores the maximum absolute value for a delta
            max_value_delta = (self.delta_page_embed_in - 1) // 2
            delta_clamped = torch.clamp(delta, -max_value_delta, max_value_delta)

            # Once the delta is clamped, the index is calculated (the model can't
            # be fed a negative value for the delta, so the range [-2048,2048], for
            # example, is transformed into the range [0, 4096], where the index 0
            # represents a delta of -2048)
            delta_index = delta_clamped + max_value_delta
            delta_index = torch.clamp(delta_index, 0, self.delta_page_embed_in - 1)

            # Now, we can calculate the embeddings for both the page delta and the
            # block offset. The unsqueeze method is used to create a two-dimensional
            # tensor, even if there is just one element
            # print("="*40)
            # print(f"Delta index before unsqueezing (shape {delta_index.shape}): {delta_index}. Delta index after unsqueezing (shape {delta_index.unsqueeze(0).shape}): {delta_index.unsqueeze(0)}")
            # print(f"Block offset before unsqueezing (shape {block.shape}): {block}. Block offset after unsqueezing (shape {block.unsqueeze(0).shape}): {block.unsqueeze(0)}")
            page_emb = self.page_delta_embedding(delta_index.unsqueeze(0))
            block_emb = self.block_offset_embedding(block.unsqueeze(0))
            # print(f"Page embedding (shape {page_emb.shape}): {page_emb}")
            # print(f"Block embedding (shape {block_emb.shape}): {block_emb}")

            # The embeddings are concatenated to form an input for the LSTM. The
            # concatenation is done using the last dimension, causing all
            # elements of the resulting embedding to be in the same dimension
            input_emb = torch.cat([page_emb, block_emb], dim=-1)
            # print(f"Input embedding (shape {input_emb.shape}): {input_emb}")
            embedded_inputs.append(input_emb)

            # The last page is saved for the next access
            last_page = page
        
        # The embeddings are stacked. The resulting tensor is of shape
        # (seq_len, 1, input_size)
        embedded_seq = torch.stack(embedded_inputs, dim=0)

        # However, PyTorch expects the first dimension to be the batch size
        # and the second dimension to be the sequence length, so we
        # need to change both dimensions. The resulting tensor is of shape
        # (1, seq_len, input_size)
        embedded_seq = embedded_seq.transpose(0, 1)

        # The whole sequence is now processed by the LSTM
        lstm_out, (hidden_new, cell_new) = self.lstm(embedded_seq, (hidden, cell))

        # Now, each LSTM output is used to calculate a prediction. There are as many
        # predictions as timesteps. Each LSTM output is of shape (1, seq_len, hidden_size)
        for t in range(seq_len):
            # The hidden state for this timestep is obtained. The hidden state
            # contains info about the sequence of memory accesses for the current
            # thread ID
            state_t = lstm_out[0, t, :]

            # The predictions are done using the fully connected layers. The
            # input for these layers is the hidden state
            page_delta_pred = self.fc_page_delta(state_t.unsqueeze(0))
            block_pred = self.fc_block(state_t.unsqueeze(0))

            # The prediction is appended to the predictions list
            predictions.append((page_delta_pred, block_pred))
        
        # The predictions and the final state are returned
        return predictions, (hidden_new, cell_new, last_page)
    
    def reset_state(self, thread_id:int=None):
        """
            This method resets the state for a specific thread ID or
            for all thread IDs

        Args:
            thread_id (int, optional):  thread ID whose state will be reset.
                                        Defaults to None. If None, all thread
                                        ID's states will be reset
        """

        if thread_id is None:
            self.states = dict()
        else:
            if thread_id in self.states:
                del self.states[thread_id]
    
    def get_state(self, thread_id:int):
        """
            Returns the state for a given thread ID

        Args:
            thread_id (int):    ID for the thread whose state will be returned
        
        Returns:
            A tuple (hidden_state, cell_state, last_page) if the thread ID has
            a state, or None if the thread ID doesn't have a state
        """
        return self.states.get(thread_id, None)

    def get_config(self):
        """
            Returns the model's hyperparameters as a dictionary.

        Returns:
            dict:   Dictionary where the keys are strings (the names of
                    the hyperparameters) and they values are numbers
                    indicating the value of each hyperparameter.
        """

        return {
            "delta_page_embed_in": self.delta_page_embed_in,
            "page_embed_dim": self.page_embed_dim,
            "block_embed_dim": self.block_embed_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout
        }
    

# This is the seed that Optuna will use for initializing the Tree-structured
# Parzen Estimator
OPTUNA_TPE_SEED = 21

class OptunaHyperparameterSearch:
    """
        This class uses Optuna for Bayesian hyperparameter optimization.
    """
    
    def __init__(self, train_data:list, test_data:list, base_output_dir:str="experiments"):
        """
            The class constructor. It initializes important attributes.

        Args:
            train_data (list):                  A list containing the memory accesses used
                                                to train the model for which the hyperparameters
                                                are being found.
            test_data (list):                   A list containing the memory accesses used
                                                to test the model and check if the found
                                                hyperparameters are good.
            base_output_dir (str, optional):    The directory where the results of each experiment
                                                will be saved. Defaults to "experiments".
        """
        self.train_data = train_data
        self.test_data = test_data
        self.base_output_dir = base_output_dir
        self.trial_count = 0
    
    def optimize(self, study_name:str, n_trials:int=50):
        """
            This method performs Bayesian optimization using Optuna, finding the best
            hyperparameters
        
            Args:
                study_name(str):            The name for this study.
                n_trials(int, optional):    Number of trials to run. Defaults to 50.
            
            Returns:
                A tuple containing the best parameters and the study object
        """

        # A study object is created. It will coordinate the whole optimization
        study = optuna.create_study(
            # We tell the study that we want to maximize the goal metric.
            # In this case, the goal metric will be test_tolerant_acc. If we
            # wanted to use the test loss as the goal metric, we should assign
            # 'minimize' to direction
            direction='maximize',

            # We specify the name for the study
            study_name=study_name,

            # We initialize the Tree-structured Parzen Estimator (TPE) using the seed
            # that we previously defined. The TPE allows the study to learn from
            # previous trials to suggest better parameters for the next trial
            sampler=optuna.samplers.TPESampler(seed=OPTUNA_TPE_SEED)
        )
        
        # We execute the self._objetive method (defined below) n_trials times. In
        # each trial, Optuna will suggest different hyperparameters
        study.optimize(self._objective, n_trials=n_trials, show_progress_bar=True)
        
        # After the study is finished, we print the results to the screen
        print("="*70)
        print("OPTIMIZATION COMPLETED")
        print("="*70)
        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best test tolerant accuracy: {study.best_value:.4f}")
        print(f"Best parameters:")
        for key, value in study.best_params.items():
            print(f"\t{key}: {value}")
        
        # Then, we use joblib to save the whole study just in case we need
        # all trials later. Before that, we need to create the base output
        # directory just in case it doesn't exist (with exist_ok we tell
        # the makedirs method that we don't want an exception to be thrown
        # if the directory already exists)
        os.makedirs(self.base_output_dir, exist_ok=True)
        joblib.dump(study, f"{self.base_output_dir}/optuna_study.pkl")
        
        # Finally, we return the best hyperparameters
        return study.best_params
    
    def _objective(self, trial:optuna.Trial):
        """
            This is the objective function for Optuna.
        
            Args:
                trial(optuna.Trial):    An Optuna trial object.
            
            Returns:
                Test tolerant accuracy (metric to maximize)
        """

        # The number of trials is increased by one
        self.trial_count += 1
        
        # These are the hyperparameters that we will be using for the model.
        # Each hyperparameter has a name and a list or a range of possible
        # values to choose from
        model_config = {
            # Size for the delta page embedding input. Optuna chooses one from the list
            "delta_page_embed_in": trial.suggest_categorical("delta_page_embed_in", [(2**10)*2+1, (2**11)*2+1, (2**12)*2+1, (2**13)*2+1, (2**14)*2+1, (2**15)*2+1]),

            # Size for the page embedding output. Optuna chooses one from the list
            "page_embed_dim": trial.suggest_categorical("page_embed_dim", [16, 24, 32, 48, 64]),
            
            # Size for the block embedding output. Optuna chooses one from the list
            "block_embed_dim": trial.suggest_categorical("block_embed_dim", [8, 12, 16, 20, 24]),
            
            # Size for the LSTM hidden size. Optuna chooses one from the list
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 96, 128, 192, 256]),
            
            # Number of layers for the LSTM. Optuna chooses one integer in the
            # range [1,3]
            "num_layers": trial.suggest_int("num_layers", 1, 3),

            # Dropout rate to use when num_layers is more than 1. Optuna chooses
            # a random float in the range [0.0,0.3]
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        }

        # This is similar to the previous dictionary, but it only contains
        # hyperparameters that are relevant for training
        training_config = {
            # Learning rate to use during training. Optuna chooses one float in the
            # range [0.0001,0.01]
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2),

            # Number of accesses to skip in order to avoid prefetching too early.
            # Optuna chooses one integer in the range [3,7]
            "lookahead_size": trial.suggest_int("lookahead_size", 3, 7),

            # Number of training examples in one batch. Optuna chooses one
            # from the list
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),

            # Number of accesses in each TBPTT sequence. Optuna chooses one
            # integer in the range [10,20]
            "tbptt_length": trial.suggest_int("tbptt_length", 10, 20),

            # Number of training epochs. They are fixed to 10 to speed up
            # trials, but will surely be larger during training once the best
            # hyperparameters have been found
            "num_epochs": 10
        }
        
        # For each trial, we print the chosen hyperparameters
        print(f"\nTrial {self.trial_count}")
        print("Model architecture:")
        for key, value in model_config.items():
            print(f"  {key}: {value}")
        print("Training configuration:")
        for key, value in training_config.items():
            print(f"  {key}: {value}")
        
        try:
            # The model is created, and the model configuration is
            # specified as a parameter to the constructor
            prefetcher = LSTMBasedPrefetcher(model_config)

            # The training configuration is set to the values chosen
            # by Optuna
            prefetcher.set_training_config(training_config)
            
            # We set the path where the model will be saved and create the
            # directory where it will be located
            #experiment_name = f"trial_{self.trial_count:03d}"
            #model_path = f"{self.base_output_dir}/{experiment_name}/model"
            #os.makedirs(f"{self.base_output_dir}/{experiment_name}", exist_ok=True)
            
            # We train and test the model. We set graph_name to None to
            # avoid saving graphs for trials
            metrics = prefetcher.train_and_test(
                self.train_data,
                self.test_data,
                model_name=None,#model_path,
                graph_name=None
            )
            
            # After finishing the training and testing, we obtain the value
            # for the tolerant accuracy during test (the metric to maximize)
            test_tolerant_acc = metrics["test_tolerant_acc"]
            print(f"Result: {test_tolerant_acc:.4f}")
            
            # We report the metric of this trial to Optuna, which uses it
            # for pruning (i.e., stopping the current trial if the model
            # is not working well)
            trial.report(test_tolerant_acc, step=0)
            
            # Finally, we return the recorded metric
            return test_tolerant_acc

        # If any exception occurred, we return the worst possible metric (in
        # this case, 0 is the worst tolerant accuracy)          
        except Exception as e:
            print(f"ERROR: {str(e)}")
            traceback.print_exc()
            return 0.0

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

class LSTMBasedPrefetcher(MLPrefetchModel):
    # This allows for the calculation of a "tolerant accuracy" which takes
    # into account the fact that a prefetch can be useful even if the access
    # to the prefetched block is not exactly lookahead_size accesses ahead
    tolerance_size = 20

    # The number of memory acesses to skip when prefetching. This
    # help prevent early prefetches
    lookahead_size = 5

    # The number of training epochs for the LSTM
    num_epochs = 30

    # The default size for a batch
    batch_size = 256

    # The learning rate for the LSTM
    learning_rate = 0.002

    # The length of the sequence for TBPTT. When calling backward(), gradients
    # are used to calculate the new weights of the neural network. If backward()
    # is called after a large amount of accesses, it can be very slow and consume
    # a lot of RAM. Instead, backward() is called every few steps to calculate
    # weights continuously, saving RAM and time. The model can still be stateful:
    # the hidden state is kept even after calling backward.
    tbptt_length = 15

    def __init__(self, model_config:dict=None):
        """
            Constructor for the LSTMBasedPrefetcher class.

        Args:
            model_config (dict, optional):  A dictionary where the keys are strings (
                                            the names of the hyperparameters) and the
                                            values are numbers indicating (the value of
                                            each hyperparameter). See LSTM.get_config()
                                            for more details.
        """

        # If no configuration was provided, the model is initialized
        # using the default values
        if model_config is None:
            self.model = LSTM()
        else:
            self.model = LSTM(**model_config)
    
    def set_training_config(self, config:dict):
        """
            Allows to change the default training configuration.

        Args:
            config (dict):  Dictionary where the keys are strings (the names of the
                            hyperparameters) and the values are numbers indicating
                            the value of each hyperparameter. An example for this
                            dictionary is the following:
                            {
                                "lookahead_size": 7,
                                "learning_rate": 0.001,
                                "num_epochs": 20,
                                "batch_size": 512,
                                "tbptt_length": 15
                            }
        """
        # It's important to check if each hyperparameter is present in
        # the dictionary before using it
        if "lookahead_size" in config:
            self.lookahead_size = config["lookahead_size"]
        if "learning_rate" in config:
            self.learning_rate = config["learning_rate"]
        if "num_epochs" in config:
            self.num_epochs = config["num_epochs"]
        if "batch_size" in config:
            self.batch_size = config["batch_size"]
        if "tbptt_length" in config:
            self.tbptt_length = config["tbptt_length"]

    def load(self, path:str):
        """
            This function loads the model from the given path

        Args:
            path (str): Path where the file that stores the model
                        is located
        """
        self.model = torch.jit.load(path)
        
        # If cuda is available, the model is moved to the GPU 
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def save(self, path:str):
        """
            Saves the model to the given path

            Args:
                path (str): path to the file that will
                            store the model
        """

        # The model is set to training mode (no gradients will
        # be updated)
        self.model.eval()
        
        # Option A: torch.jit.script (more robust). This function
        # directly compiles the Python code without the need of
        # an input example
        try:
            scripted = torch.jit.script(self.model)
            scripted.save(path)
            print(f"Model successfully saved using torch.jit.script to {path}")
        except Exception as e:
            print(f"torch.jit.scrip failed: {e}")
            print("Trying to save using torch.jit.trace instead")
            
            # Option B: if torch.jit.script fails, we will try to use
            # torch.jit.trace. This function needs an example input, so
            # we will have to manually create it. For simplicity, we will
            # assume a batch size of 1

            # The example_seq tensor will be a (5, 2) tensor containing a sequence
            # of 5 randomly generated memory accesses. 
            example_seq = torch.randint(0, BLOCK_OFFSET_EMBEDDING_INPUT, (5, 2))

            # This is an example thread ID for the example sequence
            example_thread_id = 12345

            # If cuda is available, the example_seq tensor is moved to the GPU 
            if torch.cuda.is_available():
                example_seq = example_seq.cuda()
            
            # The input is used to save the model via the trace function
            traced = torch.jit.trace(
                self.model.forward_sequence,
                (example_seq, example_thread_id, None)
            )
            traced.save(path)
            print(f"Model successfully saved using torch.jit.trace to {path}")

    
    def create_tbptt_sequences(self, data:list):
        """
            This method groups the accesses by thread ID and creates sequences
            for truncated BPTT.

        Args:
            data (list):    List of accesses. Each access is another list with
                            [instr_id, cycle, load_address, ip, thread_id, cache_hit]
            
        Yields:
            A tuple (thread_id, sequence_tensor, targets_list, whole_windows_list)
            where:
                - sequence_tensor has a shape (seq_len, 2) where each row contains [page, block]
                - targets_list is a list of (target_delta_idx, target_block)
                - whole_windows_list is a list of whole windows used for accuracy
        """

        # Accesses are grouped by thread ID
        thread_id_sequences = defaultdict(list)

        # Each access is read
        for line in data:
            # The info for this access is stored in variables
            instr_id, cycle, load_address, ip, thread_id, cache_hit = line

            # The page is calculated
            page = load_address >> 12

            # The block offset is calculated
            block_offset = (load_address >> 7) & 0x1F

            # The access is added to the corresponding thread ID sequence
            thread_id_sequences[thread_id].append([page, block_offset])
        
        # For each thread ID we create sequences of length tbptt_length
        for thread_id, accesses in thread_id_sequences.items():
            # We need at least tbptt_length + lookahead_size + tolerance_size accesses
            # to be able to calculate the accuracy for each prediction in the sequence
            min_length = self.tbptt_length + self.lookahead_size + self.tolerance_size

            # If the current thread ID doesn't have enough accesses, it is discarded
            if len(accesses) < min_length:
                continue
                
            # If there are enough accesses, we can create sequences
            for start_idx in range(0, len(accesses) - min_length + 1, self.tbptt_length):
                # The end index is the index of the last access for the TBPTT sequence. In
                # order for us to properly calculate the accuracy, after each TBPTT sequence
                # it is necessary to add the predicted access and some more accesses for
                # tolerant accuracy calculation.
                end_idx = min(start_idx + self.tbptt_length, len(accesses) - self.lookahead_size - self.tolerance_size - 1)

                # If there are not enough accesses for both the TBPTT sequence and the
                # rest of the accesses for tolerant accuracy calculation, the TBPTT sequence
                # cannot be formed
                if end_idx <= start_idx:
                    break
                    
                # If there are enough accesses, the sequence length is calculated. This should
                # be tbptt_length, but if we are close to the end it could be shorter.
                seq_len = end_idx - start_idx
                sequence = accesses[start_idx:end_idx]

                # In these lists we will store the targets and the windows for each timestep in the sequence
                targets = []
                whole_windows = []

                # We iterate for each input access in the sequence
                for t in range(seq_len):
                    # The target access (the access we want to predict) is
                    # lookahead_size + 1 steps ahead
                    target_idx = start_idx + t + 1 + self.lookahead_size

                    # The target should be within the access window
                    assert target_idx < len(accesses), "The target index is beyond the access window's limits"

                    # We obtain the page address for the current access
                    current_page = accesses[start_idx + t][0]
                    
                    # We obtain the page address and the block offset of the
                    # target access we want to predict
                    target_page = accesses[target_idx][0]
                    target_block = accesses[target_idx][1]

                    # We calculate the delta between both pages and transform
                    # it into an index
                    delta = target_page - current_page
                    max_value_delta = (self.model.delta_page_embed_in - 1) // 2
                    delta_clamped = max(-max_value_delta, min(max_value_delta, delta))
                    delta_index = delta_clamped + max_value_delta
                    delta_index = max(0, min(self.model.delta_page_embed_in - 1, delta_index))

                    # We append the target delta and the target block offset to
                    # the list of targets for this sequence
                    targets.append((delta_index, target_block))

                    # Now we select the accesses that will form the window that will later
                    # be used to calculate accuracy
                    window_end = min(
                        start_idx + t + 1 + self.lookahead_size + self.tolerance_size,
                        len(accesses)
                    )

                    whole_window = accesses[start_idx+t:window_end]
                    whole_windows.append(whole_window)
                
                # After creating this TBPTT sequence, the targets list should contain
                # at least one target, corresponding to one input
                assert len(targets) > 0, "No valid targets were found"
                
                # The sequence is transformed into a tensor. This tensor will be the
                # input, containing as many input accesses as there are targets
                device = next(self.model.parameters()).device
                sequence_tensor = torch.LongTensor(sequence[:len(targets)])

                yield ip, sequence_tensor, targets, whole_windows

    
    def train_and_test(self, train_data:list, test_data:list, model_name:str=None, graph_name:str=None):
        """
            This function trains the LSTM model on the given training
            data and then tests it on the given test data
        
        Args:
            train_data (list):  List of memory accesses used to train
                                the model
            test_data (list):   List of memory accesses used to test
                                the model
            model_name (str):   Path to save model checkpoints including the
                                name of the checkpoing file. For example,
                                model_name can be "models/my_model" so that
                                the checkpoints will be saved as
                                "models/my_model-1.pt", "models/my_model-2.pt",
                                etc.
            graph_name (str):   Path to save training graphs
        """
        print(f"Called train_and_test method. Training with the following hyperparameters:")
        print(f"\t- learning_rate = {self.learning_rate}")
        print(f"\t- lookahead_size = {self.lookahead_size}")
        print(f"\t- batch_size = {self.batch_size}")
        print(f"\t- num_epochs = {self.num_epochs}")
        print(f"\t- tbptt_length = {self.tbptt_length}")

        # If the model's loss during test doesn't improve for 3 epochs,
        # training ends to avoid overfitting
        patience = 3

        # This indicates the best (lowest) loss found during test
        best_test_loss = float("inf")

        # This indicates for how many epochs the model hasn't improved
        # its loss during test
        epochs_without_improvement = 0

        # The optimizer used to update the model's parameters
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # The loss functions for page delta and block predictions.
        # CrossEntropyLoss is better for classification tasks (predicting indices)
        criterion_page = nn.CrossEntropyLoss()
        criterion_block = nn.CrossEntropyLoss()

        # If there is a GPU available, the model and the loss functions are moved
        # there to accelerate computations
        if torch.cuda.is_available():
            print("Using CUDA")
            self.model = self.model.cuda()
            criterion_page = criterion_page.cuda()
            criterion_block = criterion_block.cuda()
        else:
            print("NOT using CUDA")

        # These lists will store relevant metrics that will be plotted later
        avg_train_strict_accs = []
        avg_train_tolerant_accs = []
        avg_test_strict_accs = []
        avg_test_tolerant_accs = []
        total_train_loss = []
        total_test_loss = []

        # Training loop
        for epoch in range(self.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"{'='*60}")
            
            # ============================================================
            # TRAINING PHASE
            # ============================================================

            # The model is set to training mode to allow its parameters to be updated
            self.model.train()

            # The model's state is reset for all thread ID's since a new epoch has
            # just begun and we shoulnd't have memory from the last epoch (the
            # weights will be updated, but the model won't remember having seen
            # any accesses before)
            self.model.reset_state()
            
            # The training accuracies and losses will be stored here
            train_strict_accs = []
            train_tolerant_accs = []
            train_losses = []

            # This counter will be increased by 1 for each tbptt sequence
            sequence_count = 0

            print("Training...")
            for thread_id, sequence, targets, whole_windows in self.create_tbptt_sequences(train_data):
                sequence_count+=1

                # We obtain the previous state for this thread ID (or None if this is the first time
                # that the model sees memory accesses requested by this thread ID)
                prev_state = self.model.get_state(thread_id)

                # If there is a previous state, we call detach() to truncate backpropagation
                # through time, since the gradients should have already been calculated for
                # the last sequence of this thread ID
                if prev_state is not None:
                    hidden, cell, last_page = prev_state
                    prev_state = (hidden.detach(), cell.detach(), last_page)
                
                # The whole sequence is processed
                predictions, final_state = self.model.forward_sequence(
                    sequence, thread_id, prev_state
                )

                # Now, we can calculate the loss and the accuracy for the predictions.
                # First, the gradients are cleared because we are going to calculate
                # new ones
                optimizer.zero_grad()

                # These variables will help calculate total loss and accuracy
                total_loss = 0
                seq_strict_correct = 0
                seq_tolerant_correct = 0

                # Now, we iterate through every example in the sequence
                for t, ((page_delta_pred, block_pred), (target_delta_idx, target_block), whole_window) in enumerate(
                    zip(predictions, targets, whole_windows)
                ):
                    # The target tensors are created
                    target_delta_tensor = torch.LongTensor([target_delta_idx])
                    target_block_tensor = torch.LongTensor([target_block])

                    # If cuda is available, the tensors are moved to the GPU
                    if torch.cuda.is_available():
                        target_delta_tensor = target_delta_tensor.cuda()
                        target_block_tensor = target_block_tensor.cuda()
                    
                    # Now, the loss is calculated for the deltas and the block offsets,
                    # and both are accumulated into one global loss
                    loss_page = criterion_page(page_delta_pred, target_delta_tensor)
                    loss_block = criterion_block(block_pred, target_block_tensor)
                    total_loss += loss_page + loss_block

                    # After calculating the loss, the accuracy is calculated.
                    # The page delta is obtained by finding the index with the
                    # highest value in the page delta index tensor. This index
                    # is then transformed into a delta
                    pred_delta_idx = torch.argmax(page_delta_pred, dim=1).item()
                    max_value_delta = (self.model.delta_page_embed_in - 1) // 2
                    pred_delta = pred_delta_idx - max_value_delta
                    pred_delta = max(-max_value_delta, min(pred_delta, max_value_delta))

                    # The predicted block is easier to obtain
                    pred_block = torch.argmax(block_pred, dim=1).item()

                    # Now, both accuracies (strict and tolerant) are calculated, and their
                    # result is added to the counters
                    strict, tolerant = self.accuracy(pred_delta, pred_block, whole_window)
                    seq_strict_correct += strict
                    seq_tolerant_correct += tolerant
                
                # Once the sequence was fully treated, backward pass is done (the
                # calculated gradients are used to update the model's weights)
                total_loss.backward()
                optimizer.step()

                # The final state is saved
                self.model.states[thread_id] = final_state

                # The metrics are saved
                train_losses.append(total_loss.item())
                train_strict_accs.append(seq_strict_correct / len(targets))
                train_tolerant_accs.append(seq_tolerant_correct / len(targets))

                # A period is shown on the terminal each 100 sequences
                if sequence_count % 100 == 0:
                    print(".", end="", flush=True)
            
            # After training, the average strict accuracy and the average tolerant
            # accuracy are calculated and printed
            avg_strict = sum(train_strict_accs) / len(train_strict_accs) if train_strict_accs else 0
            avg_tolerant = sum(train_tolerant_accs) / len(train_tolerant_accs) if train_tolerant_accs else 0
            print()
            print(f'Training strict accuracy: {avg_strict:.4f}')
            print(f'Training tolerant accuracy: {avg_tolerant:.4f}')
            print(f'Training loss: {sum(train_losses):.4f}')

            # Then, they are added to their lists
            avg_train_strict_accs.append(avg_strict)
            avg_train_tolerant_accs.append(avg_tolerant)
            total_train_loss.append(sum(train_losses))

            # ============================================================
            # TESTING PHASE
            # ============================================================

            # In this case, the model is set to evaluation mode
            self.model.eval()

            # Now, the model is reset in order for it to forget what accesses
            # it saw during training
            self.model.reset_state()
            
            # The test accuracies and losses will be stored here
            test_strict_accs = []
            test_tolerant_accs = []
            test_losses = []

            # Just like during training, this variable will be incremented by
            # 1 each time a new sequence is processed
            sequence_count = 0

            print("Testing...")

            # We use torch.no_grad() because it's not necessary to calculate
            # gradients during test
            with torch.no_grad():
                for thread_id, sequence, targets, whole_windows in self.create_tbptt_sequences(test_data):
                    sequence_count+=1

                    # We get the previous state for this thread ID (if any)
                    prev_state = self.model.get_state(thread_id)

                    # We perform the forward pass just like it was done during training
                    predictions, final_state = self.model.forward_sequence(
                        sequence, thread_id, prev_state
                    )

                    # In these variables we will accumulate the loss and the accuracy
                    total_loss = 0
                    seq_strict_correct = 0
                    seq_tolerant_correct = 0

                    # We iterate for each example in the test dataset
                    for t, ((page_delta_pred, block_pred), (target_delta_idx, target_block), whole_window) in enumerate(
                        zip(predictions, targets, whole_windows)
                    ):
                        # Tensors are created using the target delta index and the
                        # target block offset
                        target_delta_tensor = torch.LongTensor([target_delta_idx])
                        target_block_tensor = torch.LongTensor([target_block])
                        
                        # These tensors are moved to the GPU if available
                        if torch.cuda.is_available():
                            target_delta_tensor = target_delta_tensor.cuda()
                            target_block_tensor = target_block_tensor.cuda()
                        
                        # The loss for the page delta prediction and the loss for the block
                        # offset prediction are calculated and accumulated in one variable
                        loss_page = criterion_page(page_delta_pred, target_delta_tensor)
                        loss_block = criterion_block(block_pred, target_block_tensor)
                        total_loss += loss_page + loss_block

                        # The predicted delta is obtained from the predicted index
                        pred_delta_idx = torch.argmax(page_delta_pred, dim=1).item()
                        max_value_delta = (self.model.delta_page_embed_in - 1) // 2
                        pred_delta = pred_delta_idx - max_value_delta
                        pred_delta = max(-max_value_delta, min(pred_delta, max_value_delta))
                        
                        # The predicted block is easier to obtain (no calculations necessary)
                        pred_block = torch.argmax(block_pred, dim=1).item()
                        
                        # The strict and tolerant accuracy for this access is calculated
                        strict, tolerant = self.accuracy(pred_delta, pred_block, whole_window)
                        seq_strict_correct += strict
                        seq_tolerant_correct += tolerant
                    
                    # After this sequence, the final state for this thread ID is
                    # saved for later
                    self.model.states[thread_id] = final_state
                    
                    # The metrics are added to their lists
                    test_losses.append(total_loss.item())
                    test_strict_accs.append(seq_strict_correct / len(targets))
                    test_tolerant_accs.append(seq_tolerant_correct / len(targets))
                    
                    # A period is shown on the terminal each 100 sequences
                    if sequence_count % 100 == 0:
                        print('.', end='', flush=True)
                
            # The average accuracies for this epoch are calculated and shown
            # on the screen
            avg_strict = sum(test_strict_accs) / len(test_strict_accs) if test_strict_accs else 0
            avg_tolerant = sum(test_tolerant_accs) / len(test_tolerant_accs) if test_tolerant_accs else 0
            print()
            print(f'Test strict accuracy: {avg_strict:.4f}')
            print(f'Test tolerant accuracy: {avg_tolerant:.4f}')
            print(f'Test loss: {sum(test_losses):.4f}')

            # The average accuracies and the total loss are added to
            # the correct list
            avg_test_strict_accs.append(avg_strict)
            avg_test_tolerant_accs.append(avg_tolerant)
            total_test_loss.append(sum(test_losses))

            # If a model name was given, a checkpoint of the model is saved there
            if model_name is not None:
                checkpoint_path = f"{model_name}-epoch{epoch+1}.pt"
                self.save(checkpoint_path)
                print(f"Model saved to {checkpoint_path}")
            
            # The total current loss is calculated and, if the model is not
            # improving, early stopping is triggered
            current_loss = sum(test_losses)
        
            if current_loss < best_test_loss:
                print(f"Test loss improved from {best_test_loss:.6f} to {current_loss:.6f}")
                best_test_loss = current_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(f"No improvement in test loss. Patience: {epochs_without_improvement}/{patience}")
            
            if epochs_without_improvement >= patience:
                print("\nEarly stopping triggered")
                break

        # Once all epochs are completed, the results are plotted
        if graph_name is not None:
            self._plot_training_results(
                avg_train_strict_accs, avg_train_tolerant_accs,
                avg_test_strict_accs, avg_test_tolerant_accs,
                total_train_loss, total_test_loss, graph_name
            )
        
        # Finally, all metrics are returned
        return {
            'train_strict_acc': avg_train_strict_accs[-1] if avg_train_strict_accs else 0,
            'train_tolerant_acc': avg_train_tolerant_accs[-1] if avg_train_tolerant_accs else 0,
            'test_strict_acc': avg_test_strict_accs[-1] if avg_test_strict_accs else 0,
            'test_tolerant_acc': avg_test_tolerant_accs[-1] if avg_test_tolerant_accs else 0,
            'train_loss': total_train_loss[-1] if total_train_loss else float('inf'),
            'test_loss': total_test_loss[-1] if total_test_loss else float('inf'),
            'epochs_trained': len(avg_train_strict_accs)
        }

    def _plot_training_results(self, avg_train_strict_accs, avg_train_tolerant_accs,
                               avg_test_strict_accs, avg_test_tolerant_accs,
                               total_train_loss, total_test_loss, graph_name):
        """
            This is a helper function used by the train_and_test function
            to plot training results.
        """

        epochs = range(1, len(avg_train_strict_accs) + 1)

        plt.figure(figsize=(20, 5))

        # Strict Accuracy plot
        plt.subplot(1, 4, 1)
        plt.plot(epochs, avg_train_strict_accs, label='Train Strict Acc', marker='o')
        plt.plot(epochs, avg_test_strict_accs, label='Test Strict Acc', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Strict Accuracy (Exact Match)')
        plt.legend()
        plt.grid(True)

        # Tolerant Accuracy plot
        plt.subplot(1, 4, 2)
        plt.plot(epochs, avg_train_tolerant_accs, label='Train Tolerant Acc', marker='o')
        plt.plot(epochs, avg_test_tolerant_accs, label='Test Tolerant Acc', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Tolerant Accuracy (Within Window)')
        plt.legend()
        plt.grid(True)

        # Train Loss plot
        plt.subplot(1, 4, 3)
        plt.plot(epochs, total_train_loss, label='Train Loss', marker='o', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train Loss')
        plt.legend()
        plt.grid(True)

        # Test Loss plot
        plt.subplot(1, 4, 4)
        plt.plot(epochs, total_test_loss, label='Test Loss', marker='o', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Test Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{graph_name}.png", dpi=300)
        print(f"Training plots saved to {graph_name}.png")
    
    def accuracy(self, pred_delta:int, pred_block:int, whole_window:list):
        """
            This method calculates the model's accuracy for one prediction
        
        Args:
            pred_delta(int):    The predicted page delta
            
            pred_block(int):    The predicted block offset

            whole_window(list): The whole window of accesses, going from the
                                current access to the access that is located
                                self.lookahead_size + 1 + self.tolerance_size
                                accesses later
        
        Returns:
            tuple: (strict_accuracy, tolerant_accuracy), where each variable
            is 1 if the prediction was correct or 0 if it wasn't. If strict
            accuracy is 1, then tolerant accuracy should be 1 too, since the
            tolerant accuracy is less restrictive
        """

        # These flags will be returned later
        strict_correct = 0
        tolerant_correct = 0

        # The input page is the page corresponding to the first page
        # of the sequence
        input_page = whole_window[0][0]

        # Based on the predicted delta and the current page, we can
        # calculate the predicted page
        pred_page = input_page + pred_delta

        # The target access is located 1 + self.lookahead_size accesses later.
        # This will be used for strict accuracy calculation. The +1 represents
        # that, if self.lookahead_size is 4, for example, we want to skip 4
        # accesses and then predict the 5th one
        target_idx = 1 + self.lookahead_size

        # The target should be within the window
        assert target_idx < len(whole_window), "Error: while calculating the accuracy, the target access was not within the window"

        # The target page and the target block offset are obtained
        target_page = whole_window[target_idx][0]
        target_block = whole_window[target_idx][1]
        
        # If both the page and the block were correct, we can consider
        # this a success for the strict accuracy part
        if pred_page == target_page and pred_block == target_block:
            strict_correct = 1
        
        # Now, we need to check for tolerant accuracy. The first access
        # of the tolerant window is the access right after the current
        # access
        tolerance_start = 1

        # The end of the tolerance window should be 1 + self.lookahead_size
        # + self.tolerance_size accesses later, but maybe the window is
        # shorter, so there is no problem with that
        tolerance_end = min(
            1 + self.lookahead_size + self.tolerance_size,
            len(whole_window)
        )
        
        # For every access in the tolerance window...
        for j in range(tolerance_start, tolerance_end):
            # We obtain the page address and the block offset of that access
            window_page = whole_window[j][0]
            window_block = whole_window[j][1]
            
            # If the predicted page and the predicted block offset are
            # both correct, we don't need to keep iterating
            if pred_page == window_page and pred_block == window_block:
                tolerant_correct = 1
                break
        
        # If the strict prediction was correct, the tolerant
        # one should be too
        if strict_correct == 1:
            assert tolerant_correct == 1, "Error: strict_correct was 1, but tolerant_correct wasn't"

        # Finally both flags are returned
        return strict_correct, tolerant_correct

Model = LSTMBasedPrefetcher