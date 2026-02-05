# LSTM-Based Prefetcher

from abc import abstractmethod
from collections import defaultdict

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

# El tamaño de entrada para el embedding de página
# absoluta utilizado para representar la primera página
# accedida por la red. Con direcciones de 64 bits y bloques
# de 128 bytes, hay un total de 2^57 páginas posibles, lo cual
# es un número gigantesco. En vez de eso, se hace un hash con
# la dirección de página (se divide entre este tamaño
# de input de la tabla de embedding de página). Así, se mapean
# direcciones de página a valores entre 0 y 9999. Es
# cierto que habrá colisiones, pero merece la pena para
# conseguir un menor consumo de recursos
ABS_PAGE_EMBEDDING_INPUT = 10000

# El tamaño del vector de embedding de páginas absolutas.
# Este hiperparámetro determina la riqueza de la
# representación. Cuanto más grande sea el vector de
# embedding, más capacidad de expresión se obtendrá, pero
# también se complicará más la arquitectura de la red
ABS_PAGE_EMBEDDING_OUTPUT = 24


# El tamaño de entrada del embedding de delta de página.
# Con un valor de 4097 se tiene:
# · 2048 páginas hacia atrás [-2048,-1]
# · 2048 páginas hacia adelante [1,2048]
# · La página actual [0]
# En caso de que el delta sea más grande, se tendrá que
# hacer un clamp (llevarlo a -2048 o 2048). El tamaño de
# entrada es menor que en las páginas absolutas porque los
# deltas son más simples
DELTA_PAGE_EMBEDDING_INPUT = 4097

# El tamaño del vector de embedding de delta de página. Debe
# ser igual al tamaño del vector de embedding de páginas
# absolutas, pues la entrada de la LSTM debe medir lo mismo
# independientemente de si es el primer acceso o no
DELTA_PAGE_EMBEDDING_OUTPUT = ABS_PAGE_EMBEDDING_OUTPUT


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


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Tabla de embeddings para representar la primera
        # página de la secuencia de accesos (como es la
        # primera página no se puede utilizar un delta)
        self.page_abs_embedding = nn.Embedding(ABS_PAGE_EMBEDDING_INPUT, ABS_PAGE_EMBEDDING_OUTPUT)

        # Tabla de embeddings para representar los deltas
        # de páginas a partir de la segunda página de la
        # secuencia de accesos
        self.page_delta_embedding = nn.Embedding(DELTA_PAGE_EMBEDDING_INPUT, DELTA_PAGE_EMBEDDING_OUTPUT)  # ±2048
        
        # Tabla de embeddings para representar el offset
        # de bloque dentro de la página
        self.block_offset_embedding = nn.Embedding(BLOCK_OFFSET_EMBEDDING_INPUT, BLOCK_OFFSET_EMBEDDING_OUTPUT)
        
        # Esta es la capa LSTM que procesará las secuencias
        # de accesos
        self.lstm = nn.LSTM(
            # El tamaño del vector de entrada en cada paso
            # temporal, donde se mezclan los valores del
            # embedding de página (absoluto o delta) y los
            # valores del embedding de offset de bloque
            input_size=ABS_PAGE_EMBEDDING_OUTPUT + BLOCK_OFFSET_EMBEDDING_OUTPUT,

            # Dimensión del estado oculto de la LSTM. Es el
            # vector de memoria que la LSTM mantiene mientras
            # se procesa la secuencia
            hidden_size=LSTM_HIDDEN_SIZE,

            # El número de capas LSTM apiladas. Las primeras
            # capas capturan patrones simples, mientras
            # que las últimas capturan patrones de mayor nivel
            # sobre los patrones capturados por las anteriores
            num_layers=LSTM_NUM_LAYERS,

            # Define el formato de los tensores de entrada/salida.
            # La forma es (batch_size, sequence_length_features).
            # Por ejemplo: (256, 4, 36) indica que:
            # · Hay 256 ejemplos en cada batch.
            # · Cada ejemplo es una secuencia de 4 accesos.
            # · Cada acceso tiene 36 elementos (repartidos entre
            #   elementos del embedding de delta de página y
            #   elementos del embedding de offset de bloque)
            batch_first=True
        )
        
        # Una capa fully connected que toma el último estado oculto
        # de la LSTM y devuelve un delta de página
        self.fc_page_delta = nn.Linear(LSTM_HIDDEN_SIZE, DELTA_PAGE_EMBEDDING_INPUT)

        # Una capa fully connected que toma el último estado oculto
        # de la LSTM y devuelve un offset de bloque
        self.fc_block = nn.Linear(LSTM_HIDDEN_SIZE, BLOCK_OFFSET_EMBEDDING_INPUT)
    
    def forward(self, sequence):
        # Esta función define el flujo hacia adelante (forward pass) de
        # los datos a través de la red neuronal. El parámetro sequence
        # contiene el input que se va a procesar, el cual tiene un
        # tamaño de (batch, seq_len, 2):
        # · batch es el batch_size (tamaño de batch).
        # · seq_len es la longitud de la secuencia de accesos a memoria.
        # · Cada ejemplo tiene 2 datos (dirección de página y offset de
        #   bloque).
        # Un ejemplo concreto de entrada de tamaño (2,4,2) podría ser
        # el siguiente:
        # 
        # sequence = torch.tensor([
        #     # Secuencia 1:
        #     [[1000, 5],   # Acceso 1: página 1000, bloque 5
        #     [1000, 6],   # Acceso 2: página 1000, bloque 6
        #     [1001, 0],   # Acceso 3: página 1001, bloque 0
        #     [1001, 2]],  # Acceso 4: página 1001, bloque 2
        #
        #     # Secuencia 2:
        #     [[5234, 10],  # Acceso 1: página 5234, bloque 10
        #     [5234, 11],  # Acceso 2: página 5234, bloque 11
        #     [5240, 0],   # Acceso 3: página 5240, bloque 0
        #     [5240, 1]]   # Acceso 4: página 5240, bloque 1
        # ])
        
        # En esta lista inicialmente vacía se van a almacenar los embeddings
        # de cada paso temporal. Se irá añadiendo un elemento por cada paso
        # "t" en la secuencia. Al final se tendrán seq_len elementos en
        # esta lista.
        embedded = []

        # Se itera por cada paso temporal en la secuencia. Esto se hace así
        # para poder calcular los deltas de página correctamente (o, en caso
        # de la primera página, para calcular la dirección absoluta de página)
        for t in range(sequence.shape[1]):
            # Si estamos en el primer paso temporal, solo se tiene en cuenta
            # la dirección de la página de este acceso
            if t == 0:
                # Se calcula el embedding de página dada la dirección de
                # página (0 en la tercera dimensión) de cada ejemplo del
                # batch (dos puntos en la primera dimensión) en este paso
                # temporal (t en la segunda dimensión)
                page_emb = self.page_abs_embedding(sequence[:, t, 0] % ABS_PAGE_EMBEDDING_INPUT)
            else:
                # Si el paso temporal es 1 o más, tenemos una página
                # anterior con la que calcular el delta. Se obtiene
                # el valor del delta para cada ejemplo del batch
                # calculando la diferencia entre la dirección de página
                # de este acceso y la dirección de página del acceso
                # anterior
                delta = sequence[:, t, 0] - sequence[:, t-1, 0]

                # En caso de que el delta sea muy grande, se clampea
                # (es decir, se trae al límite más cercano)
                max_value_delta = (DELTA_PAGE_EMBEDDING_INPUT-1)//2
                delta_clamped = torch.clamp(delta, -max_value_delta, max_value_delta) + max_value_delta

                # En base al valor de delta obtenido se calcula el
                # valor de embedding
                page_emb = self.page_delta_embedding(delta_clamped)
            
            # Una vez calculado el embedding correspondiente a la página,
            # se calcula el embedding correspondiente al bloque. En este
            # caso se hace igual en todos los pasos temporales
            block_emb = self.block_offset_embedding(sequence[:, t, 1])

            # Se almacena en la lista de embeddings los embeddings de página
            # y de bloque obtenidos en este paso temporal para todos los
            # ejemplos de este batch. page_emb es de tamaño (batch_size,ABS_PAGE_EMBEDDING_OUTPUT),
            # y block_emb es de tamaño (batch_size,BLOCK_OFFSET_EMBEDDING_OUTPUT). Al usar dim=-1 se
            # concatenan ambos tensores de embeddings por la última
            # dimensión, dando lugar a un tensor de embeddings de tamaño
            # (batch_size,ABS_PAGE_EMBEDDING_OUTPUT+BLOCK_OFFSET_EMBEDDING_OUTPUT),
            # donde cada fila representa el input del paso temporal actual
            embedded.append(torch.cat([page_emb, block_emb], dim=-1))
        
        # Los tensores de embedding se convierten a un único tensor 3D por
        # la dimensión 1 (la dimensión temporal). De esta forma, se pasa de
        # tener varios tensores (batch_size, ABS_PAGE_EMBEDDING_OUTPUT+BLOCK_OFFSET_EMBEDDING_OUTPUT) a un tensor de
        # tamaño (batch_size, seq_len, ABS_PAGE_EMBEDDING_OUTPUT+BLOCK_OFFSET_EMBEDDING_OUTPUT)
        embedded = torch.stack(embedded, dim=1)
        
        # Se llama a la LSTM con el batch de embeddings. La entrada es el tensor
        # 3D embedded, de tamaño (batch_size, seq_len, ABS_PAGE_EMBEDDING_OUTPUT+BLOCK_OFFSET_EMBEDDING_OUTPUT). El output son dos
        # elementos:
        # · lstm_out es un tensor de tamaño (batch_size, seq_len, hidden_size) que
        #   contiene la salida del estado oculto en cada paso temporal.
        # · _ es una dupla con los estados finales (hidden y cell). En este caso
        #   no hacen falta, así que se descartan.
        lstm_out, _ = self.lstm(embedded)

        # Solo necesitamos el estado oculto del último paso temporal en cada
        # ejemplo del batch:
        # · Con los primeros dos puntos indicamos que queremos los estados
        #   ocultos de todos los ejemplos del batch
        # · Con el -1 indicamos que solo queremos el estado oculto del último
        #   paso temporal, que es el que usaremos para hacer la predicción
        # · Con los dos puntos del final indicamos que queremos todos los
        #   elementos del estado oculto (los hidden_size elementos)
        final_state = lstm_out[:, -1, :]
        
        # Utilizamos la capa fully connected que predice el delta de
        # la siguiente página a predecir. El input es un final_state
        # de tamaño (batch_size, hidden_state) y la salida son unos
        # logits de tamaño (batch_size, DELTA_PAGE_EMBEDDING_INPUT),
        # donde cada entrada es un logit para un posible delta
        page_delta_pred = self.fc_page_delta(final_state)

        # Lo mismo con la capa fully connected que predice el offset
        # de bloque. El tamaño de input es el mismo: (batch_size, hidden_state),
        # pero en este caso el tamaño de output es (batch_size, BLOCK_OFFSET_EMBEDDING_INPUT)
        block_pred = self.fc_block(final_state)
        
        # Se devuelven los logits de los deltas de página predichos
        # y de los offsets de bloque predichos
        return page_delta_pred, block_pred

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
    # Instead of using hard labels and tell the prefetcher to guess
    # the k next accesses, this allows the prefetcher to guess any of
    # many next accesses, where the closest ones are more valuable
    TOLERANCE_SIZE = 20

    # The number of memory accesses used as input for the LSTM
    HISTORY_SIZE = 4

    # The number of memory acesses to skip when prefetching. This
    # help prevent early prefetches
    LOOKAHEAD_SIZE = 5

    # The number of training epochs for the LSTM
    NUM_EPOCHS = 30

    # The default size for a batch
    BATCH_SIZE = 256

    # The learning rate for the LSTM
    LEARNING_RATE = 0.002

    def __init__(self):
        """
            Constructor for the LSTMBasedPrefetcher class
        """
        self.model = LSTM()

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
        except Exception as e:
            print(f"Script failed: {e}")
            print("Trying to save using trace instead")
            
            # Option B: if torch.jit.script fails, we will try to use
            # torch.jit.trace. This function needs an example input, so
            # we will have to cook some inputs. For simplicity, we will
            # assume a batch size of 1

            # The example_pages tensor will be a (1, HISTORY_SIZE) tensor
            # where each element represents the page address of an access in
            # the sequence
            example_pages = torch.randint(0, 10000, (1, self.HISTORY_SIZE))

            # The example_blocks tensor will be a (1, HISTORY_SIZE) tensor
            # where each element represents the block offset of an access
            # in the sequence
            example_blocks = torch.randint(0, 32, (1, self.HISTORY_SIZE))

            # If cuda is available, the tensors are moved to the GPU 
            if torch.cuda.is_available():
                example_pages = example_pages.cuda()
                example_blocks = example_blocks.cuda()

            # These two tensors are stacked into one tensor of size (1, history, 2)
            example_input = torch.stack([example_pages, example_blocks], dim=-1)
            
            # The input is used to save the model via the trace function
            traced = torch.jit.trace(self.model, example_input)
            traced.save(path)

    def batch(self, data:list, batch_size:int=None):
        """
            This function batches the memory acces data for the
            LSTM model.

        Args:
            data (list):                List of memory accesses, where each memory access
                                        is another list with this structure:
                                        [instr_id, cycle, load_address, ip, cache_hit]
                         
            batch_size (int, optional): The size of the batch. If None, the
                                        size is set to self.BATCH_SIZE

        Yields:
            _type_: _description_
        """

        # The default batch size is set if no batch size
        # was specified as a parameter
        if batch_size is None:
            batch_size = self.BATCH_SIZE

        # Here we will store the memory accesses
        bucket_data = defaultdict(list)
        
        # In these lists we will accumulate data for each batch. When
        # a batch is formed, these lists will be cleared

        # This list contains input sequences of size (HISTORY_SIZE, 2)
        # containing [page address, block offset] for each memory access
        # in the sequence
        batch_sequences = []

        # This list contains target accesses to learn. Their shape is (2,)
        # because they just contain [page delta, block offset] for each
        # input example  
        batch_targets = []

        # This list contains data about all accesses in the window for each 
        # input example. This info will be useful when calculating the
        # accuracy
        batch_whole_windows = []
        
        # Total window size: history + lookahead + 1 target block
        window_size = self.HISTORY_SIZE + self.LOOKAHEAD_SIZE + 1
        
        # Each line (each mem access) is read
        for line in data:
            # Each mem access contains five pieces of data
            instr_id, cycle, load_address, ip, cache_hit = line
            
            # The page address is calculated. 4096-byte pages are assumed
            page = load_address >> 12

            # The block offset is calculated. 128-byte blocks are assumed. Since
            # each 4096-byte page contains 32 128-byte blocks (2^5), only the five
            # least significant bits of the block address are selected
            block_offset = (load_address >> 7) & 0x1F
            
            # The bucket key is the IP by default
            bucket_key = ip
            
            # The buffer for this IP is stored in a variable to
            # avoid getting too verbose
            bucket_buffer = bucket_data[bucket_key]
            
            # The current access is appended as [page, block_offset]
            bucket_buffer.append([page, block_offset])
            
            # If we have enough memory accesses to create a training example,
            # we create it. This is necessary because there are not enough
            # memory accesses at the beginning, but once many memory accesses
            # are recorded it is possible to create a training example with
            # the proper size
            if len(bucket_buffer) >= window_size:
                # The history sequence consists of the first self.HISTORY_SIZE
                # memory accesses
                history_sequence = bucket_buffer[:self.HISTORY_SIZE]
                
                # The target access (the access that must be predicted) is the
                # access located self.HISTORY_SIZE + self.LOOKAHEAD_SIZE accesses
                # away from the first access
                target_access = bucket_buffer[self.HISTORY_SIZE + self.LOOKAHEAD_SIZE]

                # The LSTM doesn't predict page addresses: it predicts page deltas.
                # Due to this, it's necessary to calculate the delta for the target
                last_history_page = history_sequence[-1][0]
                target_page = target_access[0]
                target_block = target_access[1]
                target_delta = target_page - last_history_page

                # Since the LSTM can't predict all possible deltas, we have to clamp
                # the delta to a correct value
                max_value_delta = (DELTA_PAGE_EMBEDDING_INPUT-1)//2
                target_delta_clamped = max(-max_value_delta, min(max_value_delta, target_delta))

                # Also, the LSTM can't predict negativa values. It only predicts
                # indices, so it's necessary to transform this clamped delta
                # into an index between 0 and DELTA_PAGE_EMBEDDING_INPUT
                target_delta_clamped_index = target_delta_clamped + max_value_delta
                
                # The history and the target are added to the sequence
                batch_sequences.append(history_sequence)
                batch_targets.append([target_delta_clamped_index, target_block])
                
                # The whole window is recorded for accuracy calculation
                batch_whole_windows.append(bucket_buffer[:window_size])
                
                # After we finish with this window, we slide it: the oldest
                # access is removed
                bucket_buffer.pop(0)
            
            # If we have accumulated a full batch, we yield it (this makes
            # this function a generator, since it only yields one batch
            # at a time, instead of returning all batches at once)
            if len(batch_sequences) == batch_size:
                # It's necessary to convert the info to tensors

                # batch_sequences: list of shape (batch_size, HISTORY_SIZE, 2)
                sequences_tensor = torch.LongTensor(batch_sequences)

                # batch_targets: list of shape (batch_size, 2)
                targets_tensor = torch.LongTensor(batch_targets)
                
                # If CUDA is avaulable, the tensors are sent to the GPU
                # for faster computation
                if torch.cuda.is_available():
                    sequences_tensor = sequences_tensor.cuda()
                    targets_tensor = targets_tensor.cuda()
                
                # This batch is yielded
                yield (
                    # Input sequences: (batch_size, HISTORY_SIZE, 2)
                    sequences_tensor,

                    # Target accesses: (batch_size, 2)
                    targets_tensor,

                    # Info about the whole window for each access
                    # (used for accuracy calculation)
                    batch_whole_windows
                )
                
                # The info about this batch is cleared, the next
                # batch will be prepared using the next memory accesses
                batch_sequences = []
                batch_targets = []
                batch_whole_windows = []
        
        # If there are no more memory accesses left and there is a batch
        # being prepared, it is returned (altough it won't be a whole
        # batch)
        if len(batch_sequences) > 0:
            # The same procedure is followed
            sequences_tensor = torch.LongTensor(batch_sequences)
            targets_tensor = torch.LongTensor(batch_targets)
            
            if torch.cuda.is_available():
                sequences_tensor = sequences_tensor.cuda()
                targets_tensor = targets_tensor.cuda()
            
            yield (
                sequences_tensor,
                targets_tensor,
                batch_whole_windows
            )
    
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
        print(f'Called train_and_test method.')

        print(f'HISTORY_SIZE = {self.HISTORY_SIZE}')
        print(f'LOOKAHEAD_SIZE = {self.LOOKAHEAD_SIZE}')

        # If the model's loss during test doesn't improve for 3 epochs,
        # training ends to avoid overfitting
        patience = 3

        # This indicates the best (lowest) loss found during test
        best_test_loss = float("inf")

        # This indicates for how many epochs the model hasn't improved
        # its loss during test
        epochs_without_improvement = 0

        # The optimizer used to update the model's parameters
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)

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
        for epoch in range(self.NUM_EPOCHS):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.NUM_EPOCHS}")
            print(f"{'='*60}")
            
            # ============================================================
            # TRAINING PHASE
            # ============================================================

            # The model is set to training mode to allow its parameters to be updated
            self.model.train()
            
            # The training accuracies and losses will be stored here
            train_strict_accs = []
            train_tolerant_accs = []
            train_losses = []

            # This is a helper variable that is used to show the training progress
            train_percent = len(train_data) // self.BATCH_SIZE // 100

            print("Training...")
            for i, (sequences, targets, whole_windows) in enumerate(self.batch(train_data)):
                # sequences is a tensor of shape (batch_size, HISTORY_SIZE, 2)
                # targets is a tensor of shape (batch_size, 2) with [page_delta_index, block]
                
                # The gradients are cleared because we are going
                # to calculate new ones
                optimizer.zero_grad()

                # Both the predicted page deltas and block offsets are obtained
                # by performing a forward pass. The shapes are the following:
                # · page_delta_pred: (batch_size, DELTA_PAGE_EMBEDDING_INPUT)
                # · block_pred: (batch_size, BLOCK_OFFSET_EMBEDDING_INPUT)
                page_delta_pred, block_pred = self.model(sequences)

                # The target page deltas and block offsets are extracted
                # from the targets tensor. Both resulting tensors will have
                # a shape of (batch_size,)
                target_page_deltas = targets[:, 0].long()
                target_blocks = targets[:, 1].long()

                # Once we have the logits (the page_delta_pred and block_pred tensors)
                # and the targets (the target_page_deltas and target_blocks tensors) we
                # can calculate the losses
                loss_page = criterion_page(page_delta_pred, target_page_deltas)
                loss_block = criterion_block(block_pred, target_blocks)
                
                # Both losses are combined into one global loss
                loss_train = loss_page + loss_block

                # Once we have the losses, the accuracy is calculated
                strict_acc, tolerant_acc = self.accuracy(
                    page_delta_pred, 
                    block_pred, 
                    target_page_deltas, 
                    target_blocks,
                    sequences,
                    whole_windows
                )

                # Now, we can do the backward pass and backpropagation
                loss_train.backward()
                optimizer.step()

                # The obtained metrics are appended to the lists accordingly
                train_losses.append(float(loss_train.item()))
                train_strict_accs.append(float(strict_acc))
                train_tolerant_accs.append(float(tolerant_acc))

                # If we progressed enough, a period is shown on screen
                if train_percent != 0 and i % train_percent == 0:
                    print('.', end='', flush=True)

            print()
            avg_strict = sum(train_strict_accs) / len(train_strict_accs)
            avg_tolerant = sum(train_tolerant_accs) / len(train_tolerant_accs)
            print(f'Training strict accuracy: {avg_strict:.4f}')
            print(f'Training tolerant accuracy: {avg_tolerant:.4f}')
            print(f'Training loss: {sum(train_losses):.4f}')

            # The accuracies are averaged and the losses are added, and both are
            # appended to their corresponding list
            avg_train_strict_accs.append(avg_strict)
            avg_train_tolerant_accs.append(avg_tolerant)
            total_train_loss.append(sum(train_losses))

            # ============================================================
            # TESTING PHASE
            # ============================================================

            # In this case, the model is set to evaluation mode
            self.model.eval()
            
            # The test accuracies and losses will be stored here
            test_strict_accs = []
            test_tolerant_accs = []
            test_losses = []

            # This is a helper variable that is used to show the test progress
            test_percent = len(test_data) // self.BATCH_SIZE // 100

            print("Testing...")

            # We use torch.no_grad() because it's not necessary to calculate
            # gradients during test
            with torch.no_grad():
                for i, (sequences, targets, whole_windows) in enumerate(self.batch(test_data)):

                    # We obtain the logits (forward pass)
                    page_delta_pred, block_pred = self.model(sequences)

                    # We extract the target tensors
                    target_page_deltas = targets[:, 0].long()
                    target_blocks = targets[:, 1].long()

                    # Then, using the logits and the targets we calculate the loss
                    loss_page = criterion_page(page_delta_pred, target_page_deltas)
                    loss_block = criterion_block(block_pred, target_blocks)

                    # Just like we did during training, both losses are added
                    loss_test = loss_page + loss_block

                    # Then, we calculate the accuracy
                    strict_acc, tolerant_acc = self.accuracy(
                        page_delta_pred, 
                        block_pred, 
                        target_page_deltas, 
                        target_blocks,
                        sequences,
                        whole_windows
                    )

                    # The metrics are appended to their lists
                    test_losses.append(float(loss_test.item()))
                    test_strict_accs.append(float(strict_acc))
                    test_tolerant_accs.append(float(tolerant_acc))

                    # The progress indicator is updated
                    if test_percent != 0 and i % test_percent == 0:
                        print('.', end='', flush=True)

            print()
            avg_strict = sum(test_strict_accs) / len(test_strict_accs)
            avg_tolerant = sum(test_tolerant_accs) / len(test_tolerant_accs)
            print(f'Test strict accuracy: {avg_strict:.4f}')
            print(f'Test tolerant accuracy: {avg_tolerant:.4f}')
            print(f'Test loss: {sum(test_losses):.4f}')

            # The accuracies are averaged and the losses are added, and both are
            # appended to their corresponding list
            avg_test_strict_accs.append(avg_strict)
            avg_test_tolerant_accs.append(avg_tolerant)
            total_test_loss.append(sum(test_losses))

            # ============================================================
            # SAVE MODEL CHECKPOINT
            # ============================================================
            if model_name is not None:
                checkpoint_path = f"{model_name}-epoch{epoch+1}.pt"
                self.save(checkpoint_path)
                print(f"Model saved to {checkpoint_path}")

            # ============================================================
            # EARLY STOPPING
            # ============================================================

            # The loss for the current epoch during test is calculated
            current_loss = sum(test_losses)
            
            # If the loss has decreased, then that's alright. We can keep
            # training without worrying about anything
            if current_loss < best_test_loss:
                print(f"Test loss improved from {best_test_loss:.6f} to {current_loss:.6f}")
                best_test_loss = current_loss
                epochs_without_improvement = 0

            # If the loss hasn't decreased, then maybe we should start to
            # worry a bit. Maybe there is some overfitting going on...
            else:
                epochs_without_improvement += 1
                print(f"No improvement in test loss. Patience: {epochs_without_improvement}/{patience}")

            # If the model goes for too many epochs without improving its
            # loss, then there is clearly overfitting, and the training is
            # stopped
            if epochs_without_improvement >= patience:
                print("\nEarly stopping triggered")
                break

        # ============================================================
        # PLOT RESULTS
        # ============================================================
        if graph_name is not None:
            self._plot_training_results(
                avg_train_strict_accs,
                avg_train_tolerant_accs,
                avg_test_strict_accs,
                avg_test_tolerant_accs,
                total_train_loss,
                total_test_loss,
                graph_name
            )

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
    
    def accuracy(self, page_delta_pred:torch.LongTensor, block_pred:torch.LongTensor, target_page_deltas:torch.LongTensor, target_blocks:torch.LongTensor, sequences:torch.LongTensor, whole_windows:list):
        """
            This method calculates the model's accuracy
        
        Args:
            page_delta_pred(torch.LongTensor):      Logits for page deltas. Size: (batch_size, 4097).
            block_pred(torch.LongTensor):           Logits for block offsets. Size: (batch_size, 32).
            target_page_deltas(torch.LongTensor):   Ground truth page delta indices. Size: (batch_size,).
            target_blocks(torch.LongTensor):        Ground truth block offsets. Size: (batch_size,).
            sequences(torch.LongTensor):            Input sequences for the model. Size: (batch_size, HISTORY_SIZE, 2)
            whole_windows(list):                    List of lists with complete windows for each example in the batch.
        
        Returns:
            tuple: (strict_accuracy, tolerant_accuracy), where:
                -   strict_accuracy tells the accuracy when predicting the
                    acces that is located exactly LOOKAHEAD_SIZE after the
                    last history access
                -   tolerant_accuracy tells the accuracy when predicting any
                    of the accesses between the last history access and the
                    access that is located LOOKAHEAD_SIZE+TOLERANCE_SIZE ahead
        """
        # First, we get the batch size (since a batch can be incomplete if the
        # number of examples is not a multiple of the batch size, it's necessary
        # to check this instead of assuming a fixed size)
        batch_size = page_delta_pred.shape[0]
        
        # We obtain the predicted indices for both the page delta and the block
        # offset for each example in the batch. This is done by using argmax, since
        # page_delta_pred and block_pred contain logits. The resulting tensors
        # have a shape of (batch_size,)
        pred_page_delta_indices = torch.argmax(page_delta_pred, dim=1)
        pred_block_offsets = torch.argmax(block_pred, dim=1)
        
        # The page delta indices need to be transformed into actual
        # deltas, since the indices go from 0 to DELTA_PAGE_EMBEDDING_INPUT.
        # This means that, if the indices go from 0 to 4097, the deltas must
        # go from -2048 to 2048 (index 0 is delta -2048, index 1 is delta -2047...)
        max_value_delta = (DELTA_PAGE_EMBEDDING_INPUT-1)//2
        pred_page_deltas = pred_page_delta_indices - max_value_delta
        
        # These counters will help us track the amount of correct prefetches
        strict_correct = 0
        tolerant_correct = 0
        
        # Then, we iterate through each access in this batch
        for i in range(batch_size):
            # First, we extract the last page from the history
            last_history_page = sequences[i, -1, 0].item()
            
            # Then, we obtain the predicted page given the last history page
            # and the predicted delta
            pred_delta = pred_page_deltas[i].item()
            pred_page = last_history_page + pred_delta

            # We also obtain the predicted block offset
            pred_block = pred_block_offsets[i].item()
            
            # Now, we obtain the whole window for this example, which is
            # a list of [page, block] pairs
            window = whole_windows[i]
            
            # ============================================================
            # STRICT ACCURACY: Check exact match with target
            # ============================================================

            # The target is at position HISTORY_SIZE + LOOKAHEAD_SIZE
            target_idx = self.HISTORY_SIZE + self.LOOKAHEAD_SIZE
            
            # The target should be inside the window
            if target_idx < len(window):
                # We check the window to obtain the page address and the
                # block offset of the correct access
                target_page_abs = window[target_idx][0]
                target_block_abs = window[target_idx][1]
                
                # Then, we check if the predicted page and block offset are
                # the same as the real ones
                if pred_page == target_page_abs and pred_block == target_block_abs:
                    strict_correct += 1
            
            # ============================================================
            # TOLERANT ACCURACY: Check match within tolerance window
            # ============================================================

            # Tolerance window spans from HISTORY_SIZE to
            # HISTORY_SIZE + LOOKAHEAD_SIZE + TOLERANCE_SIZE
            tolerance_start = self.HISTORY_SIZE
            tolerance_end = self.HISTORY_SIZE + self.LOOKAHEAD_SIZE + self.TOLERANCE_SIZE
            
            # The end of the tolerance window is set to the end of the window
            # if there are not enough accesses
            tolerance_end = min(tolerance_end, len(window))
            
            # Now, we check if the prediction matches any access in the tolerance window
            match_found = False

            # We iterate through each access in the tolerance window
            for j in range(tolerance_start, tolerance_end):
                # We obtain the page and the block of the current access
                window_page = window[j][0]
                window_block = window[j][1]
                
                # If the access matches the prediction, we can stop iterating
                if pred_page == window_page and pred_block == window_block:
                    match_found = True
                    break
            
            # If we found a matching access, the counter is updated
            if match_found:
                tolerant_correct += 1
        
        # Finally, we calculate accuracies by dividing the counters by the batch size.
        # The result will be 1 if all predictions were correct and 0 if all predictions
        # were incorrect
        strict_accuracy = strict_correct / batch_size
        tolerant_accuracy = tolerant_correct / batch_size
        
        return strict_accuracy, tolerant_accuracy

Model = LSTMBasedPrefetcher