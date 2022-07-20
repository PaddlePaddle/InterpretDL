import paddle
import numpy as np
import os, sys

from tqdm import tqdm
from interpretdl.common.file_utils import download_and_decompress

from .abc_interpreter import Interpreter

class TrainingDynamics():
    """
    Training Dynamics Interpreter focus the behavior of each training sample by 
    running a normal SGD training process.
    
    By recording the training dynamics, interpreter can diagnose dataset with 
    hand-designed features or by learning solution.
    
    After interpretation on the level of data, we can handle better the datasets 
    with underlying label noises, thus can achieve a better performance on it.
    
    More training dynamics based methods including [Forgetting Events] 
    and [Dataset Mapping] will be available in this interpreter.
    """

    def __init__(self, paddle_model: callable, device: str = 'gpu:0', use_cuda=None):
        """
        
        Args:
            paddle_model (callable): A model with :py:func:`forward` and possibly :py:func:`backward` functions.
            device (str): The device used for running ``paddle_model``, options: ``"cpu"``, ``"gpu:0"``, ``"gpu:1"`` 
                etc.
        """
        Interpreter.__init__(self, paddle_model, device, use_cuda)

    def generator(self,
                  train_loader: callable,
                  optimizer: paddle.optimizer,
                  epochs: int):

        """Run the training process and record the forgetting events statistics.

        Args:
            train_loader (callable): A training data generator.
            optimizer (paddle.optimizer): The paddle optimizer.
            epochs (int): The number of epochs to train the model.

        Returns:
            training_dynamics (dict): A pointwise training dynamics(history) for each epoch.
        """

        training_dynamics = {}

        paddle.set_device(self.device)

        for i in range(epochs):
            counter = 0
            correct = 0
            total = 0
            for step_id, (indices,x_train,y_train) in enumerate(train_loader()):

                if not isinstance(x_train[0], np.ndarray):
                    x_train = x_train.numpy()
                    
                x_train = paddle.to_tensor(x_train)
                y_train = paddle.to_tensor(np.array(y_train).reshape((-1, 1)))
                logits = self.paddle_model(x_train)
                predicted = paddle.argmax(logits, axis=1).numpy()
                bsz = len(predicted)

                loss = paddle.nn.functional.softmax_with_cross_entropy(logits, y_train)
                avg_loss = paddle.mean(loss)
                y_train = y_train.reshape((bsz, )).numpy()

                acc = (predicted == y_train).astype(int)
                avg_loss.backward()
                optimizer.step()
                optimizer.clear_grad()

                correct += np.sum(acc)
                total += bsz
                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d]\t\tLoss: %.4f Acc@1: %.3f%%' %
                                 (i + 1, epochs, step_id + 1, avg_loss.numpy().item(), 100. * correct / total))
                sys.stdout.flush()
                
                #record training dynamics information 
                with paddle.no_grad():
                    softmax = paddle.nn.Softmax()
                    training_dynamics_per_epoch = softmax(logits).detach().cpu().numpy()

                    if len(indices) == train_loader.batch_size:
                        for j,index in enumerate(indices):
                            index = index.item()
                            training_dynamics_previous = training_dynamics.get(index,[])
                            training_dynamics_previous.append(training_dynamics_per_epoch[j])
                            training_dynamics[index] = training_dynamics_previous
                    else:
                        for j,index in enumerate(indices):
                            index = index.item()
                            training_dynamics_previous = training_dynamics.get(index,[])
                            training_dynamics_previous.append(np.full([training_dynamics_per_epoch.shape[1],], np.nan))
                            training_dynamics[index] = training_dynamics_previous                       
                
        return training_dynamics
            
    def transform(self,logits,assigned_targets):
        """Transform training dynamics with linear interpolation.

        Args:
            logits (dict): A dictionary recording training dynamics.
            assigned_targets (list): The assigned targets of dataset,.

        """
        logits = [(k, logits[k]) for k in sorted(logits.keys())]
        logits = np.asarray([logits[i][1] for i in range(len(logits))])

        # Linear interpolation of logits given
        for logit in logits:
            bad_indexes = np.isnan(logit)
            good_indexes = np.logical_not(bad_indexes)
            interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], logit[good_indexes])
            logit[bad_indexes] = interpolated
        logits = logits.astype(np.float16)

        targets_list = np.argsort(-logits.mean(axis=1), axis=1)
        self.training_dynamics = np.ones_like(logits,dtype=np.float16)
        self.labels = np.ones_like(logits[:,0,:],dtype=np.int16)
        
        for index,targets in enumerate(tqdm(targets_list,desc=f"Save")):
        # save ground turth td
            self.labels[index,0] = assigned_targets[index]
            self.training_dynamics[index,:,0] = logits[index,:,assigned_targets[index]].tolist()
        # save topk td
            top_i=1
            for target in targets:
                if target != assigned_targets[index]:
                    self.labels[index,top_i] = target
                    self.training_dynamics[index,:,top_i] = logits[index,:,target].tolist()
                    top_i+=1

    def save(self,logits,assigned_targets,label_flip=None,save_path=None):
        """Save transformed training dynamics .

        Args:
            save_path (_type_, optional): The filepath to save the processed image. 
                If None, the image will not be saved. Default: None
        """
        if save_path is None:
            save_path = 'assets'
        if not os.path.exists(save_path):
            os.makedirs(save_path)    
        
        self.transform(logits=logits,assigned_targets=assigned_targets)
        np.savez_compressed(os.path.join(save_path, 'training_dynamics.npz'),
                            **{'label_flip': label_flip, 'labels': self.labels, 'td': self.training_dynamics})
        return self

    
class LSTM(paddle.nn.Layer):
    def __init__(self,input_size=1,hidden_size=64,num_layers=2):
        super(LSTM, self).__init__()
        
        # maybe need initialisation
        self.classifier = paddle.nn.Linear(in_features=hidden_size,out_features=2)
        self.softmax = paddle.nn.Softmax()
        self.lstm = paddle.nn.LSTM(input_size=input_size, 
                                   hidden_size=hidden_size, 
                                   num_layers=num_layers,
                                   time_major=False,)
    def forward(self, x):
        if len(x.shape) !=3:
            x = paddle.unsqueeze(x,axis=2)
        out, (_, _) = self.lstm(x)
        out = self.classifier(out[:,-1,:])
        out = self.softmax(out)
        return out
    
class BHDFInterpreter():

    """
        [Beyond hand-designed Feature Interpreter]

        Representation learning from training dynamics leads to a LSTM-instanced noise detector,
        which is transferable to different datasets.
    """

    def __init__(self, detector: callable = None, device: str = 'gpu:0', use_cuda=None):
        """
        Args:
            detector (callable, optional): A detector model for identifying the mislabeled samples. Defaults to None.
            device (str, optional): The device used for running ``detector``, options: ``"cpu"``, ``"gpu:0"``, 
                ``"gpu:1"`` etc. Defaults to 'gpu:0'.
        """
        
        paddle.set_device(device)
                
        if detector is not None:
            self.detector = detector
        else:
            self.detector = LSTM()
            download_and_decompress(url="https://github.com/PaddlePaddle/InterpretDL/files/9120427/noise_detector_trained.pdparams.zip",
                    path="assets/")
            paddle.Model(self.detector).load("assets/noise_detector_trained")

    def interpret(self,
                  training_dynamics=None,
                  training_dynamics_path="assets/training_dynamics.npz"):
        """Call this function to rank samples' correctness.

        Args:
            training_dynamics (dict, optional): Training dynamics is a dictionary, which has keys as follows: 
            {
            `label_flip`: list: The position of label contamination where True indicates label noise;
            `labels`: numpy.ndarray: with shape of length of dataset * class number, generated by TrainingDynamics.generator;
            `td`: numpy.ndarray: with shape of length of dataset * training epochs * class number, point-wise probability for each epoch, generated by TrainingDynamics.generator
            }
            
        Returns:
            (numpy.ndarray, list): (order,predictions) where order is {ranking of label correctness 
            in form of data indices list} and predictions is {point-wise predictions as clean}.
        """

        if training_dynamics is not None:
            training_dynamics = paddle.to_tensor(training_dynamics['td'][:,:,0]).astype(paddle.float32)
        elif training_dynamics_path is not None:
            training_dynamics = paddle.to_tensor(np.load(training_dynamics_path)['td'][:,:,0]).astype(paddle.float32)
        else:
            raise Exception('Invalid form or path')

        dataset = paddle.io.TensorDataset([training_dynamics,paddle.zeros((len(training_dynamics),))])
        loader = paddle.io.DataLoader(dataset, batch_size=128, shuffle=False)        

        predictions=[]
        with paddle.no_grad():
            for batch_id, data in enumerate(loader()):
                x_data = data[0]
                predicts = self.detector(x_data).cpu().detach().numpy()
                predictions.extend(predicts[:,1])
        
        order = np.argsort(predictions)
        
        return order,predictions
