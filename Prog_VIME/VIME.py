"""VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain (VIME) Codebase.
Reference: Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar, 
"VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain," 
Neural Information Processing Systems (NeurIPS), 2020.
Paper link: TBD
Last updated Date: October 11th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------
vime_semi.py
- Semi-supervised learning parts of the VIME framework
- Using both labeled and unlabeled data to train the predictor with the help of trained encoder
"""

# Necessary packages
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from sklearn.preprocessing import OneHotEncoder
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.contrib import layers as contrib_layers
from tensorflow.keras import layers as contrib_layers
# from vime_utils import mask_generator, pretext_generator
from sklearn.metrics import accuracy_score, roc_auc_score
import tensorflow.keras.backend as K
#============================= UTILS ==========================
def mask_generator (p_m, x):
  """Generate mask vector.
  
  Args:
    - p_m: corruption probability
    - x: feature matrix
    
  Returns:
    - mask: binary mask matrix 
  """
  mask = np.random.binomial(1, p_m, x.shape)
  return mask

def pretext_generator (m, x):
  """Generate corrupted samples.
  
  Args:
    m: mask matrix
    x: feature matrix
    
  Returns:
    m_new: final mask matrix after corruption
    x_tilde: corrupted feature matrix
  """
  
  # Parameters
  no, dim = x.shape  
  # Randomly (and column-wise) shuffle data
  x_bar = np.zeros([no, dim])
  for i in range(dim):
    idx = np.random.permutation(no)
    x_bar[:, i] = x[idx, i]
    
  # Corrupt samples
  x_tilde = x * (1-m) + x_bar * m  
  # Define new mask matrix
  m_new = 1 * (x != x_tilde)

  return m_new, x_tilde

def convert_vector_to_matrix(vector):
  """Convert one dimensional vector into two dimensional matrix
  
  Args:
    - vector: one dimensional vector
    
  Returns:
    - matrix: two dimensional matrix
  """
  # Parameters
  no = len(vector)
  dim = len(np.unique(vector))
  # Define output
  matrix = np.zeros([no,dim])
  
  # Convert vector to matrix
  for i in range(dim):
    idx = np.where(vector == i)
    matrix[idx, i] = 1
    
  return matrix

def perf_metric (metric, y_test, y_test_hat):
  """Evaluate performance.
  
  Args:
    - metric: acc or auc
    - y_test: ground truth label
    - y_test_hat: predicted values
    
  Returns:
    - performance: Accuracy or AUROC performance
  """
  # Accuracy metric
  if metric == 'acc':
    if len(y_test.shape) == 1:
        y_test = convert_vector_to_matrix(y_test)

    result = accuracy_score(np.argmax(y_test, axis = 1), 
                            np.argmax(y_test_hat, axis = 1))
  # AUROC metric
  elif metric == 'auc':
    result = roc_auc_score(y_test[:, 1], y_test_hat[:, 1])      
    
  return result
#===================== VIME-SEMI ==============================
def vime_semi(x_train, y_train, x_unlab, x_test, parameters, 
              p_m, K, beta, file_name, total_train):
  """Semi-supervied learning part in VIME.
  
  Args:
    - x_train, y_train: training dataset
    - x_unlab: unlabeled dataset
    - x_test: testing features
    - parameters: network parameters (hidden_dim, batch_size, iterations)
    - p_m: corruption probability
    - K: number of augmented samples
    - beta: hyperparameter to control supervised and unsupervised loss
    - file_name: saved filed name for the encoder function
    
  Returns:
    - y_test_hat: prediction on x_test
  """
  
  # Network parameters
  hidden_dim = parameters['hidden_dim']
  act_fn = tf.nn.relu
  batch_size = parameters['batch_size']
  iterations = parameters['iterations']

  # Basic parameters
  # y_train = tf.one_hot(y_train,len(y_train.unique())).numpy()
  #y_train = y_train#.numpy()
  #onehot = np.zeros((y_train.size, y_train.max()+1))
  #onehot[np.arange(y_train.size), y_train] = 1
  #y_train=onehot
  print("y_train.shape: ", y_train.shape)
  data_dim = 26 #len(x_train[0, :])
  label_dim = len(y_train[0,:])
  
  # Divide training and validation sets (9:1)
  idx = np.random.permutation(len(x_train[:, 0]))
  train_idx = idx[:int(len(idx)*0.9)]
  valid_idx = idx[int(len(idx)*0.9):]
  
  x_valid = x_train[valid_idx, :]
  
  y_valid = y_train[valid_idx,:]
  
  # total_train = np.copy(x_train)
  x_train = x_train[train_idx, :]
  y_train = y_train[train_idx, :]  
  
  # Input placeholder
  # Labeled data
  x_input = tf.placeholder(tf.float32, [None, data_dim])
  y_input = tf.placeholder(tf.float32, [None, label_dim])
  
  # Augmented unlabeled data
  xu_input = tf.placeholder(tf.float32, [None, None, data_dim])
  
  ## Predictor
  def predictor(x_input):
    """Returns prediction.
    
    Args: 
      - x_input: input feature
      
    Returns:
      - y_hat_logit: logit prediction
      - y_hat: prediction
    """
    with tf.variable_scope('predictor', reuse=tf.AUTO_REUSE):     
      # Stacks multi-layered perceptron
      # inter_layer = contrib_layers.fully_connected(x_input, 
      #                                              hidden_dim, 
      #                                              activation_fn=act_fn)
      # inter_layer = contrib_layers.fully_connected(inter_layer, 
      #                                              hidden_dim, 
      #                                              activation_fn=act_fn)

      # y_hat_logit = contrib_layers.fully_connected(inter_layer, 
      #                                              label_dim, 
      #                                              activation_fn=None)
      # tf.compat.v1.layers.dense(inputs=x_input,units=hidden_dim,activation=act_fn)
      # inp = contrib_layers.InputLayer(x_input)
      inter_layer = tf.compat.v1.layers.dense(x_input, 
                                                   256, 
                                                   activation=act_fn)
      inter_layer = tf.compat.v1.layers.dense(inter_layer, 
                                                   512, 
                                                   activation=act_fn)

      y_hat_logit = tf.compat.v1.layers.dense(inter_layer, 
                                                   label_dim, 
                                                   activation=None)
      y_hat = tf.nn.softmax(y_hat_logit)

    return y_hat_logit, y_hat

  # Build model
  y_hat_logit, y_hat = predictor(x_input)    
  yv_hat_logit, yv_hat = predictor(xu_input)
  
  # Define losses
  # Supervised loss
  y_loss = tf.losses.softmax_cross_entropy(y_input, y_hat_logit)  
  # Unsupervised loss
  yu_loss = tf.reduce_mean(tf.nn.moments(yv_hat_logit, axes = 0)[1])
  
  # Define variables
  p_vars = [v for v in tf.trainable_variables() \
            if v.name.startswith('predictor')]    
  # Define solver
  solver = tf.train.AdamOptimizer().minimize(y_loss + \
                                 beta * yu_loss, var_list=p_vars)

  # Load encoder from self-supervised model
  encoder = keras.models.load_model(file_name)
  
  # Encode validation and testing features
  x_valid = encoder.predict(x_valid)  
  x_test = encoder.predict(x_test)

  # Start session
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  
  # Setup early stopping procedure
  class_file_name = './save_model/class_model.ckpt'
  saver = tf.train.Saver(p_vars)
    
  yv_loss_min = 1e10
  yv_loss_min_idx = -1
  
  # Training iteration loop
  for it in range(iterations):

    # Select a batch of labeled data
    batch_idx = np.random.permutation(len(x_train[:, 0]))[:batch_size]
    x_batch = x_train[batch_idx, :]
    y_batch = y_train[batch_idx, :]    
    
    # Encode labeled data
    x_batch = encoder.predict(x_batch)  
    
    # Select a batch of unlabeled data
    batch_u_idx = np.random.permutation(len(x_unlab[:, 0]))[:batch_size]
    xu_batch_ori = x_unlab[batch_u_idx, :]
    
    # Augment unlabeled data
    xu_batch = list()
    
    for rep in range(K):      
      # Mask vector generation
      m_batch = mask_generator(p_m, xu_batch_ori)
      # Pretext generator
      _, xu_batch_temp = pretext_generator(m_batch, xu_batch_ori)
      
      # Encode corrupted samples
      xu_batch_temp = encoder.predict(xu_batch_temp)
      xu_batch = xu_batch + [xu_batch_temp]
    # Convert list to matrix
    xu_batch = np.asarray(xu_batch)

    # Train the model
    _, y_loss_curr = sess.run([solver, y_loss], 
                              feed_dict={x_input: x_batch, y_input: y_batch, 
                                         xu_input: xu_batch})  
    # Current validation loss
    yv_loss_curr = sess.run(y_loss, feed_dict={x_input: x_valid, 
                                               y_input: y_valid})
  
    if it % 100 == 0:
      print('Iteration: ' + str(it) + '/' + str(iterations) + 
            ', Current loss: ' + str(np.round(yv_loss_curr, 4)))      
      
    # Early stopping & Best model save
    if yv_loss_min > yv_loss_curr:
      yv_loss_min = yv_loss_curr
      yv_loss_min_idx = it

      # Saves trained model
      saver.save(sess, class_file_name)
      
    if yv_loss_min_idx + 100 < it:
      break

  #%% Restores the saved model
  imported_graph = tf.train.import_meta_graph(class_file_name + '.meta')
  
  sess = tf.Session()
  imported_graph.restore(sess, class_file_name)
    
  # Predict on x_test
  y_test_hat = sess.run(y_hat, feed_dict={x_input: x_test})
    
  #=======================================================================
  # import pdb; pdb.set_trace()
  # predict on all data
  
  all_batch = list()
  for i in range(int(len(total_train[:,0])/batch_size)+1):
    if len(total_train[i*batch_size:])>=batch_size:
        batch_x = total_train[i*batch_size:(i+1)*batch_size]
    else:  
        batch_x = total_train[i*batch_size:]
        if len(batch_x)==0:
          break      
    xtmp = encoder.predict(batch_x)
    all_batch = all_batch + [xtmp]

  # all_batch = np.asarray(all_batch)

#   un_batch = list()
  # for i in range(int(len(x_unlab[:,0])/batch_size)+1):
  #   if len(x_unlab[i*batch_size:])>=batch_size:
  #       batch_x = x_unlab[i*batch_size:(i+1)*batch_size]
  #   else:
  #       batch_x = x_unlab[i*batch_size:]
  #       if len(batch_x)==0:
  #         break
  #   xtmp = encoder.predict(batch_x)
  #   all_batch = all_batch + [xtmp]

  all_batch = np.asarray(np.vstack(all_batch))
  #=======================================================================
  #import pdb; pdb.set_trace()
  pseudoLabels = sess.run(y_hat, feed_dict={x_input: all_batch})
  
  return y_test_hat,pseudoLabels


def vime_self (x_unlab, p_m, alpha, parameters):
  """Self-supervised learning part in VIME.
  
  Args:
    x_unlab: unlabeled feature
    p_m: corruption probability
    alpha: hyper-parameter to control the weights of feature and mask losses
    parameters: epochs, batch_size
    
  Returns:
    encoder: Representation learning block
  """
    
  # Parameters
  _, dim = x_unlab.shape
  epochs = parameters['epochs']
  batch_size = parameters['batch_size']
  
  # Build model  
  
  inputs = contrib_layers.Input(shape=(dim,))
  # Encoder  
  h = contrib_layers.Dense(int(256), activation='relu', name='encoder1')(inputs)  
  h = contrib_layers.Dense(int(128), activation='relu', name='encoder2')(h)  
  h = contrib_layers.Dense(int(26), activation='relu', name='encoder3')(h)  
  # Mask estimator
  output_1 = contrib_layers.Dense(dim, activation='sigmoid', name = 'mask')(h)  
  # Feature estimator
  output_2 = contrib_layers.Dense(dim, activation='sigmoid', name = 'feature')(h)
  #Projection Network
  
  
  model = Model(inputs = inputs, outputs = [output_1, output_2])
  
  model.compile(optimizer='rmsprop',
                loss={'mask': 'binary_crossentropy', 
                      'feature': 'mean_squared_error'},
                loss_weights={'mask':1, 'feature':alpha})
  
  # Generate corrupted samples
  m_unlab = mask_generator(p_m, x_unlab)
  m_label, x_tilde = pretext_generator(m_unlab, x_unlab)
  
  # Fit model on unlabeled data
  model.fit(x_tilde, {'mask': m_label, 'feature': x_unlab}, epochs = epochs, batch_size= batch_size)
      
  # Extract encoder part
  layer_name = model.layers[3].name
  layer_output = model.get_layer(layer_name).output #model.layers[1:4]
  encoder = models.Model(inputs=model.input, outputs=layer_output)
  
  return encoder

def mlp(x_train, y_train, x_test, parameters):
  """Multi-layer perceptron (MLP).
  
  Args: 
    - x_train, y_train: training dataset
    - x_test: testing feature
    - parameters: hidden_dim, epochs, activation, batch_size
    
  Returns:
    - y_test_hat: predicted values for x_test
  """  
  
  # Convert labels into proper format
  if len(y_train.shape) == 1:
    y_train = convert_vector_to_matrix(y_train)
    
  # Divide training and validation sets (9:1)
  idx = np.random.permutation(len(x_train[:, 0]))
  train_idx = idx[:int(len(idx)*0.9)]
  valid_idx = idx[int(len(idx)*0.9):]
  
  # Validation set
  x_valid = x_train[valid_idx, :]
  y_valid = y_train[valid_idx, :]
  
  # Training set
  x_train = x_train[train_idx, :]
  y_train = y_train[train_idx, :]  
  
  # Reset the graph
  K.clear_session()
    
  # Define network parameters
  hidden_dim = parameters['hidden_dim']
  epochs_size = parameters['epochs']
  act_fn = parameters['activation']
  batch_size = parameters['batch_size']
  
  # Define basic parameters
  data_dim = len(x_train[0, :])
  label_dim = len(y_train[0, :])

  # Build model
  model = Sequential()
  model.add(Dense(256, input_dim = data_dim, activation = act_fn))
  model.add(Dense(128, activation = act_fn))  
  model.add(Dense(label_dim, activation = 'softmax'))
  
  model.compile(loss = 'categorical_crossentropy', optimizer='sgd', 
                metrics = ['acc'])
  
  es = EarlyStopping(monitor='val_loss', mode = 'min', 
                     verbose = 1, restore_best_weights=True, patience=50)
  
  # Fit model on training dataset
  model.fit(x_train, y_train, validation_data = (x_valid, y_valid), 
            epochs = epochs_size, batch_size = batch_size, 
            verbose = 0, callbacks=[es])
  
  # Predict on x_test
  y_test_hat = model.predict(x_test)
  
  return y_test_hat