import tensorflow as tf
import pandas as pd
import matplotlib as plt
import numpy as np
import HyperParam as hp
import datetime
import tqdm

from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import Sequential,layers,initializers,regularizers,models

def train(model,ds_train,ds_val,hp):
    """
    train

    Args:
        model (tf.model): 
        ds_train (tf.dataset):
        ds_val (tf.dataset):

    Returns:
        history: train history
    """
    print('开始训练')
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    tf_callbcak=tf.keras.callbacks.TensorBoard(log_dir=train_log_dir)

    history=model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=hp.epochs,
        callbacks=tf_callbcak,
    )
    
    model.save(hp.model_name)
    return history

def test(model:Sequential,test_data:pd.DataFrame,test_data_label:pd.DataFrame):
    """

    Args:
        model (Sequential): _description_
        test_data (pd.DataFrame): _description_
        test_data_label (pd.DataFrame): _description_
    """
    test_data=test_data.reset_index(drop=True)
    test_data_label.index=test_data.index
    
    pred_data=model.predict(np.array(test_data))
    pred_data=pd.DataFrame(pred_data,columns=test_data.columns)
    pred_data.index=test_data.index
    
    pred_data['Label']=test_data_label['Label']
    pred_data['Loss']=np.mean(np.abs(pred_data-test_data), axis = 1)
    
    return pred_data

def Assessment(threshold,pred_data:pd.DataFrame):
    """

    Args:
        threshold (_type_): _description_
        pred_data (pd.DataFrame): _description_
    """
    scored_data=pd.DataFrame(index=pred_data.index)
    scored_data['threshold']=threshold
    scored_data['Loss']=pred_data['Loss']
    scored_data['detection']=pred_data['Loss']>scored_data['threshold']
    scored_data['True Label']=pred_data['Label']
    
    scored_data['True Negative'] = np.where((scored_data['detection'] == False) & (scored_data['True Label'] == False), True, False)
    scored_data['True Positive'] = np.where((scored_data['detection'] == True) & (scored_data['True Label'] == True), True, False)
    scored_data['False Positive'] = np.where((scored_data['detection'] == True) & (scored_data['True Label'] == False), True, False)
    scored_data['False Negative'] = np.where((scored_data['detection'] == False) & (scored_data['True Label'] == True), True, False)
    
    TN=scored_data['True Negative'].sum()
    TP=scored_data['True Positive'].sum()
    FP=scored_data['False Positive'].sum()
    FN=scored_data['False Negative'].sum()
    
    acc=(TP+TN)/(TN+TP+FN+FP)
    recall=(TP)/(TP+FN)
    pre=(TP)/(TP+FP)
    
    score={
        'Accuracy':acc,
        'Recall':recall,
        'Precision':pre,
    }
    
    return scored_data,score
    
def creat_model_1D(hyperp):
    """
    creat model with hyperParam,
    a fully connected stacked au-toencoder with three layers in the encoder and three layers in decoder networks
    The encoder initially reduces the dimension of the input vector from 63 to 4
    used ReLU activation in all hidden layers and Adam optimizer with learning rate of 0.001.
    
    exp:
    layers: [1,2,3,4,5] len=5
    dropout:[1,2,3,4,5]
    Args:
        hyperp (HyperParam): class of all hyper parameters
    """
    model=Sequential([
        #connect to the input X
        layers.Dropout(hyperp.dropout_rate[0],input_shape=(hyperp.input_size,)),
    ])
  
    #encoder
    for i in range(0,len(hyperp.layers)-1,1):
        model.add(
            layers.Dense(hyperp.layers[i],
                         activation=hyperp.activation,
                         kernel_initializer=initializers.glorot_uniform(hyperp.random_seed),
                         #kernel_regularizer=regularizers.l2(hyperp.regularizer),
                    )
            )
        model.add(layers.Dropout(hyperp.dropout_rate[i+1]))
        
    #bottleneck
    model.add(
        layers.Dense(hyperp.layers[-1],
                     activation=hyperp.activation,
                     kernel_initializer=initializers.glorot_uniform(hyperp.random_seed),
                )
    )
    
    #decoder
    
    for i in range(len(hyperp.layers)-2,-1,-1):
        model.add(
            layers.Dense(hyperp.layers[i],
                         activation=hyperp.activation,
                         kernel_initializer=initializers.glorot_uniform(hyperp.random_seed),
                         kernel_regularizer=regularizers.l2(hyperp.regularizer)
                    )
            )
        model.add(layers.Dropout(hyperp.dropout_rate[i+1]))
        
    model.add(
        layers.Dense(hyperp.input_size,
                     activation=hyperp.activation,
                     kernel_initializer=initializers.glorot_uniform(hyperp.random_seed),
                )
    )
        
    model.compile(
        loss=hyperp.loss,
        optimizer=hyperp.optimizer,
    )
    
    print(model.summary())
    
    return model

def creat_model_2D(hparam):
    """_summary_
    layers:[1,2,3,4,       5]
              ↑            ↑
            Conv         Dense
    Args:
        hparam (_type_): _description_

    Returns:
        model: Sequential
    """
    #encoder
    input=keras.Input(shape=hparam.input_size)
    x=layers.Conv2D(filters=hparam.layers[0],
                    kernel_size=[3,3],
                    strides=1,
                    padding='same',
                    activation=hparam.activation,
                )(input)
    x=layers.MaxPooling2D(pool_size=(2,2),strides=2,padding='same')(x)
    x=layers.Dropout(rate=hparam.dropout_rate[0])(x)
    
    for i in range(1,len(hparam.layers)-1,1):
        x=layers.Conv2D(filters=hparam.layers[i],
                    kernel_size=[3,3],
                    strides=1,
                    padding='same',
                    activation=hparam.activation,
                )(x)
        x=layers.MaxPooling2D(pool_size=(2,2),strides=2,padding='same')(x)
        x=layers.Dropout(rate=hparam.dropout_rate[i])(x)
    
    #bottle neck
    last_img_sahpe=x.shape[1:]
    print('before flatten shape:',last_img_sahpe)
    x=layers.Flatten()(x)
    after_flatten_shape=x.shape[1] # output_shape->(?,n)
    print('after flatten shape:',x.shape)
    # nnnnnn->nn
    bottleneck=layers.Dense(hparam.layers[-1],
                            activation=hparam.activation,
                            kernel_initializer=initializers.glorot_uniform(hparam.random_seed),
                            kernel_regularizer=regularizers.l2(hparam.regularizer),
                            )(x)
    
    #decoer
    #nn->nnnnnn
    x=layers.Dense(after_flatten_shape,
                   activation=hparam.activation,
                   kernel_initializer=initializers.glorot_uniform(hparam.random_seed),
                   kernel_regularizer=regularizers.l2(hparam.regularizer),
                  )(bottleneck)
    
    x=layers.Reshape(target_shape=last_img_sahpe)(x)
    
    for i in range(len(hparam.layers)-2,-1,-1):
        x=layers.UpSampling2D(size=(2,2))(x)
        x=layers.Conv2D(filters=hparam.layers[i],
                    kernel_size=[3,3],
                    strides=1,
                    padding='same',
                    activation=hparam.activation,
                )(x)
        if i > 0 :
            x=layers.Dropout(rate=hparam.dropout_rate[i])(x)
        
    output=layers.Conv2D(filters=1,
                    kernel_size=[3,3],
                    strides=1,
                    padding='same',
                    activation=hparam.activation,
                )(x)
    
    model=models.Model(inputs=input,outputs=output)
    model.compile(optimizer=hparam.optimizer,
                  loss=hparam.loss)
    print(model.summary())
    
    return model

def find_best_threshold(train_loss,model,ds_data,ds_data_label,increment=50):
    """
    μ+z⋅σ
    Args:
        train_loss (_type_): _description_
        model (_type_): _description_
        ds_data (_type_): _description_
        ds_data_label (_type_): _description_
        increment (int, optional): _description_. Defaults to 50.

    Returns:
        bestthreshold: _description_
    """
    loss_mean=np.mean(train_loss)
    loss_std=np.std(train_loss)
    
    maxium=min(1,loss_mean+3*loss_std)
    minium=max(0,loss_mean-3*loss_std)
    step=(maxium-minium)/increment
    
    print('start predict...')
    pred_data=test(model=model,
                           test_data=ds_data,
                           test_data_label=ds_data_label,
                           )
    
    print('threshold test area::','[',str(minium),',',str(maxium),']',',step:',str(step))
    ba,br,bp=0,0,0
    with tqdm(total=increment) as t : 
        for i in np.arange(minium,maxium,step):
            scored_data,score=Assessment(threshold=i,
                                         pred_data=pred_data,
                                         )
            if(score['Accuracy']>ba):
                ba=score['Accuracy']
                br=score['Recall']
                bp=score['Precision']
                bestthreshold=i
                bs=scored_data
            
            t.set_postfix({'Accuracy':'{0:1.5f}'.format(ba),
                           'Recall':'{0:1.5f}'.format(br),
                           'Precision':'{0:1.5f}'.format(bp)
                           })
            t.update(1)
            
    print('threshold:',bestthreshold)
    return bestthreshold,pred_data,bs