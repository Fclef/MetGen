from __future__ import absolute_import
from __future__ import print_function

import random

from tensorflow.keras import regularizers
from tensorflow.keras.layers import Lambda, Input, Dense ,Input, Flatten, Multiply, Reshape, Concatenate
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout,Conv1D, Conv2D, MaxPooling1D, Conv2DTranspose, BatchNormalization
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix
from random import randint
import pickle
import seaborn as sn
from sklearn import preprocessing
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics.cluster import adjusted_rand_score
import tensorflow as tf
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
from sklearn.decomposition import PCA

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.75
# tf.Session(config=config)

seed = randint(0,100)
# load data
f = open('./data/met_data/METgen_c19_g7312_data.pckl', 'rb')
data = pickle.load(f)
cancer_GE, tissue_GE, sampled_true, sampled_false, cancer_label,target_label, pos_label, tissue_label = data[0], data[2], data[6], data[7], data[1], data[4], data[5], data[3]
f.close()

m500_annotation = pd.read_table('./data/M.meta.plus.txt')
primary = m500_annotation["cohort"].values
cls_tissue  = m500_annotation["biopsy_tissue"].values
label_encoder = LabelEncoder()

f = open('./data/m500_7312_code.pckl', 'rb')
data = pickle.load(f)
met_code = data
f.close()
print(np.shape(met_code))
print('data are loaded!')
cancer_pos = cancer_GE.values[list(sampled_true[:, 0]), :]
eva_label_pos = primary[list(sampled_true[:, 2])]
tissue_pos = tissue_GE.values[list(sampled_true[:, 1]), :]
tissue_label_pos = tissue_label[list(sampled_true[:, 1])]
target_pos = met_code[list(sampled_true[:, 2]), :]
cancer_neg = cancer_GE.values[list(sampled_false[:, 0]), :]
eva_label_neg = cancer_label[list(sampled_false[:, 0])]
tissue_neg = tissue_GE.values[list(sampled_false[:, 1]), :]
tissue_label_neg = tissue_label[list(sampled_false[:, 1])]
target_neg = met_code[list(sampled_false[:, 2]), :]
label_pos = np.ones(19000)
label_neg = np.zeros(19000*3)

cancer = np.concatenate([cancer_pos, cancer_neg])
print('cancer samples are done!')

eva_label = np.concatenate([eva_label_pos, eva_label_neg])
eva_label = np.concatenate((eva_label, primary))
eva_label = label_encoder.fit_transform(eva_label)
eva_label = eva_label[0:19000*4]
print('cancer labels are done!')

tissue_label = np.concatenate([tissue_label_pos, tissue_label_neg])
tissue_label = np.concatenate((tissue_label, cls_tissue))
tissue_label = label_encoder.fit_transform(tissue_label)
tissue_label = tissue_label[0:19000*4]
print('tissue labels are done!')

tissue = np.concatenate([tissue_pos, tissue_neg])
print('tissue samples are done!')
target = np.concatenate([target_pos, target_neg])
print('target labels are done!')
label = np.concatenate([label_pos, label_neg])
print('similarity labels are done!')

cancer = np.expand_dims(cancer, -1)
tissue = np.expand_dims(tissue, -1)
target = np.expand_dims(target, -1)


cancer_pos = np.expand_dims(cancer_pos, -1)
tissue_pos = np.expand_dims(tissue_pos, -1)
target_pos = np.expand_dims(target_pos, -1)

cancer_neg = np.expand_dims(cancer_neg, -1)
tissue_neg = np.expand_dims(tissue_neg, -1)
target_neg = np.expand_dims(target_neg, -1)
print('data preprocess is done!')
#
indices = np.arange(label.shape[0])
np.random.shuffle(indices)
cancer, tissue, target, label, eva_label, tissue_label = cancer[indices], tissue[indices], target[indices], label[indices], eva_label[indices], tissue_label[indices]

#build CLS for cancer
def cls(nb_classes):
    input_main = Input((100,))
    # first layer
    dense = Dense(60, activation='relu',
                  # kernel_regularizer = regularizers.l1_l2(l1=1e-4, l2=1e-4)
                  )(input_main)
    drop = Dropout(0.25)(dense)
    dense = Dense(40, activation='relu',
                  # kernel_regularizer = regularizers.l1_l2(l1=1e-4, l2=1e-4)
                  )(drop)
    dense = Dropout(0.25)(dense)
    dense = Dense(nb_classes)(dense)
    softmax = Activation('softmax', name='Classifier')(dense)

    return Model(inputs=input_main, outputs=softmax)

cancer_model = cls(22)
tissue_model = cls(16)

cancer_model.load_weights(
    '/ix/yufeihuang/timothy/Disentanglement/Model_saved/met500_cls/cancer_model.h5', skip_mismatch=False)
tissue_model.load_weights(
    '/ix/yufeihuang/timothy/Disentanglement/Model_saved/met500_cls/tissue_model.h5', skip_mismatch=False)



def sampling(m_v):

    z_mean, z_log_var = m_v #(mean and var)
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]

    epsilon = K.random_normal(shape=(batch, dim))
    sample = z_mean + K.exp(0.5 * z_log_var) * epsilon
    return sample

def filterTheDict(dictObj, callback):
    newDict = dict()
    for (key, value) in dictObj.items():
        if callback((key, value)):
            newDict[key] = value
    return newDict

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 5
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def cancerVAE(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    # build encoder
    inputs = Input(shape = (input_shape, 1), name = 'encoder_input')
    block1 = Conv1D(filters=filters_L1,
                    kernel_size=kernel_size,
                    input_shape=(input_shape, 1),
                    kernel_initializer='he_normal',
                    strides=stride1)(inputs)
    block1 = BatchNormalization(axis=-1)(block1)
    block1 = Activation('relu')(block1)
    block2 = Conv1D(filters=filters_L2,
                    kernel_size=1,
                    kernel_initializer='he_normal',
                    strides=1)(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('relu')(block2)
    flatten = Flatten()(block2)

    z_mean    = Dense(latent_dim, name='z_mean')(flatten)
    z_log_var = Dense(latent_dim, name='z_log_var')(flatten)

    # use reparameterization trick
    x   = [z_mean, z_log_var]
    z   = Lambda(sampling, name='z_sample')(x)

    # instantiate encoder
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    plot_model(encoder, to_file='vae_encoder.png', show_shapes=True)

    # build decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    dense         = Dense(filters_L2 * 128, activation='relu')(latent_inputs)
    dense         = BatchNormalization()(dense)
    d_block2      = Reshape((128, 1, filters_L2))(dense)
    d_block2      = Conv2DTranspose(filters_L1, (1, 1), strides=(1, 1), activation='relu')(d_block2)
    d_block2      = BatchNormalization(axis=-1)(d_block2)
    d_block1      = Conv2DTranspose(1, (kernel_size, 1), strides=(stride1, 1), activation='sigmoid')(d_block2)
    outputs       = Reshape((input_shape, 1))(d_block1)

    # instantiate decoder
    decoder = Model(latent_inputs, outputs, name='decoder')
    plot_model(decoder, to_file='vae_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    code    = encoder(inputs)[0]
    vae_cancer = Model(inputs, [code,outputs,z_mean,z_log_var])

    return vae_cancer

def tissueVAE(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    # build encoder
    inputs = Input(shape=(input_shape, 1), name='encoder_input')
    block1 = Conv1D(filters=filters_L1,
                    kernel_size=kernel_size,
                    input_shape=(input_shape, 1),
                    kernel_initializer='he_normal',
                    strides=stride1)(inputs)
    block1 = BatchNormalization(axis=-1)(block1)
    block1 = Activation('relu')(block1)
    block2 = Conv1D(filters=filters_L2,
                    kernel_size=1,
                    kernel_initializer='he_normal',
                    strides=1)(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('relu')(block2)
    flatten = Flatten()(block2)

    z_mean = Dense(latent_dim, name='z_mean')(flatten)
    z_log_var = Dense(latent_dim, name='z_log_var')(flatten)

    # use reparameterization trick
    x = [z_mean, z_log_var]
    z = Lambda(sampling, name='z_sample')(x)

    # instantiate encoder
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    plot_model(encoder, to_file='vae_encoder.png', show_shapes=True)

    # build decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    dense = Dense(filters_L2 * 128, activation='relu')(latent_inputs)
    dense = BatchNormalization()(dense)
    d_block2 = Reshape((128, 1, filters_L2))(dense)
    d_block2 = Conv2DTranspose(filters_L1, (1, 1), strides=(1, 1), activation='relu')(d_block2)
    d_block2 = BatchNormalization(axis=-1)(d_block2)
    d_block1 = Conv2DTranspose(1, (kernel_size, 1), strides=(stride1, 1), activation='sigmoid')(d_block2)
    outputs = Reshape((input_shape, 1))(d_block1)

    # instantiate decoder
    decoder = Model(latent_inputs, outputs, name='decoder')
    plot_model(decoder, to_file='vae_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    code    = encoder(inputs)[0]
    vae_tissue = Model(inputs, [code,outputs,z_mean,z_log_var])
    return vae_tissue

def baseVAE(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    # build encoder
    inputs = Input(shape = (input_shape, 1), name = 'encoder_input')
    block1 = Conv1D(filters=filters_L1,
                    kernel_size=kernel_size,
                    input_shape=(input_shape, 1),
                    kernel_initializer='he_normal',
                    strides=stride1)(inputs)
    block1 = BatchNormalization(axis=-1)(block1)
    block1 = Activation('relu')(block1)
    block2 = Conv1D(filters=filters_L2,
                    kernel_size=1,
                    kernel_initializer='he_normal',
                    strides=1)(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('relu')(block2)
    flatten = Flatten()(block2)

    z_mean    = Dense(latent_dim, name='z_mean')(flatten)
    z_log_var = Dense(latent_dim, name='z_log_var')(flatten)

    # use reparameterization trick
    x   = [z_mean, z_log_var]
    z   = Lambda(sampling, name='z_sample')(x)

    # instantiate encoder
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    plot_model(encoder, to_file='vae_encoder.png', show_shapes=True)

    # build decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    dense         = Dense(filters_L2 * 128, activation='relu')(latent_inputs)
    dense         = BatchNormalization()(dense)
    d_block2      = Reshape((128, 1, filters_L2))(dense)
    d_block2      = Conv2DTranspose(filters_L1, (1, 1), strides=(1, 1), activation='relu')(d_block2)
    d_block2      = BatchNormalization(axis=-1)(d_block2)
    d_block1      = Conv2DTranspose(1, (kernel_size, 1), strides=(stride1, 1), activation='sigmoid')(d_block2)
    outputs       = Reshape((input_shape, 1))(d_block1)

    # instantiate decoder
    decoder = Model(latent_inputs, outputs, name='decoder')
    plot_model(decoder, to_file='vae_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    code    = encoder(inputs)[0]
    vae = Model(inputs, [code,outputs,z_mean,z_log_var])

    return vae

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def TCGA_CNN():
    input_main = Input((input_shape, 1))
    # first layer
    block1 = Conv1D(filters = filters_L1,
                    kernel_size = kernel_size,
                    input_shape = (input_shape, 1),
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
                    strides= stride1)(input_main)
    block1 = BatchNormalization(axis=-1)(block1)
    block1 = Activation('relu')(block1)
    # block1 = Dropout(dropoutRate)(block1)
    # second layer
    block2 = Conv1D(filters = filters_L2,
                    kernel_size = 4,
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
                    strides= 4)(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('relu')(block2)

    # flatten
    flatten= Flatten()(block2)
    fcl    = Dropout(0.25)(flatten)
    fcl    = Dense(512)(fcl)
    return Model(inputs=input_main, outputs=fcl)

def CNN_mixer():
    input_cancer = Input(shape=(input_shape, 1))
    input_tissue = Input(shape=(input_shape, 1))
    met_code = Input(shape=(100, 1))

    code_cancer = feature_extractor(input_cancer)
    code_tissue = feature_extractor(input_tissue)

    code_cancer = Flatten(name='cancer_code')(code_cancer)
    code_tissue = Flatten(name='tissue_code')(code_tissue)

    concatted = Concatenate(axis=1)([code_cancer, code_tissue])
    concatted = Reshape((2, 512, 1))(concatted)

    code = Conv2D(filters=64,
                  kernel_size=(2, 1),
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-3),
                  input_shape=(2, 512, 1),
                  strides=1)(concatted)
    code = BatchNormalization(axis=-1)(code)
    code = Conv2D(filters=32,
                  kernel_size=(1, 32),
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-3),
                  activation='relu',
                  strides=(1, 32))(code)
    code = BatchNormalization(axis=-1)(code)
    code = Flatten()(code)
    code = Dropout(rate=0.25)(code)
    code = Dense(units=200, activation='relu')(code)
    code = Dropout(rate=0.25)(code)
    code = Dense(units=latent_dim, name='learned_code')(code)

    met_latent_code = Flatten()(met_code)

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([code, met_latent_code])
    return Model([input_cancer, input_tissue, met_code], distance)

input_shape = 7312
latent_dim = 100
epochs = 400
filters_L1 = 64
filters_L2 = 16
kernel_size = 32
stride1 = 32
lr = 0.0005
decay = 1e-6

feature_extractor = TCGA_CNN()
model = CNN_mixer()

feature_extractor.summary()
model.summary()

plot_model(model, to_file='met500_gen.png', show_shapes=True)

if __name__ == '__main__':
    kf = StratifiedKFold(n_splits=5, shuffle= True, random_state=seed)
    for train_index, test_index in kf.split(cancer, label):
        cancer_train, cancer_test = cancer[train_index], cancer[test_index]
        tissue_train, tissue_test = tissue[train_index], tissue[test_index]
        target_train, target_test = target[train_index], target[test_index]
        label_train, label_test   = label[train_index] , label[test_index]
        tissue_label_train, tissue_label_test = tissue_label[train_index], tissue_label[test_index]
        eva_label_train, eva_label_test = eva_label[train_index], eva_label[test_index]

        f = open('/ix/yufeihuang/timothy/Disentanglement/data/met_data/tr_te_index.pckl', 'wb')
        pickle.dump([train_index, test_index], f)
        f.close()

        rms = RMSprop()
        adam = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=decay, amsgrad=True)
        model.compile(loss=contrastive_loss,
                    optimizer=adam, metrics=[accuracy])
        print('Model is training...')
        hist = model.fit(x=[cancer_train, tissue_train, target_train],
                        y=label_train,
                        batch_size=128,
                        epochs=epochs,
                        validation_data=([cancer_test, tissue_test, target_test], label_test),
                        verbose=0,
                        shuffle=True)

        # training curve contrastive
        golden_size = lambda width: (width, 2. * width / (1 + np.sqrt(5)))
        fig, ax = plt.subplots(figsize=golden_size(6))
        hist_vae = {k: hist.history[k] for k in ('loss', 'val_loss')}
        hist_vae_df = pd.DataFrame(hist_vae)
        hist_vae_df.plot(ax=ax)
        ax.set_ylabel('Contrastive Loss')
        ax.set_xlabel('# epochs')

        # compute final accuracy on training and test sets
        # model.save('/ix/yufeihuang/timothy/Disentanglement/Model_saved/met_gen/met_gen.h5')
        y_pred_tr = model.predict([cancer_train, tissue_train, target_train])
        tr_acc = compute_accuracy(label_train, y_pred_tr)
        y_pred_te = model.predict([cancer_test, tissue_test, target_test])
        te_acc = compute_accuracy(label_test, y_pred_te)
        print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
        print('tr_range:',  (np.min(y_pred_tr), np.max(y_pred_tr)))
        print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
        print('te_range:', (np.min(y_pred_te), np.max(y_pred_te)))
        fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
        n_bins = 30
        # We can set the number of bins with the `bins` kwarg
        axs[0].hist(y_pred_tr, bins=n_bins)
        axs[1].hist(y_pred_te, bins=n_bins)
        # evaluation
        # eval_cancer_true, eval_tissue_true, eval_target_true, eva_label_true = cancer_test[label_test == 1],\
        #                                                                        tissue_test[label_test == 1], target_test[label_test == 1], eva_label_test[label_test == 1]
        eval_cancer_true, eval_tissue_true, eval_target_true, eva_label_true, tissue_label_true = cancer_test[label_test == 1],\
                                                                            tissue_test[label_test == 1], target_test[label_test == 1], eva_label_test[label_test == 1], tissue_label_test[label_test == 1]
        eval_cancer_false, eval_tissue_false, eval_target_false, eva_label_false,tissue_label_false = cancer_test[label_test == 0],\
                                                                            tissue_test[label_test == 0], target_test[label_test == 0], eva_label_test[label_test == 0], tissue_label_test[label_test == 0]
        Eva_model = Model(inputs= model.inputs, outputs= model.get_layer('learned_code').output)

        class_pred_true  = np.argmax(cancer_model.predict(Eva_model.predict([eval_cancer_true , eval_tissue_true , eval_target_true])) , axis=1)
        # class_pred_false = np.argmax(cancer_model.predict(Eva_model.predict([eval_cancer_false, eval_tissue_false, eval_target_false])), axis=1)

        accuracy_true = accuracy_score(class_pred_true, eva_label_true)
        # accuracy_false = accuracy_score(class_pred_false, eva_label_false)
        print("classification_cancer_pos:", accuracy_true)
        # print("classification_cancer_neg:", accuracy_false)

        class_pred_true  = np.argmax(tissue_model.predict(Eva_model.predict([eval_cancer_true , eval_tissue_true , eval_target_true])), axis=1)
        # class_pred_false = np.argmax(tissue_model.predict(Eva_model.predict([eval_cancer_false, eval_tissue_false, eval_target_false])), axis=1)

        accuracy_true = accuracy_score(class_pred_true, tissue_label_true)
        # accuracy_false = accuracy_score(class_pred_false, tissue_label_false)
        print("classification_tissue_pos:", accuracy_true)
        # print("classification_tissue_neg:", accuracy_false)


        ##latent visualization
        latent_model = Model(inputs= model.inputs, outputs= [model.get_layer('cancer_code').output, model.get_layer('tissue_code').output, model.get_layer('learned_code').output])
        latent_cancer, latent_tissue, latent_learned = latent_model.predict([cancer_test, tissue_test, target_test])

        X_embedded_cancer = TSNE(n_components=2).fit_transform(latent_cancer)
        print('cancer TSNE is done')
        X_embedded_tissue = TSNE(n_components=2).fit_transform(latent_tissue)
        print('tissue TSNE is done')
        X_embedded_learned = TSNE(n_components=2).fit_transform(latent_learned)
        print('code TSNE is done')
        fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)

        f1 = axs[0][0].scatter(X_embedded_cancer[:, 0], X_embedded_cancer[:, 1], c=eva_label_test)
        f2 = axs[0][1].scatter(X_embedded_tissue[:, 0], X_embedded_tissue[:, 1], c=tissue_label_test)
        f3 = axs[1][0].scatter(X_embedded_learned[:, 0], X_embedded_learned[:, 1], c=eva_label_test)
        f4 = axs[1][1].scatter(X_embedded_learned[:, 0], X_embedded_learned[:, 1], c=tissue_label_test)
        # fig.colorbar(f1)
        # fig.colorbar(f2)
        # fig.colorbar(f3)
        # fig.colorbar(f4)

        pca = PCA(n_components=50)
        cancer_comp = pca.fit_transform(np.squeeze(cancer_test))
        tissue_comp = pca.fit_transform(np.squeeze(tissue_test))
        X_data_cancer = TSNE(n_components=2).fit_transform(np.squeeze(cancer_comp))
        print('Cancer data TSNE is done')
        X_data_tissue = TSNE(n_components=2).fit_transform(np.squeeze(tissue_comp))
        print('Tissue data TSNE is done')
        fig_1, axs_1 = plt.subplots(2, sharey=True, tight_layout=True)
        axs_1[0].scatter(X_data_cancer[:, 0], X_data_cancer[:, 1], c=eva_label_test)
        axs_1[1].scatter(X_data_tissue[:, 0], X_data_tissue[:, 1], c=tissue_label_test)

plt.show()
