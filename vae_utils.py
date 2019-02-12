import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.metrics import silhouette_score
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras import backend as K


        
def generate_data(n=1000, type='linear', leaky_slope=0.01, contrast=True, tg_var=0.5, return_params=False):
    C = np.random.normal(0, 1, (4, 4))
    D = np.random.normal(0, 1, (4, 4))
    E = np.random.normal(0, 1, (4, 4))
    
    half = int(n/2)
    Z = np.zeros((n, 4))
    Z[:, 0:2] = np.random.normal(0, tg_var, Z[:, 0:2].shape) 
    Z[:, 2:4] = np.random.normal(0, 3, Z[:, 2:4].shape)
    tg = Z.dot(C)
    if type=='abs':
        temp = np.zeros((n, 6))
        temp[:, 0] = np.abs(tg[:, 0]) - tg[:, 1]
        temp[:, 1] = np.abs(tg[:, 0] + tg[:, 1])
        temp[:, 4] = tg[:, 0]
        temp[:, 2] = np.abs(tg[:, 2]) - tg[:, 3]
        temp[:, 3] = np.abs(tg[:, 2] + tg[:, 3])
        temp[:, 5] = tg[:, 2]
        tg = temp
    elif type=='expit':
        tg = expit(tg)
        tg = tg.dot(D)
        tg = np.where(tg > 0, tg, tg * leaky_slope)
        tg = tg.dot(E)
    elif type=='linear':
        pass
    elif type=='leaky':
        tg = np.where(tg > 0, tg, tg * leaky_slope)
        tg = tg.dot(D)
        tg = np.where(tg < 0, tg, tg * leaky_slope)
        tg = tg.dot(E)
    else:
        raise ValueError('Invalid argument for parameter: type')
    
    if contrast:
        labels = Z[:, 0] > 0
    else:
        labels = Z[:, -1] > 0

    # background
    S = np.zeros((n, 4))
    S[:, 0:2] = np.random.normal(0, 0, S[:, 0:2].shape)
    S[:, 2:4] = np.random.normal(0, 3, S[:, 2:4].shape)
    bg = S.dot(C)
    if type=='abs':
        temp = np.zeros((n, 6))
        temp[:, 0] = np.abs(bg[:, 0]) - bg[:, 1]
        temp[:, 1] = np.abs(bg[:, 0] + bg[:, 1])
        temp[:, 4] = bg[:, 0]
        temp[:, 2] = np.abs(bg[:, 2]) - bg[:, 3]
        temp[:, 3] = np.abs(bg[:, 2] + bg[:, 3])
        temp[:, 5] = bg[:, 2]
        bg = temp
    elif type=='expit':
        bg = expit(bg)
    elif type=='linear':
        pass
    elif type=='leaky':
        bg = np.where(bg > 0, bg, bg * leaky_slope)
        bg = bg.dot(D)
    else:
        raise ValueError('Invalid argument for parameter: type')
        
    tg = tg - tg.mean(axis=0)
    bg = bg - bg.mean(axis=0)
    if return_params:
        return tg, labels, bg, C
    return tg, labels, bg

def standard_vae(input_dim=4, intermediate_dim=12, latent_dim=2, beta=1):
    input_shape = (input_dim, )

    inputs = tf.keras.layers.Input(shape=input_shape, name='encoder_input')
    if isinstance(intermediate_dim, int):
        x = tf.keras.layers.Dense(intermediate_dim, activation='relu')(inputs)
    else:
        x = inputs
        for dim in intermediate_dim:
            x = tf.keras.layers.Dense(dim, activation='relu')(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = tf.keras.models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
    # encoder.summary()
    # tf.keras.utils.plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = tf.keras.layers.Input(shape=(latent_dim,), name='z_sampling')
    if isinstance(intermediate_dim, int):
        x = tf.keras.layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
    else:
        x = latent_inputs
        for dim in intermediate_dim:
            x = tf.keras.layers.Dense(dim, activation='relu')(x)        
    outputs = tf.keras.layers.Dense(input_dim)(x)

    # instantiate decoder model
    decoder = tf.keras.models.Model(latent_inputs, outputs, name='decoder')
    # decoder.summary()
    # tf.keras.utils.plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = tf.keras.models.Model(inputs, outputs, name='vae_mlp')

    reconstruction_loss = tf.keras.losses.mse(inputs, outputs)
    reconstruction_loss *= input_dim

    kl_loss = 1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var)
    kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = tf.keras.backend.mean(reconstruction_loss + beta*kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    return vae, encoder, decoder

def contrastive_vae(input_dim=4, intermediate_dim=12, latent_dim=2, beta=1, disentangle=False, gamma=0):
    input_shape = (input_dim, )
    batch_size = 100
    tg_inputs = tf.keras.layers.Input(shape=input_shape, name='tg_input')
    bg_inputs = tf.keras.layers.Input(shape=input_shape, name='bg_input')
    
    if isinstance(intermediate_dim, int):
        intermediate_dim = [intermediate_dim]
    
    z_h_layers = []
    for dim in intermediate_dim:
        z_h_layers.append(tf.keras.layers.Dense(dim, activation='relu'))
    z_mean_layer = tf.keras.layers.Dense(latent_dim, name='z_mean')
    z_log_var_layer = tf.keras.layers.Dense(latent_dim, name='z_log_var')
    z_layer = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')

    def z_encoder_func(inputs):
        z_h = inputs
        for z_h_layer in z_h_layers:
            z_h = z_h_layer(z_h)
        z_mean = z_mean_layer(z_h)
        z_log_var = z_log_var_layer(z_h)
        z = z_layer([z_mean, z_log_var])
        return z_mean, z_log_var, z

    s_h_layers = []
    for dim in intermediate_dim:
        s_h_layers.append(tf.keras.layers.Dense(dim, activation='relu'))
    s_mean_layer = tf.keras.layers.Dense(latent_dim, name='s_mean')
    s_log_var_layer = tf.keras.layers.Dense(latent_dim, name='s_log_var')
    s_layer = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='s')

    def s_encoder_func(inputs):
        s_h = inputs
        for s_h_layer in s_h_layers:
            s_h = s_h_layer(s_h)
        s_mean = s_mean_layer(s_h)
        s_log_var = s_log_var_layer(s_h)
        s = s_layer([s_mean, s_log_var])
        return s_mean, s_log_var, s

    tg_z_mean, tg_z_log_var, tg_z = z_encoder_func(tg_inputs)
    tg_s_mean, tg_s_log_var, tg_s = s_encoder_func(tg_inputs)
    bg_s_mean, bg_s_log_var, bg_s = s_encoder_func(bg_inputs)

    z_encoder = tf.keras.models.Model(tg_inputs, [tg_z_mean, tg_z_log_var, tg_z], name='z_encoder')
    s_encoder = tf.keras.models.Model(tg_inputs, [tg_s_mean, tg_s_log_var, tg_s], name='s_encoder')

    # build decoder model
    cvae_latent_inputs = tf.keras.layers.Input(shape=(2 * latent_dim,), name='sampled')
    cvae_h = cvae_latent_inputs
    for dim in intermediate_dim:
        cvae_h = tf.keras.layers.Dense(dim, activation='relu')(cvae_h)
    cvae_outputs = tf.keras.layers.Dense(input_dim)(cvae_h)

    cvae_decoder = tf.keras.models.Model(inputs=cvae_latent_inputs, outputs=cvae_outputs, name='decoder')

    # decoder.summary()

    def zeros_like(x):
        return tf.zeros_like(x)

    tg_outputs = cvae_decoder(tf.keras.layers.concatenate([tg_z, tg_s], -1))
    zeros = tf.keras.layers.Lambda(zeros_like)(tg_z)
    bg_outputs = cvae_decoder(tf.keras.layers.concatenate([zeros, bg_s], -1))
    fg_outputs = cvae_decoder(tf.keras.layers.concatenate([tg_z, zeros], -1))

    cvae = tf.keras.models.Model(inputs=[tg_inputs, bg_inputs], 
                                 outputs=[tg_outputs, bg_outputs], 
                                 name='contrastive_vae')

    cvae_fg = tf.keras.models.Model(inputs=tg_inputs, 
                                    outputs=fg_outputs, 
                                    name='contrastive_vae_fg')

    # cvae.summary()
    if disentangle:
        discriminator = Dense(1, activation='sigmoid')
        
        z1 = Lambda(lambda x: x[:int(batch_size/2),:])(tg_z)
        z2 = Lambda(lambda x: x[int(batch_size/2):,:])(tg_z)
        s1 = Lambda(lambda x: x[:int(batch_size/2),:])(tg_s)
        s2 = Lambda(lambda x: x[int(batch_size/2):,:])(tg_s)
        q_bar = tf.keras.layers.concatenate(
            [tf.keras.layers.concatenate([s1, z2], axis=1),
            tf.keras.layers.concatenate([s2, z1], axis=1)],
            axis=0)
        q = tf.keras.layers.concatenate(
            [tf.keras.layers.concatenate([s1, z1], axis=1),
            tf.keras.layers.concatenate([s2, z2], axis=1)],
            axis=0)
        q_bar_score = discriminator(q_bar)
        q_score = discriminator(q)        
        tc_loss = K.log(q_score / (1 - q_score)) 
        
        discriminator_loss = - K.log(q_score) - K.log(1 - q_bar_score)

    reconstruction_loss = tf.keras.losses.mse(tg_inputs, tg_outputs)
    reconstruction_loss += tf.keras.losses.mse(bg_inputs, bg_outputs)
    reconstruction_loss *= input_dim


    kl_loss = 1 + tg_z_log_var - tf.keras.backend.square(tg_z_mean) - tf.keras.backend.exp(tg_z_log_var)
    kl_loss += 1 + tg_s_log_var - tf.keras.backend.square(tg_s_mean) - tf.keras.backend.exp(tg_s_log_var)
    kl_loss += 1 + bg_s_log_var - tf.keras.backend.square(bg_s_mean) - tf.keras.backend.exp(bg_s_log_var)
    kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    if disentangle:
        cvae_loss = K.mean(reconstruction_loss) + beta*K.mean(kl_loss) + gamma * K.mean(tc_loss) + K.mean(discriminator_loss)
    else:
        cvae_loss = K.mean(reconstruction_loss) + K.mean(kl_loss)

    cvae.add_loss(cvae_loss)
    cvae.compile(optimizer='adam')
    
    return cvae, cvae_fg, z_encoder, s_encoder, cvae_decoder


def contrastive_vae_no_bias(input_dim=4, intermediate_dim=12, latent_dim=2, beta=1):
    original_dim = input_dim
    input_shape = (original_dim, )
    
    # inputs = tf.keras.layers.Input(shape=input_shape, name='encoder_input')
    tg_inputs = tf.keras.layers.Input(shape=input_shape, name='tg_input')
    bg_inputs = tf.keras.layers.Input(shape=input_shape, name='bg_input')

    if isinstance(intermediate_dim, int):
        intermediate_dim = [intermediate_dim]    
    
    z_h_layers = []
    for dim in intermediate_dim:
        z_h_layers.append(tf.keras.layers.Dense(dim, activation='relu', use_bias=False))
        
    z_mean_layer = tf.keras.layers.Dense(latent_dim, name='z_mean', use_bias=False)
    z_log_var_layer = tf.keras.layers.Dense(latent_dim, name='z_log_var', use_bias=False)
    z_layer = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')
    # z_encoder = tf.keras.models.Model(inputs, [z_mean, z_log_var, z], name='z_encoder')

    def z_encoder_func(inputs):
        z_h = inputs
        for z_h_layer in z_h_layers:
            z_h = z_h_layer(z_h)
        z_mean = z_mean_layer(z_h)
        z_log_var = z_log_var_layer(z_h)
        z = z_layer([z_mean, z_log_var])
        return z_mean, z_log_var, z

    s_h_layers = []
    for dim in intermediate_dim:
        s_h_layers.append(tf.keras.layers.Dense(dim, activation='relu', use_bias=False))
    s_mean_layer = tf.keras.layers.Dense(latent_dim, name='s_mean', use_bias=False)
    s_log_var_layer = tf.keras.layers.Dense(latent_dim, name='s_log_var', use_bias=False)
    s_layer = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='s')

    def s_encoder_func(inputs):
        s_h = inputs
        for s_h_layer in s_h_layers:
            s_h = s_h_layer(s_h)
        s_mean = s_mean_layer(s_h)
        s_log_var = s_log_var_layer(s_h)
        s = s_layer([s_mean, s_log_var])
        return s_mean, s_log_var, s

    # s_h = tf.keras.layers.Dense(intermediate_dim, activation='relu')(inputs)
    # s_mean = tf.keras.layers.Dense(latent_dim, name='s_mean')(s_h)
    # s_log_var = tf.keras.layers.Dense(latent_dim, name='s_log_var')(s_h)
    # s = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='s')([s_mean, s_log_var])
    # s_encoder = tf.keras.models.Model(inputs, [s_mean, s_log_var, s], name='s_encoder')

    tg_z_mean, tg_z_log_var, tg_z = z_encoder_func(tg_inputs)
    tg_s_mean, tg_s_log_var, tg_s = s_encoder_func(tg_inputs)
    bg_s_mean, bg_s_log_var, bg_s = s_encoder_func(bg_inputs)

    z_encoder = tf.keras.models.Model(tg_inputs, [tg_z_mean, tg_z_log_var, tg_z], name='z_encoder')
    s_encoder = tf.keras.models.Model(tg_inputs, [tg_s_mean, tg_s_log_var, tg_s], name='s_encoder')

    # build decoder model
    cvae_latent_inputs = tf.keras.layers.Input(shape=(2 * latent_dim,), name='sampled')
    cvae_h = cvae_latent_inputs
    for dim in intermediate_dim:
        cvae_h = tf.keras.layers.Dense(dim, activation='relu', use_bias=False)(cvae_h)
    cvae_outputs = tf.keras.layers.Dense(original_dim, use_bias=False)(cvae_h)

    cvae_decoder = tf.keras.models.Model(inputs=cvae_latent_inputs, outputs=cvae_outputs, name='decoder')

    # decoder.summary()

    def zeros_like(x):
        return tf.zeros_like(x)

    tg_outputs = cvae_decoder(tf.keras.layers.concatenate([tg_z, tg_s], -1))
    zeros = tf.keras.layers.Lambda(zeros_like)(tg_z)
    bg_outputs = cvae_decoder(tf.keras.layers.concatenate([zeros, bg_s], -1))
    fg_outputs = cvae_decoder(tf.keras.layers.concatenate([tg_z, zeros], -1))

    cvae = tf.keras.models.Model(inputs=[tg_inputs, bg_inputs], 
                                 outputs=[tg_outputs, bg_outputs], 
                                 name='contrastive_vae')

    cvae_fg = tf.keras.models.Model(inputs=tg_inputs, 
                                    outputs=fg_outputs, 
                                    name='contrastive_vae_fg')

    reconstruction_loss = tf.keras.losses.mse(tg_inputs, tg_outputs)
    reconstruction_loss += tf.keras.losses.mse(bg_inputs, bg_outputs)
    reconstruction_loss *= original_dim


    kl_loss = 1 + tg_z_log_var - tf.keras.backend.square(tg_z_mean) - tf.keras.backend.exp(tg_z_log_var)
    kl_loss += 1 + tg_s_log_var - tf.keras.backend.square(tg_s_mean) - tf.keras.backend.exp(tg_s_log_var)
    kl_loss += 1 + bg_s_log_var - tf.keras.backend.square(bg_s_mean) - tf.keras.backend.exp(bg_s_log_var)
    kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    cvae_loss = tf.keras.backend.mean(reconstruction_loss + beta*kl_loss)
    cvae.add_loss(cvae_loss)
    cvae.compile(optimizer='adam')
    
    return cvae, cvae_fg, z_encoder, s_encoder, cvae_decoder


def shuffle_weights(model, weights=None):
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)

def sampling(args):
    z_mean, z_log_var = args
    batch = tf.keras.backend.shape(z_mean)[0]
    dim = tf.keras.backend.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

def plot_latent_space(encoder, x, y, plot=True, name='z_mean'):
    from sklearn.metrics import silhouette_score
    z_mean, _, _ = encoder.predict(x, batch_size=128)
    ss = round(silhouette_score(z_mean, y), 3)
    if plot:
        plt.figure()
        plt.scatter(z_mean[:, 0],z_mean[:, 1], c=y, cmap='Accent')
        plt.title(name + ', Silhouette score: ' + str(ss))
    return ss


def plot_latent_space4d(encoder, x, y, plot=True, name='z_mean'):
    from sklearn.metrics import silhouette_score
    z_mean, _, _ = encoder.predict(x, batch_size=128)
    if plot:
        plt.figure()
        ss = round(silhouette_score(z_mean[:,0:2], y), 3)
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y, cmap='Accent')
        plt.title(name + '12, Silhouette score: ' + str(ss))
        
        plt.figure()
        ss = round(silhouette_score(z_mean[:,2:4], y), 3)
        plt.scatter(z_mean[:, 2], z_mean[:, 3], c=y, cmap='Accent')
        plt.title(name + '34, Silhouette score: ' + str(ss))
    return ss

def plot_reconstructions_vae(model, x, y, plot=True):
    x_reconstructed = model.predict(x)
    mn = min(np.min(x_reconstructed), np.min(x))
    mx = max(np.max(x_reconstructed), np.max(x))
    plt.subplot(1, 3, 1)
    plt.imshow(x[:10], vmin=mn, vmax=mx, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(x_reconstructed[:10], vmin=mn, vmax=mx, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(y[:10, np.newaxis], cmap='gray')    
    
def plot_reconstructions_cvae(model, model_fg, x, bg, fg, y, plot=True):
    x_reconstructed, bg_reconstructed = model.predict([x, bg])
    mn = min(np.min(x_reconstructed), np.min(x))
    mx = max(np.max(x_reconstructed), np.max(x))
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(x[:10], vmin=mn, vmax=mx, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(x_reconstructed[:10], vmin=mn, vmax=mx, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(y[:10, np.newaxis], cmap='gray')    
    plt.suptitle('Target Path')
    
    mn = min(np.min(bg_reconstructed), np.min(bg))
    mx = max(np.max(bg_reconstructed), np.max(bg))
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(bg[:10], vmin=mn, vmax=mx, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(bg_reconstructed[:10], vmin=mn, vmax=mx, cmap='gray')
    plt.suptitle('Background Path')
    
    fg_reconstructed = model_fg.predict(fg)
    mn = min(np.min(fg_reconstructed), np.min(fg))
    mx = max(np.max(fg_reconstructed), np.max(fg))
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(fg[:10], vmin=mn, vmax=mx, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(fg_reconstructed[:10], vmin=mn, vmax=mx, cmap='gray')
    plt.suptitle('Foreground Path')
    

def plot_clean_digits_only(cvae, z_encoder, s_encoder, decoder, x, bg, plot=True):    
    x = x[:10]; bg = bg[:10]
    x_out, bg_out = cvae.predict([x, bg])
    _, _, x_z = z_encoder.predict(x)
    _, _, x_s = s_encoder.predict(x)
    x_out_enc_dec = decoder.predict(np.concatenate((x_z, x_s), axis=-1))
    x_clean = decoder.predict(np.concatenate((x_z, np.zeros_like(x_s)), axis=-1))
    x_dirty = decoder.predict(np.concatenate((np.zeros_like(x_z), x_s), axis=-1))
    x_empty = decoder.predict(np.concatenate((np.zeros_like(x_z), np.zeros_like(x_s)), axis=-1))
    
    plt.figure(figsize=[15, 3])
    plt.suptitle('Original Digits')
    for i in range(5):
        mn = np.min(x)
        mx = np.max(x)
        plt.subplot(1, 5, i+1)
        plt.imshow(x[i].reshape(28, 28), vmin=mn, vmax=mx, cmap='gray')
        
    plt.figure(figsize=[15, 3])
    plt.suptitle('Zeroing Irrelevant Latent Varialbes')
    for i in range(5):
        mn = np.min(x_clean)
        mx = np.max(x_clean)
        plt.subplot(1, 5, i+1)
        plt.imshow(x_clean[i].reshape(28, 28), vmin=mn, vmax=mx, cmap='gray')

    plt.figure(figsize=[15, 3])
    plt.suptitle('Original Digits')
    for i in range(5):
        mn = np.min(x)
        mx = np.max(x)
        plt.subplot(1, 5, i+1)
        plt.imshow(x[i+5].reshape(28, 28), vmin=mn, vmax=mx, cmap='gray')
        
    plt.figure(figsize=[15, 3])
    plt.suptitle('Zeroing Irrelevant Latent Varialbes')
    for i in range(5):
        mn = np.min(x_clean)
        mx = np.max(x_clean)
        plt.subplot(1, 5, i+1)
        plt.imshow(x_clean[i+5].reshape(28, 28), vmin=mn, vmax=mx, cmap='gray')

    
def plot_clean_digits(cvae, z_encoder, s_encoder, decoder, x, bg, plot=True):    
    x = x[:10]; bg = bg[:10]
    x_out, bg_out = cvae.predict([x, bg])
    _, _, x_z = z_encoder.predict(x)
    _, _, x_s = s_encoder.predict(x)
    x_out_enc_dec = decoder.predict(np.concatenate((x_z, x_s), axis=-1))
    x_clean = decoder.predict(np.concatenate((x_z, np.zeros_like(x_s)), axis=-1))
    x_dirty = decoder.predict(np.concatenate((np.zeros_like(x_z), x_s), axis=-1))
    x_empty = decoder.predict(np.concatenate((np.zeros_like(x_z), np.zeros_like(x_s)), axis=-1))
    mn = min(np.min(x), np.min(x_clean), np.min(x_out_enc_dec), np.min(x_clean), np.min(x_dirty), np.min(x_empty))
    mx = max(np.max(x), np.max(x_clean), np.max(x_out_enc_dec), np.max(x_clean), np.max(x_dirty), np.max(x_empty))
    
    plt.figure(figsize=[15, 3])
    plt.suptitle('Original Digits')
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(x[i].reshape(28, 28), vmin=mn, vmax=mx, cmap='gray')
        
    plt.figure(figsize=[15, 3])
    plt.suptitle('Reconstructed (cVAE) Digits')
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(x_out[i].reshape(28, 28), vmin=mn, vmax=mx, cmap='gray')
        
    plt.figure(figsize=[15, 3])
    plt.suptitle('Reconstructed (Enc-Dec) Digits')
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(x_out_enc_dec[i].reshape(28, 28), vmin=mn, vmax=mx, cmap='gray')

    plt.figure(figsize=[15, 3])
    plt.suptitle('Zeroing Irrelevant Latent Varialbes')
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(x_clean[i].reshape(28, 28), vmin=mn, vmax=mx, cmap='gray')

    plt.figure(figsize=[15, 3])
    plt.suptitle('Zeroing Relevant Latent Varialbes')
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(x_dirty[i].reshape(28, 28), vmin=mn, vmax=mx, cmap='gray')
        
    plt.figure(figsize=[15, 3])
    plt.suptitle('Zeroing Both Latent Variables')
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(x_empty[i].reshape(28, 28), vmin=mn, vmax=mx, cmap='gray')
        
def plot_sweeps_mnist(decoder):
    for option in [0, 1]:
        n = 30
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                if option==0:
                    z_sample = np.array([[xi, yi, 0, 0]])
                else:
                    z_sample = np.array([[0, 0, xi, yi]])
                
                x_decoded = decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        start_range = digit_size // 2
        end_range = n * digit_size + start_range + 1
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap='Greys_r')


def original_and_reconstructed_images(vae, n=4):
    batch_files = np.take(all_files, range(n), mode='wrap')
    celeb_images = helper.get_batch(batch_files, 28, 28, 'RGB') 
    plt.figure()
    plt.imshow(helper.images_square_grid(celeb_images, 'RGB'))    
    
    celeb_images = vae.predict(celeb_images)
    plt.figure()
    
    

def get_vae(original_dim=4, intermediate_dim=3, latent_dim=2, epochs=500, batch_size=64):
    input_shape = (original_dim, )
    
    inputs = tf.keras.layers.Input(shape=input_shape, name='encoder_input')
    x = tf.keras.layers.Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = tf.keras.models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
#     encoder.summary()
    # tf.keras.utils.plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = tf.keras.layers.Input(shape=(latent_dim,), name='z_sampling')
    x = tf.keras.layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = tf.keras.layers.Dense(original_dim)(x)

    # instantiate decoder model
    decoder = tf.keras.models.Model(latent_inputs, outputs, name='decoder')
    # decoder.summary()
    # tf.keras.utils.plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = tf.keras.models.Model(inputs, outputs, name='vae_mlp')

    reconstruction_loss = tf.keras.losses.mse(inputs, outputs)
    reconstruction_loss *= original_dim

    kl_loss = 1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var)
    kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    
    return vae, encoder, decoder

def get_cvae(original_dim=4, intermediate_dim=3, latent_dim=2, epochs=500, batch_size=64):
    input_shape = (original_dim, )
    
    # inputs = tf.keras.layers.Input(shape=input_shape, name='encoder_input')
    tg_inputs = tf.keras.layers.Input(shape=input_shape, name='tg_input')
    bg_inputs = tf.keras.layers.Input(shape=input_shape, name='bg_input')

    z_h_layer = tf.keras.layers.Dense(intermediate_dim, activation='relu')
    z_mean_layer = tf.keras.layers.Dense(latent_dim, name='z_mean')
    z_log_var_layer = tf.keras.layers.Dense(latent_dim, name='z_log_var')
    z_layer = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')
    # z_encoder = tf.keras.models.Model(inputs, [z_mean, z_log_var, z], name='z_encoder')

    def z_encoder(inputs):
        z_h = z_h_layer(inputs)
        z_mean = z_mean_layer(z_h)
        z_log_var = z_log_var_layer(z_h)
        z = z_layer([z_mean, z_log_var])
        return z_mean, z_log_var, z

    s_h_layer = tf.keras.layers.Dense(intermediate_dim, activation='relu')
    s_mean_layer = tf.keras.layers.Dense(latent_dim, name='s_mean')
    s_log_var_layer = tf.keras.layers.Dense(latent_dim, name='s_log_var')
    s_layer = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='s')

    def s_encoder(inputs):
        s_h = s_h_layer(inputs)
        s_mean = s_mean_layer(s_h)
        s_log_var = s_log_var_layer(s_h)
        s = s_layer([s_mean, s_log_var])
        return s_mean, s_log_var, s

    # s_h = tf.keras.layers.Dense(intermediate_dim, activation='relu')(inputs)
    # s_mean = tf.keras.layers.Dense(latent_dim, name='s_mean')(s_h)
    # s_log_var = tf.keras.layers.Dense(latent_dim, name='s_log_var')(s_h)
    # s = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='s')([s_mean, s_log_var])
    # s_encoder = tf.keras.models.Model(inputs, [s_mean, s_log_var, s], name='s_encoder')

    tg_z_mean, tg_z_log_var, tg_z = z_encoder(tg_inputs)
    tg_s_mean, tg_s_log_var, tg_s = s_encoder(tg_inputs)
    bg_s_mean, bg_s_log_var, bg_s = s_encoder(bg_inputs)

    z_encoder = tf.keras.models.Model(tg_inputs, [tg_z_mean, tg_z_log_var, tg_z], name='z_encoder')
    s_encoder = tf.keras.models.Model(tg_inputs, [tg_s_mean, tg_s_log_var, tg_s], name='s_encoder')
    s_encoder2 = tf.keras.models.Model(bg_inputs, [bg_s_mean, bg_s_log_var, bg_s], name='s_encoder2')

    # build decoder model
    cvae_latent_inputs = tf.keras.layers.Input(shape=(2 * latent_dim,), name='sampled')
    cvae_h = tf.keras.layers.Dense(intermediate_dim, activation='relu')(cvae_latent_inputs)
    cvae_outputs = tf.keras.layers.Dense(original_dim)(cvae_h)

    cvae_decoder = tf.keras.models.Model(inputs=cvae_latent_inputs, outputs=cvae_outputs, name='decoder')

    # decoder.summary()

    def zeros_like(x):
        return tf.zeros_like(x)

    tg_outputs = cvae_decoder(tf.keras.layers.concatenate([tg_z, tg_s], -1))
    zeros = tf.keras.layers.Lambda(zeros_like)(tg_z)
    bg_outputs = cvae_decoder(tf.keras.layers.concatenate([zeros, bg_s], -1))
    fg_outputs = cvae_decoder(tf.keras.layers.concatenate([tg_z, zeros], -1))

    cvae = tf.keras.models.Model(inputs=[tg_inputs, bg_inputs], 
                                 outputs=[tg_outputs, bg_outputs], 
                                 name='contrastive_vae')

    cvae_fg = tf.keras.models.Model(inputs=tg_inputs, 
                                    outputs=fg_outputs, 
                                    name='contrastive_vae_fg')

    reconstruction_loss = tf.keras.losses.mse(tg_inputs, tg_outputs)
    reconstruction_loss += tf.keras.losses.mse(bg_inputs, bg_outputs)
    reconstruction_loss *= original_dim


    kl_loss = 1 + tg_z_log_var - tf.keras.backend.square(tg_z_mean) - tf.keras.backend.exp(tg_z_log_var)
    kl_loss += 1 + tg_s_log_var - tf.keras.backend.square(tg_s_mean) - tf.keras.backend.exp(tg_s_log_var)
    kl_loss += 1 + bg_s_log_var - tf.keras.backend.square(bg_s_mean) - tf.keras.backend.exp(bg_s_log_var)
    kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    cvae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)
    cvae.add_loss(cvae_loss)
    cvae.compile(optimizer='adam')
    
    return cvae, cvae_fg, z_encoder, s_encoder, cvae_decoder

def get_cvae_no_bias(input_dim=4, intermediate_dim=3, latent_dim=2):
    original_dim = input_dim
    input_shape = (original_dim, )
    
    # inputs = tf.keras.layers.Input(shape=input_shape, name='encoder_input')
    tg_inputs = tf.keras.layers.Input(shape=input_shape, name='tg_input')
    bg_inputs = tf.keras.layers.Input(shape=input_shape, name='bg_input')

    z_h_layer = tf.keras.layers.Dense(intermediate_dim, activation='relu', use_bias=False)
    z_mean_layer = tf.keras.layers.Dense(latent_dim, name='z_mean', use_bias=False)
    z_log_var_layer = tf.keras.layers.Dense(latent_dim, name='z_log_var', use_bias=False)
    z_layer = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')
    # z_encoder = tf.keras.models.Model(inputs, [z_mean, z_log_var, z], name='z_encoder')

    def z_encoder(inputs):
        z_h = z_h_layer(inputs)
        z_mean = z_mean_layer(z_h)
        z_log_var = z_log_var_layer(z_h)
        z = z_layer([z_mean, z_log_var])
        return z_mean, z_log_var, z

    s_h_layer = tf.keras.layers.Dense(intermediate_dim, activation='relu', use_bias=False)
    s_mean_layer = tf.keras.layers.Dense(latent_dim, name='s_mean', use_bias=False)
    s_log_var_layer = tf.keras.layers.Dense(latent_dim, name='s_log_var', use_bias=False)
    s_layer = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='s')

    def s_encoder(inputs):
        s_h = s_h_layer(inputs)
        s_mean = s_mean_layer(s_h)
        s_log_var = s_log_var_layer(s_h)
        s = s_layer([s_mean, s_log_var])
        return s_mean, s_log_var, s

    # s_h = tf.keras.layers.Dense(intermediate_dim, activation='relu')(inputs)
    # s_mean = tf.keras.layers.Dense(latent_dim, name='s_mean')(s_h)
    # s_log_var = tf.keras.layers.Dense(latent_dim, name='s_log_var')(s_h)
    # s = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='s')([s_mean, s_log_var])
    # s_encoder = tf.keras.models.Model(inputs, [s_mean, s_log_var, s], name='s_encoder')

    tg_z_mean, tg_z_log_var, tg_z = z_encoder(tg_inputs)
    tg_s_mean, tg_s_log_var, tg_s = s_encoder(tg_inputs)
    bg_s_mean, bg_s_log_var, bg_s = s_encoder(bg_inputs)

    z_encoder = tf.keras.models.Model(tg_inputs, [tg_z_mean, tg_z_log_var, tg_z], name='z_encoder')
    s_encoder = tf.keras.models.Model(tg_inputs, [tg_s_mean, tg_s_log_var, tg_s], name='s_encoder')
    s_encoder2 = tf.keras.models.Model(bg_inputs, [bg_s_mean, bg_s_log_var, bg_s], name='s_encoder2')

    # build decoder model
    cvae_latent_inputs = tf.keras.layers.Input(shape=(2 * latent_dim,), name='sampled')
    cvae_h = tf.keras.layers.Dense(intermediate_dim, activation='relu', use_bias=False)(cvae_latent_inputs)
    cvae_outputs = tf.keras.layers.Dense(original_dim, use_bias=False)(cvae_h)

    cvae_decoder = tf.keras.models.Model(inputs=cvae_latent_inputs, outputs=cvae_outputs, name='decoder')

    # decoder.summary()

    def zeros_like(x):
        return tf.zeros_like(x)

    tg_outputs = cvae_decoder(tf.keras.layers.concatenate([tg_z, tg_s], -1))
    zeros = tf.keras.layers.Lambda(zeros_like)(tg_z)
    bg_outputs = cvae_decoder(tf.keras.layers.concatenate([zeros, bg_s], -1))
    fg_outputs = cvae_decoder(tf.keras.layers.concatenate([tg_z, zeros], -1))

    cvae = tf.keras.models.Model(inputs=[tg_inputs, bg_inputs], 
                                 outputs=[tg_outputs, bg_outputs], 
                                 name='contrastive_vae')

    cvae_fg = tf.keras.models.Model(inputs=tg_inputs, 
                                    outputs=fg_outputs, 
                                    name='contrastive_vae_fg')

    reconstruction_loss = tf.keras.losses.mse(tg_inputs, tg_outputs)
    reconstruction_loss += tf.keras.losses.mse(bg_inputs, bg_outputs)
    reconstruction_loss *= original_dim


    kl_loss = 1 + tg_z_log_var - tf.keras.backend.square(tg_z_mean) - tf.keras.backend.exp(tg_z_log_var)
    kl_loss += 1 + tg_s_log_var - tf.keras.backend.square(tg_s_mean) - tf.keras.backend.exp(tg_s_log_var)
    kl_loss += 1 + bg_s_log_var - tf.keras.backend.square(bg_s_mean) - tf.keras.backend.exp(bg_s_log_var)
    kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    cvae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)
    cvae.add_loss(cvae_loss)
    cvae.compile(optimizer='adam')
    
    return cvae, cvae_fg, z_encoder, s_encoder, cvae_decoder