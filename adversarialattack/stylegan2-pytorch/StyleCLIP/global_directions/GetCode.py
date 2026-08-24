


import os
#os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Use CPU only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # suppress TF warnings
import pickle
import numpy as np
from dnnlib import tflib   
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ TensorFlow GPU memory growth enabled")
    except RuntimeError as e:
        print("⚠️ Could not set memory growth:", e)

import argparse

def LoadModel(dataset_name):
    # Initialize TensorFlow.
    tflib.init_tf()
    model_path='./model/'
    model_name=dataset_name+'.pkl'
    
    tmp=os.path.join(model_path,model_name)
    with open(tmp, 'rb') as f:
        _, _, Gs = pickle.load(f)
    return Gs

def lerp(a,b,t):
     return a + (b - a) * t

#stylegan-ada
def SelectName(layer_name,suffix):
    if suffix==None:
        tmp1='add:0' in layer_name 
        tmp2='shape=(?,' in layer_name
        tmp4='G_synthesis_1' in layer_name
        tmp= tmp1 and tmp2 and tmp4  
    else:
        tmp1=('/Conv0_up'+suffix) in layer_name 
        tmp2=('/Conv1'+suffix) in layer_name 
        tmp3=('4x4/Conv'+suffix) in layer_name 
        tmp4='G_synthesis_1' in layer_name
        tmp5=('/ToRGB'+suffix) in layer_name
        tmp= (tmp1 or tmp2 or tmp3 or tmp5) and tmp4 
    return tmp

def GetSNames(sess, suffix=None):
    op = sess.graph.get_operations()
    layers = [m.values() for m in op]
    
    select_layers=[]
    for layer in layers:
        layer_name = str(layer)
        if SelectName(layer_name, suffix):
            select_layers.append(layer[0])
    return select_layers

def SelectName2(layer_name):
    tmp1='mod_bias' in layer_name 
    tmp2='mod_weight' in layer_name
    tmp3='ToRGB' in layer_name 
    
    tmp= (tmp1 or tmp2) and (not tmp3) 
    return tmp

def GetKName(Gs):
    
    layers=[var for name, var in Gs.components.synthesis.vars.items()]
    
    select_layers=[]
    for layer in layers:
        layer_name=str(layer)
        if SelectName2(layer_name):
            select_layers.append(layer)
    return select_layers

def GetCode(Gs,random_state,num_img,num_once,dataset_name):
    rnd = np.random.RandomState(random_state)  #5
    
    truncation_psi=0.7
    truncation_cutoff=8
    
    dlatent_avg=Gs.get_var('dlatent_avg')
    
    dlatents=np.zeros((num_img,512),dtype='float32')
    for i in range(int(num_img/num_once)):
        #src_latents =  rnd.randn(num_once, Gs.input_shape[1])
        src_latents = rnd.randn(num_once, Gs.input_shape[1]).astype('float32')
        src_dlatents = Gs.components.mapping.run(src_latents, None) # [seed, layer, component]
        
        # Apply truncation trick.
        if truncation_psi is not None and truncation_cutoff is not None:
                layer_idx = np.arange(src_dlatents.shape[1])[np.newaxis, :, np.newaxis]
                ones = np.ones(layer_idx.shape, dtype=np.float32)
                coefs = np.where(layer_idx < truncation_cutoff, truncation_psi * ones, ones)
                src_dlatents_np=lerp(dlatent_avg, src_dlatents, coefs)
                src_dlatents=src_dlatents_np[:,0,:].astype('float32')
                dlatents[(i*num_once):((i+1)*num_once),:]=src_dlatents
    print('get all z and w')
    
    tmp='./npy/'+dataset_name+'/W'
    np.save(tmp,dlatents)

    
def GetImg(Gs,num_img,num_once,dataset_name,save_name='images'):
    print('Generate Image')
    tmp='./npy/'+dataset_name+'/W.npy'
    dlatents=np.load(tmp) 
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    
    all_images=[]
    for i in range(int(num_img/num_once)):
        print(i)
        images=[]
        for k in range(num_once):
            tmp=dlatents[i*num_once+k]
            tmp=tmp[None,None,:]
            tmp=np.tile(tmp,(1,Gs.components.synthesis.input_shape[1],1))
            image2= Gs.components.synthesis.run(tmp, randomize_noise=False, output_transform=fmt)
            images.append(image2)
            
        images=np.concatenate(images)
        
        all_images.append(images)
        
    all_images=np.concatenate(all_images)
    
    tmp='./npy/'+dataset_name+'/'+save_name
    np.save(tmp,all_images)

'''
def GetS(dataset_name, num_img):
    print('Generating S latent codes...')
    
    tmp = f'./npy/{dataset_name}/W.npy'
    dlatents = np.load(tmp)[:num_img]  # W latent codes
    
    # Load Gs model
    Gs = LoadModel(dataset_name)

    # Prepare dlatents for synthesis
    dlatents_exp = dlatents[:, None, :]
    dlatents_exp = np.tile(dlatents_exp, (1, Gs.components.synthesis.input_shape[1], 1))

    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

    # Instead of using dlatents_in placeholder, use synthesis.run
    all_s = []
    layer_names = []

    for i in range(0, num_img, 8):  # batch size 8
        batch = dlatents_exp[i:i+8]
        images = Gs.components.synthesis.run(batch, randomize_noise=False, output_transform=None)
        
        # Extract mod_weights for each layer
        mod_weights = [
            var.eval() for name, var in Gs.components.synthesis.vars.items() 
            if 'mod_weight' in name
        ]
        all_s.append(mod_weights)
        layer_names = [name for name, var in Gs.components.synthesis.vars.items() if 'mod_weight' in name]

    # Concatenate all batches
    # all_s will be list of list: flatten it
    all_s_flat = [layer for batch in all_s for layer in batch]

    return [layer_names, all_s_flat]
'''
def GetS(dataset_name, num_img, batch_size=8):
    print('Generating S latent codes incrementally...')

    tmp = f'./npy/{dataset_name}/W.npy'
    dlatents = np.load(tmp)[:num_img]  # W latent codes

    # Load Gs model
    Gs = LoadModel(dataset_name)

    npy_dir = f'./npy/{dataset_name}'
    os.makedirs(npy_dir, exist_ok=True)
    s_path = f'{npy_dir}/S.pkl'

    layer_names = [name for name, var in Gs.components.synthesis.vars.items() if 'mod_weight' in name]

    # Open file to save incrementally
    with open(s_path, 'wb') as f:
        pickle.dump(layer_names, f)  # save layer names first

    # Process in batches
    for i in range(0, num_img, batch_size):
        batch = dlatents[i:i+batch_size]
        batch_exp = batch[:, None, :]
        batch_exp = np.tile(batch_exp, (1, Gs.components.synthesis.input_shape[1], 1))

        images = Gs.components.synthesis.run(batch_exp, randomize_noise=False, output_transform=None)

        # Extract mod_weights
        mod_weights = [var.eval() for name, var in Gs.components.synthesis.vars.items() if 'mod_weight' in name]

        # Save batch to file incrementally
        with open(s_path, 'ab') as f:
            pickle.dump(mod_weights, f)

        print(f"Saved batch {i} - {i+batch_size}")

    print("✅ Done!")

def Compute_S_mean_std_from_disk(s_path):
    print("Computing S mean/std from disk (per-layer)...")

    with open(s_path, "rb") as f:
        layer_names = pickle.load(f)

        num_layers = len(layer_names)

        sums = [None] * num_layers
        sq_sums = [None] * num_layers
        count = 0

        while True:
            try:
                # batch is: list of [512, 512] arrays, one per layer
                batch = pickle.load(f)

                for i, layer_mat in enumerate(batch):
                    # layer_mat shape: [512, 512]
                    if sums[i] is None:
                        sums[i] = layer_mat.copy()
                        sq_sums[i] = layer_mat ** 2
                    else:
                        sums[i] += layer_mat
                        sq_sums[i] += layer_mat ** 2

                count += 1

            except EOFError:
                break

    means = []
    stds = []

    for i in range(num_layers):
        mean = sums[i] / count
        var = sq_sums[i] / count - mean ** 2
        std = np.sqrt(np.maximum(var, 1e-8))

        means.append(mean)
        stds.append(std)

    return layer_names, means, stds


def convert_images_to_uint8(images, drange=[-1,1], nchw_to_nhwc=False):
    """Convert a minibatch of images from float32 to uint8 with configurable dynamic range.
    Can be used as an output transformation for Network.run().
    """
    if nchw_to_nhwc:
        images = np.transpose(images, [0, 2, 3, 1])
    
    scale = 255 / (drange[1] - drange[0])
    images = images * scale + (0.5 - drange[0] * scale)
    
    np.clip(images, 0, 255, out=images)
    images=images.astype('uint8')
    return images


def GetCodeMS(dlatents):
        m=[]
        std=[]
        for i in range(len(dlatents)):
            tmp= dlatents[i] 
            tmp_mean=tmp.mean(axis=0)
            tmp_std=tmp.std(axis=0)
            m.append(tmp_mean)
            std.append(tmp_std)
        return m,std



if __name__ == "__main__":
    import argparse
    import os
    import numpy as np
    import pickle
    from dnnlib import tflib

    parser = argparse.ArgumentParser(description='Generate StyleCLIP latent codes.')
    parser.add_argument('--dataset_name', type=str, default='ffhq')
    parser.add_argument('--code_type', choices=['w','s','s_mean_std'], default='w')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    random_state = 5
    num_img = 100_000
    num_once = 1   # <<< safer batch size

    # Force TF GPU memory growth
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Make sure model folder exists
    os.makedirs('./model', exist_ok=True)
    model_path = './model/' + dataset_name + '.pkl'

    # Download StyleGAN2 model if not exists
    if not os.path.isfile(model_path):
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/'
        name = f'stylegan2-{dataset_name}-config-f.pkl'
        os.system(f'wget {url}{name} -P ./model/')
        os.rename(f'./model/{name}', model_path)

    # Make sure npy folder exists
    npy_dir = f'./npy/{dataset_name}'
    os.makedirs(npy_dir, exist_ok=True)

    # Load Gs model
    Gs = LoadModel(dataset_name)

    if args.code_type == 'w':
        print("Generating W latent codes...")
        GetCode(Gs, random_state, num_img, num_once, dataset_name)
    
    #elif args.code_type == 's':
        #print("Generating S latent codes...")
        #save_tmp = GetS(dataset_name, num_img=2_000)
        #s_path = f'{npy_dir}/S.pkl'
        #with open(s_path, "wb") as fp:
            #pickle.dump(save_tmp, fp)
    
    elif args.code_type == 's':
        print("Generating S latent codes...")
        GetS(dataset_name, num_img=2_000, batch_size=1)


    #elif args.code_type == 's_mean_std':
        #print("Generating S mean/std...")
        #save_tmp = GetS(dataset_name, num_img=num_img)
        #dlatents = save_tmp[1]
        #m, std = GetCodeMS(dlatents)
        #save_tmp = [m, std]
        #tmp_path = f'{npy_dir}/S_mean_std'
        #with open(tmp_path, "wb") as fp:
            #pickle.dump(save_tmp, fp)

    elif args.code_type == 's_mean_std':
        print("Generating S mean/std from disk...")

        s_path = f'{npy_dir}/S.pkl'
        layer_names, mean, std = Compute_S_mean_std_from_disk(s_path)

        with open(f'{npy_dir}/S_mean_std.pkl', 'wb') as f:
            pickle.dump([layer_names, mean, std], f)

    print("✅ Done!")


    
    
    
    
