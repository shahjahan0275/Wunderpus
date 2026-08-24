
import os
import os.path
import pickle
import numpy as np
import tensorflow as tf
from dnnlib import tflib
from global_directions.utils.visualizer import HtmlPageVisualizer


def Vis(bname,suffix,out,rownames=None,colnames=None):
    num_images=out.shape[0]
    step=out.shape[1]
    
    if colnames is None:
        colnames=[f'Step {i:02d}' for i in range(1, step + 1)]
    if rownames is None:
        rownames=[str(i) for i in range(num_images)]
    
    
    visualizer = HtmlPageVisualizer(
      num_rows=num_images, num_cols=step + 1, viz_size=256)
    visualizer.set_headers(
      ['Name'] +colnames)
    
    for i in range(num_images):
        visualizer.set_cell(i, 0, text=rownames[i])
    
    for i in range(num_images):
        for k in range(step):
            image=out[i,k,:,:,:]
            visualizer.set_cell(i, 1+k, image=image)
    
    # Save results.
    visualizer.save(f'./html/'+bname+'_'+suffix+'.html')


def LoadData(path):
    """
    Loads StyleCLIP latent statistics from PKL-based pipeline.
    Compatible with:
      - S.pkl
      - W.npy
      - S_mean_std.pkl
    """

    import os
    import pickle
    import numpy as np

    # ---- paths ----
    S_pkl_path = os.path.join(path, "S.pkl")
    W_path = os.path.join(path, "W.npy")
    MS_path = os.path.join(path, "S_mean_std.pkl")

    if not os.path.exists(S_pkl_path):
        raise FileNotFoundError(f"Missing: {S_pkl_path}")
    if not os.path.exists(W_path):
        raise FileNotFoundError(f"Missing: {W_path}")
    if not os.path.exists(MS_path):
        raise FileNotFoundError(f"Missing: {MS_path}")

    # ---- load W ----
    W = np.load(W_path, allow_pickle=True)

    # ---- load S (streamed PKL) ----
    with open(S_pkl_path, "rb") as f:
        s_names = pickle.load(f)  # first object = layer names
        all_s = []
        while True:
            try:
                batch = pickle.load(f)
                all_s.extend(batch)
            except EOFError:
                break

    dlatents = all_s  # StyleCLIP naming

    # ---- build layer indices ----
    mindexs = []
    pindexs = []
    for i, name in enumerate(s_names):
        if 'ToRGB' in name:
            pindexs.append(i)
        else:
            mindexs.append(i)

    # ---- load mean/std ----
    with open(MS_path, "rb") as f:
        data = pickle.load(f)
        
        # Check if data is a list/tuple and has at least 2 items
        if isinstance(data, (list, tuple)):
            if len(data) >= 2:
                # This uses the first two items and ignores any 3rd or 4th item
                code_mean, code_std = data[0], data[1]
            else:
                code_mean = data[0]
                code_std = data[0] # Fallback if only one item exists
        else:
            # If the file was saved as multiple objects, not a list
            # We use *extra to catch any remaining values safely
            try:
                # Seek back to start if we already read it
                f.seek(0) 
                code_mean, code_std, *extra = pickle.load(f)
            except:
                code_mean, code_std = data, data # Last resort

    return dlatents, s_names, mindexs, pindexs, code_mean, code_std



def LoadModel(model_path,model_name):
    # Initialize TensorFlow.
    tflib.init_tf()
    tmp=os.path.join(model_path,model_name)
    with open(tmp, 'rb') as f:
        _, _, Gs = pickle.load(f)
    Gs.print_layers()
    return Gs

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


def convert_images_from_uint8(images, drange=[-1,1], nhwc_to_nchw=False):
    """Convert a minibatch of images from uint8 to float32 with configurable dynamic range.
    Can be used as an input transformation for Network.run().
    """
    if nhwc_to_nchw:
        images=np.rollaxis(images, 3, 1)
    return images/ 255 *(drange[1] - drange[0])+ drange[0]

class Manipulator():
    def __init__(self, dataset_name='ffhq', sess=None, load_tf=False): 
        # 1. Path Setup
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.img_path = os.path.join(BASE_DIR, "npy", dataset_name)
        self.model_path = os.path.join(BASE_DIR, "model")

        self.dataset_name = dataset_name
        self.model_name = dataset_name + '.pkl'
        
        self.alpha = [0] 
        self.num_images = 10
        self.img_index = 0  
        self.viz_size = 256
        self.manipulate_layers = None 
        
        # 2. Load the latent data (Required for both PyTorch and TF)
        self.dlatents, self.s_names, self.mindexs, self.pindexs, self.code_mean, self.code_std = LoadData(self.img_path)
        self.num_layers = len(self.dlatents)
        self.Vis = Vis
        self.noise_constant = {}

        # 3. ONLY load TensorFlow if explicitly requested
        if load_tf:
            print(f"🔄 Initializing Full TensorFlow Manipulator for {dataset_name}...")
            self.sess = sess or tf.InteractiveSession()
            init = tf.global_variables_initializer()
            self.sess.run(init)
            
            self.Gs = LoadModel(self.model_path, self.model_name)
            
            # --- START TENSORFLOW SPECIFIC LOGIC ---
            prefix = 'G_synthesis'
            for name in self.s_names:
                if 'G_synthesis' in name:
                    prefix = name.split('/')[0]
                    break

            for i in range(len(self.s_names)):
                name_parts = self.s_names[i].split('/')
                if 'ToRGB' not in name_parts:
                    size = None
                    for part in name_parts:
                        if 'x' in part and part.split('x')[0].isdigit():
                            size = int(part.split('x')[0])
                            break
                    if size is None: continue
                    target_tensor = self.s_names[i].rsplit('/', 1)[0] + '/random_normal:0'
                    tmp = (1, 1, size, size)
                    self.noise_constant[target_tensor] = np.random.random(tmp)
            
            latent_input_name = prefix + '/dlatents_in:0'
            tmp_dim = self.Gs.components.synthesis.input_shape[1]
            feed_dict = {latent_input_name: np.zeros([1, tmp_dim, 512])}
            const_shape_name = prefix + '/4x4/Const/Shape:0'
            
            all_ops = [op.name for op in tf.get_default_graph().get_operations()]
            if const_shape_name.split(':')[0] in all_ops:
                feed_dict[const_shape_name] = np.array([1, 18, 512], dtype=np.int32)
            
            names = list(self.noise_constant.keys())
            try:
                valid_names = [n for n in names if n.split(':')[0] in all_ops]
                output_values = tflib.run(valid_names, feed_dict)
                for i in range(len(valid_names)):
                    self.noise_constant[valid_names[i]] = output_values[i]
            except Exception as e:
                print(f"Warning: Noise initialization had issues: {e}")
            
            self.fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
            self.img_size = self.Gs.output_shape[-1]
            # --- END TENSORFLOW SPECIFIC LOGIC ---
        else:
            # Lite mode for PyTorch: Disable all TF/Gs variables
            print(f"✅ Initialized Lite Manipulator (No TensorFlow/MIG Conflict)")
            self.Gs = None
            self.sess = None
            self.img_size = 1024 # Standard StyleGAN2 size
    
    def GenerateImg(self,codes):
        

        num_images,step=codes[0].shape[:2]

            
        out=np.zeros((num_images,step,self.img_size,self.img_size,3),dtype='uint8')
        for i in range(num_images):
            for k in range(step):
                d={}
                for m in range(len(self.s_names)):
                    d[self.s_names[m]]=codes[m][i,k][None,:]  #need to change
                d['G_synthesis_1/4x4/Const/Shape:0']=np.array([1,18,  512], dtype=np.int32)
                d.update(self.noise_constant)
                img=tflib.run('G_synthesis_1/images_out:0', d)
                image=convert_images_to_uint8(img, nchw_to_nhwc=True)
                out[i,k,:,:,:]=image[0]
        return out
    
    
    
    def MSCode(self,dlatent_tmp,boundary_tmp):
        
        step=len(self.alpha)
        dlatent_tmp1=[tmp.reshape((self.num_images,-1)) for tmp in dlatent_tmp]
        dlatent_tmp2=[np.tile(tmp[:,None],(1,step,1)) for tmp in dlatent_tmp1] # (10, 7, 512)

        l=np.array(self.alpha)
        l=l.reshape(
                    [step if axis == 1 else 1 for axis in range(dlatent_tmp2[0].ndim)])
        
        if type(self.manipulate_layers)==int:
            tmp=[self.manipulate_layers]
        elif type(self.manipulate_layers)==list:
            tmp=self.manipulate_layers
        elif self.manipulate_layers is None:
            tmp=np.arange(len(boundary_tmp))
        else:
            raise ValueError('manipulate_layers is wrong')
            
        for i in tmp:
            dlatent_tmp2[i]+=l*boundary_tmp[i]
        
        codes=[]
        for i in range(len(dlatent_tmp2)):
            tmp=list(dlatent_tmp[i].shape)
            tmp.insert(1,step)
            codes.append(dlatent_tmp2[i].reshape(tmp))
        return codes
    
    
    def EditOne(self,bname,dlatent_tmp=None):
        if dlatent_tmp==None:
            dlatent_tmp=[tmp[self.img_index:(self.img_index+self.num_images)] for tmp in self.dlatents]
        
        boundary_tmp=[]
        for i in range(len(self.boundary)):
            tmp=self.boundary[i]
            if len(tmp)<=bname:
                boundary_tmp.append([])
            else:
                boundary_tmp.append(tmp[bname])
        
        codes=self.MSCode(dlatent_tmp,boundary_tmp)
            
        out=self.GenerateImg(codes)
        return codes,out
    
    def EditOneC(self,cindex,dlatent_tmp=None): 
        if dlatent_tmp==None:
            dlatent_tmp=[tmp[self.img_index:(self.img_index+self.num_images)] for tmp in self.dlatents]
        
        boundary_tmp=[[] for i in range(len(self.dlatents))]
        
        #'only manipulate 1 layer and one channel'
        assert len(self.manipulate_layers)==1 
        
        ml=self.manipulate_layers[0]
        tmp=dlatent_tmp[ml].shape[1] #ada
        tmp1=np.zeros(tmp)
        tmp1[cindex]=self.code_std[ml][cindex]  #1
        boundary_tmp[ml]=tmp1
        
        codes=self.MSCode(dlatent_tmp,boundary_tmp)
        out=self.GenerateImg(codes)
        return codes,out
    
        
    def W2S(self,dlatent_tmp):
        
        all_s = self.sess.run(
            self.s_names,
            feed_dict={'G_synthesis_1/dlatents_in:0': dlatent_tmp})
        return all_s
        
    


#%%
if __name__ == "__main__":
    
    
    M=Manipulator(dataset_name='ffhq')
    
    
    #%%
    M.alpha=[-5,0,5]
    M.num_images=20
    lindex,cindex=6,501
    
    M.manipulate_layers=[lindex]
    codes,out=M.EditOneC(cindex) #dlatent_tmp
    tmp=str(M.manipulate_layers)+'_'+str(cindex)
    M.Vis(tmp,'c',out)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




