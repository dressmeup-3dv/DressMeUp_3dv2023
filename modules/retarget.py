
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from torch import optim
from data_loader.test_loader import TestDataset
import trimesh
from psbody.mesh import Mesh
from utils.interpenetration import remove_interpenetration_fast
from utils.utils import *
import onnxruntime as ort
import pymeshlab
import potpourri3d as pp3d
from scipy.sparse.linalg import lsqr, cg, eigsh, inv
from scipy import sparse

ort.set_default_logger_severity(3)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class retarget(LightningModule):
    def __init__( self, opt, wandb_logger=None):
        super(retarget, self).__init__()        
       
        self.opt = opt
        self.exp_name = opt.experiment
        self.result_dir = f'./results/'
        self.set_device = opt.device
        os.makedirs(self.result_dir, exist_ok = True)
        
        self.test_data = TestDataset(opt)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=1, shuffle=False)
       


    def forward(self, batch):
        pose_vert = batch['target_verts']

        ort_sess = ort.InferenceSession('checkpoints/pose_enc.onnx',revision="fp32",providers=["CUDAExecutionProvider"])
        ort_inp = {ort_sess.get_inputs()[0].name: to_numpy(pose_vert)}
        pose_code_ort = ort_sess.run(None,ort_inp)
        pose_code_ort = np.asarray(pose_code_ort)    #[1,1,128]

        gar_verts = batch['gar_verts']
        ort_sess = ort.InferenceSession('checkpoints/gar_enc.onnx',revision="fp32",providers=["CUDAExecutionProvider"])
        ort_inp = {ort_sess.get_inputs()[0].name: to_numpy(gar_verts)}
        gar_code_ort = ort_sess.run(None,ort_inp)
        gar_code_ort = np.asarray(gar_code_ort)    #[1,1,128]
        
        

        pos = batch['model_input'].type(torch.float32)
        inp_verts = batch['gt']
        normals = batch['gt_normals'].numpy()
        
        num_points = pos.shape[1]
        
        pose_code = np.repeat(pose_code_ort,[num_points],axis=1) #(1, 2468, 128)
        gar_code = np.repeat(gar_code_ort,[num_points],axis=1) #(1, 2468, 128)
 
        forward_features = np.concatenate([pos.cpu().numpy(), pose_code,gar_code], axis=-1) #[1,2468,483]
        forward_features = np.transpose(forward_features, (0,2,1))
        

        
        ort_sess = ort.InferenceSession('checkpoints/mlp_enc.onnx',revision="fp32",providers=["CUDAExecutionProvider"])
        ort_inp = {ort_sess.get_inputs()[0].name: (forward_features),ort_sess.get_inputs()[1].name:(pose_code_ort)}
        mlp_out_ort = ort_sess.run(None,ort_inp)
        mlp_out_ort = np.asarray(mlp_out_ort)    
        mlp_out_ort = np.transpose(mlp_out_ort[0],(0,2,1))
        mlp_out_ort = torch.Tensor(mlp_out_ort)
        batch['output'] =  inp_verts + mlp_out_ort * normals 
    
        return batch




    def training_step(self, batch):
        pass
    
    def training_step_end(self, batch):
        pass


    def validation_step(self, batch, batch_idx):
        pass
    def validation_epoch_end(self, outputs):
        pass


    def test_dataloader(self) -> DataLoader:
        return self.test_loader


    def training_epoch_end(self, outputs):
        pass
                    

    def test_step(self,batch,batch_idx):
        pass
                
    def solve_laplacian(self,L, delta, vertices):
        updated_verts = np.zeros((vertices.shape))
        for i in range(3):
            updated_verts[:, i] = lsqr(L, delta[:, i])[0]
            # updated_verts[:, i] = inv(L) @ delta[:, i]
            
        return updated_verts
    


    def get_lmatrix(self,vertices, L, anchorsIdx):
        L= L.toarray() 
        num_verts = vertices.shape[0]
        
        k = len(anchorsIdx)
        amatrix = np.zeros((k,num_verts))
        for i in range(k):
            amatrix[i, anchorsIdx[i]] = 1.0
        
        L = np.vstack((L,amatrix))     
        L = sparse.coo_matrix(L, shape=(L.shape)).tocsr()
        return L

    
    def inference(self):
        for idx, batch in enumerate(self.test_loader):
            with torch.no_grad():
                var = self(batch) 
                pred_3d = var['output'][0].detach().cpu().numpy()
                name = var['target_name'][0]
                target_verts = var['target_verts'][0].detach().cpu().numpy()
                target_faces = var['target_faces'][0].detach().cpu().numpy()
                gar_name = var['gar_name'][0] 
                gar_verts = var['gar_verts'][0].detach().cpu().numpy() 
                faces = var['gar_faces'][0].detach().cpu().numpy()
                
                
                #detail transfer
                num_verts = pred_3d.shape[0]
                num_sample = int(num_verts*0.3)

                edit_indices = np.random.randint(0,num_verts, num_sample)
                L_dense = pp3d.cotan_laplacian(gar_verts, faces,denom_eps=1e-10)
                L_dense = self.get_lmatrix(gar_verts, L_dense, edit_indices)

                delta =  L_dense @ gar_verts
                #update delta
                for i in range(len(edit_indices)):
                    delta[i+num_verts, :] = pred_3d[edit_indices[i]]

                updated_verts = self.solve_laplacian(L_dense, delta, pred_3d)
                
                
                pred_mesh = Mesh(updated_verts,faces)
                body_mesh = Mesh(target_verts,target_faces)
                
                modified_verts = remove_interpenetration_fast(pred_mesh,body_mesh)
                # modified_verts = updated_verts
                
                if(self.opt.garment_name ):
                    garment_mesh_path = os.path.join(self.opt.garment_path,'mesh',f'{self.opt.garment_name}.obj')       

                else:
                    garment_mesh_path = os.path.join(self.opt.garment_path,'mesh',f'{gar_name}.obj')  


                output_path = os.path.join(self.result_dir,f'{gar_name}_{name}.obj')
                
                
                if(self.opt.save_texture): 
                    garment_mesh = trimesh.load(garment_mesh_path, process=False, maintain_order=True)  
                    garment_mesh.vertices = modified_verts
                    garment_mesh.export('results/tmp.obj')



                    original_mesh = load_obj(garment_mesh_path)
                    reposed_mesh = load_obj('results/tmp.obj', quad=False)

                    
                    write_obj(original_mesh,original_mesh,reposed_mesh,output_path)

                else: 
                    garment_mesh = trimesh.load(garment_mesh_path,process=False, maintain_order=True)  
                    garment_mesh.vertices = modified_verts
                    garment_mesh.export(output_path)
                    

                mesh = pymeshlab.MeshSet()
                mesh.load_new_mesh(output_path) 
                mesh.set_texture_per_mesh(textname="images/texture.png")
                mesh.save_current_mesh(output_path)
                
